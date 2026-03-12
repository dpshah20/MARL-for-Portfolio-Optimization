import os
import numpy as np
import pandas as pd
import torch
from collections import deque

from dataset.dataset_windows import build_windows_from_paths
from timeseries_models.encoders import CombinedEncoder
from rl_layer.actor_critic import MADDPG
from rl_layer.replay_buffer import ReplayBuffer
from rl_layer.rebalance_manager import HysteresisSelector
from rl_layer.execution import Portfolio
from rl_layer.reward_function import compute_reward_details


class TrainerRL:
    def __init__(self, cfg: dict, logger, ckpt_mgr, device: str = "cpu"):
        self.cfg = cfg
        self.device = device
        self.logger = logger
        self.ckpt_mgr = ckpt_mgr

        # ------------------------------
        # 1. Core Setup
        # ------------------------------
        self.tickers = cfg["tickers"]
        self.feature_cols = cfg["feature_cols"]
        self.window = cfg.get("window_length", 126)
        self.N_stocks = len(self.tickers)
        self.close_idx = self.feature_cols.index("Close") if "Close" in self.feature_cols else None

        # ------------------------------
        # 2. Encoder + Actor-Critic
        # ------------------------------
        enc_cfg = cfg.get("encoder", {})
        self.encoder = CombinedEncoder(
            input_dim=enc_cfg.get("input_dim", len(self.feature_cols)),
            W=enc_cfg.get("W", self.window),
            d_time=enc_cfg.get("d_time", 64),
            d_gnn=enc_cfg.get("d_gnn", 64),
            time_layers=enc_cfg.get("time_layers", 2),
            gnn_layers=enc_cfg.get("gnn_layers", 2)
        ).to(self.device)

        self.K = cfg.get("top_k", 10)
        self.agent = MADDPG(
            d_gnn=enc_cfg.get("d_gnn", 64),
            n_assets=len(self.tickers),
            device=self.device
        )

        self.replay = ReplayBuffer(capacity=cfg.get("replay_capacity", 200000))

        # Graph construction controls (important for larger universes like Nifty 100).
        graph_cfg = cfg.get("graph", {})
        self.graph_mode = graph_cfg.get("mode", "knn")
        self.graph_k = int(graph_cfg.get("k", 8))
        self.graph_corr_thr = float(graph_cfg.get("corr_threshold", 0.6))
        self.graph_abs_corr = bool(graph_cfg.get("absolute_corr", True))

        # ------------------------------
        # 3. Portfolio & Execution
        # ------------------------------
        self.portfolio = Portfolio(
            self.tickers,
            initial_nav=cfg.get("initial_nav", 1000.0)
        )

        self.selector = HysteresisSelector(
            self.tickers,
            k=self.K,
            hysteresis_days=cfg.get("hysteresis_days", 3)
        )

        # ------------------------------
        # 4. Reward & Rolling Metrics
        # ------------------------------
        self.ret_history = deque(maxlen=30)
        self.rolling_metrics = {"vol": 0.0, "cvar": 0.0, "mdd": 0.0}

        # ------------------------------
        # 5. Meta-RL Integration
        # ------------------------------
        self.meta_trainer = None
        self.meta_warmup_steps = cfg.get("meta_warmup_steps", 300)
        self.meta_transition_steps = int(cfg.get("meta_transition_steps", 50))
        self.meta_rho_min = float(cfg.get("meta_rho_min", 0.05))
        self.meta_rho_max = float(cfg.get("meta_rho_max", 0.30))

        macro_path = cfg.get("macros_weekly_path", "data/macros/combined_macros_weekly.csv")
        self._load_macros(macro_path)

        self.meta_cols = cfg.get("meta_input", {}).get("macros", [])
        if not self.meta_cols:
            raise ValueError("[Critical] No meta_input.macros found in params.yaml")

        # Meta state
        self.current_week_num = -1
        self.current_w = np.array([1.0, 0.0, 0.0, 0.0])
        self.current_rho = 0.05
        self.week_start_nav = self.portfolio.nav
        self.prev_week_stats = np.zeros(4)
        self.prev_action = np.zeros(5)
        self.prev_log_prob = None
        self.prev_entropy = None

        # RL transition state
        self.prev_state = None
        self.prev_action_weights = None
        self.prev_actor_hidden_in = None
        self.prev_nav = self.portfolio.nav
        self.actor_hidden = None

    def reset_recurrent_memory(self, clear_transition=False):
        self.actor_hidden = None
        self.prev_actor_hidden_in = None
        if clear_transition:
            self.prev_state = None
            self.prev_action_weights = None

    # ------------------------------------------------------------------
    # DATA LOADING
    # ------------------------------------------------------------------
    def _load_macros(self, path):
        if not os.path.exists(path):
            raise FileNotFoundError(f"[Critical] Macro file not found: {path}")

        df = pd.read_csv(path)
        df["Date"] = pd.to_datetime(df["Date"])
        self.macro_df = df.sort_values("Date").set_index("Date")
        self.logger.info(f"[TrainerRL] Loaded {len(df)} weekly macro records.")

    def attach_meta_trainer(self, trainer):
        self.meta_trainer = trainer

    # ------------------------------------------------------------------
    # MAIN DAILY STEP
    # ------------------------------------------------------------------
    def step_daily(
        self,
        X_day: np.ndarray,
        date_str: str,
        step_count: int,
        valid_asset_mask: np.ndarray = None,
        allow_learning: bool = True,
        phase: str = "train",
    ):
        date = pd.to_datetime(date_str)
        week_num = date.isocalendar()[1]

        # ---- Safeguard: Dimension Mismatch ----
        # If input has more assets than model expects, slice it.
        # If input has fewer, we have a problem (but usually it's more).
        N_input = X_day.shape[0]
        if N_input != self.N_stocks:
            if N_input > self.N_stocks:
                # self.logger.info(f"Slicing input from {N_input} to {self.N_stocks} assets.")
                X_day = X_day[:self.N_stocks]
                if valid_asset_mask is not None:
                    valid_asset_mask = np.asarray(valid_asset_mask[:self.N_stocks], dtype=bool)
            else:
                self.logger.error(f"Input assets {N_input} < Model assets {self.N_stocks}. Crash likely.")

        if valid_asset_mask is None:
            valid_mask = np.ones((X_day.shape[0],), dtype=bool)
        else:
            valid_mask = np.asarray(valid_asset_mask, dtype=bool).reshape(-1)
            if valid_mask.shape[0] != X_day.shape[0]:
                self.logger.error(
                    f"valid_asset_mask length {valid_mask.shape[0]} does not match assets {X_day.shape[0]}. "
                    "Falling back to all-true mask."
                )
                valid_mask = np.ones((X_day.shape[0],), dtype=bool)

        # ---- Encode state
        N = X_day.shape[0]
        A = self._build_adjacency(X_day, valid_mask=valid_mask)
        X_tensor = torch.tensor(X_day, dtype=torch.float32).unsqueeze(0).to(self.device)

        with torch.no_grad():
            z = self.encoder(X_tensor, A)            # (1, N, d)
            actor_hidden_in = self.actor_hidden
            scores_t, next_hidden = self.agent.actor(z, hidden=actor_hidden_in)
            scores = scores_t.squeeze(0)             # (N,)

        self.actor_hidden = next_hidden.detach()
        actor_hidden_in_detached = actor_hidden_in.detach() if actor_hidden_in is not None else None

        scores_np = scores.cpu().numpy()

        # ---- Weekly meta update
        if week_num != self.current_week_num:
            self._handle_weekly_update(
                date,
                date_str,
                step_count,
                allow_learning=allow_learning,
                phase=phase,
            )

        # ---- Reward from previous action
        if self.prev_state is not None:
            current_nav = self.portfolio.nav
            daily_ret = (current_nav / self.prev_nav) - 1.0

            self.ret_history.append(daily_ret)
            self.rolling_metrics = self._compute_rolling_metrics()

            reward_details = compute_reward_details(
                {
                    "ret": daily_ret,
                    "vol": self.rolling_metrics["vol"],
                    "cvar": self.rolling_metrics["cvar"],
                    "mdd": self.rolling_metrics["mdd"],
                },
                self.current_w,
                self.current_rho,
            )
            reward = reward_details["clipped_reward"]

            # ✅ STORE REAL ACTIONS (SOFT WEIGHTS)
            if allow_learning:
                self.replay.add(
                    (
                        self.prev_state.cpu(),
                        self.prev_action_weights,
                        reward,
                        z.cpu(),
                        False,
                        self.prev_actor_hidden_in.cpu() if self.prev_actor_hidden_in is not None else None,
                    )
                )

                if len(self.replay) >= self.cfg.get("batch_size", 32):
                    self._update_agent(step_count)

            self.logger.log_performance(
                {
                    "date": date_str,
                    "step": step_count,
                    "phase": phase,
                    "nav": current_nav,
                    "cash": self.portfolio.cash,
                    "cash_pct": self.portfolio.cash / self.portfolio.total_value,
                    "ret": daily_ret,
                    "reward": reward,
                    "vol": self.rolling_metrics["vol"],
                    "cvar": self.rolling_metrics["cvar"],
                    "mdd": self.rolling_metrics["mdd"],
                    "term_ret": reward_details["term_ret"],
                    "term_vol": reward_details["term_vol"],
                    "term_cvar": reward_details["term_cvar"],
                    "term_mdd": reward_details["term_mdd"],
                    "raw_reward": reward_details["raw_reward"],
                    "exposure": reward_details["exposure"],
                }
            )

        # ---- Top-K execution (non-differentiable, OK)
        masked_scores = np.array(scores_np, copy=True)
        masked_scores[~valid_mask] = -1e9
        ranked_idx = np.argsort(-masked_scores)
        ranked_list = [self.tickers[i] for i in ranked_idx]
        selected_tickers = self.selector.update(ranked_list)
        selected_tickers = [t for t in selected_tickers if valid_mask[self.tickers.index(t)]]

        if len(selected_tickers) < self.K:
            for t in ranked_list:
                t_idx = self.tickers.index(t)
                if not valid_mask[t_idx]:
                    continue
                if t in selected_tickers:
                    continue
                selected_tickers.append(t)
                if len(selected_tickers) >= self.K:
                    break

        selected_idx = [self.tickers.index(t) for t in selected_tickers]

        alloc = {t: 0.0 for t in self.tickers}
        if selected_idx:
            sel_scores = masked_scores[selected_idx]
            exps = np.exp(sel_scores - sel_scores.max())
            sel_w = exps / exps.sum()

            for i, t in enumerate(selected_tickers):
                alloc[t] = float(sel_w[i])

        idx_open = self.feature_cols.index("Open")
        prices = {
            self.tickers[i]: (float(X_day[i, -1, idx_open]) if valid_mask[i] else 0.0)
            for i in range(N)
        }

        self.prev_nav = self.portfolio.nav
        executed_allocs, trades_list = self.portfolio.execute_allocations(
            alloc,
            prices,
            target_cash=self.current_rho,
            min_trade_value=self.cfg.get("min_trade_value", 1000.0),
        )
        
        # ---- LOGGING TRADES & EXECUTION ----
        for trade in trades_list:
            trade["date"] = date_str
            trade["step"] = step_count
            self.logger.log_trade(trade)
            
        self.logger.log_execution({
            "date": date_str,
            "step": step_count,
            "phase": phase,
            "valid_assets": int(valid_mask.sum()),
            "meta_w": [float(x) for x in self.current_w],
            "meta_rho": float(self.current_rho),
            "ranked_top20": ranked_list[:20],
            "selected_tickers": selected_tickers,
            "target_allocations": alloc,
            "allocations": executed_allocs,
            "nav": self.portfolio.nav,
            "cash": self.portfolio.cash
        })

        executed_vec = np.array([executed_allocs.get(t, 0.0) for t in self.tickers], dtype=np.float32)

        # ---- Save transition pointers
        self.prev_state = z
        self.prev_action_weights = executed_vec
        self.prev_actor_hidden_in = actor_hidden_in_detached

    # ------------------------------------------------------------------
    # ACTOR–CRITIC UPDATE (FIXED)
    # ------------------------------------------------------------------
    def _update_agent(self, step):
        batch = self.replay.sample(self.cfg.get("batch_size", 32))

        states = torch.cat([x[0] for x in batch]).to(self.device)
        actions = torch.from_numpy(
            np.stack([x[1] for x in batch])
        ).float().to(self.device)
        rewards = torch.tensor([x[2] for x in batch], dtype=torch.float32).unsqueeze(1).to(self.device)
        next_states = torch.cat([x[3] for x in batch]).to(self.device)
        dones = torch.tensor([x[4] for x in batch], dtype=torch.float32).unsqueeze(1).to(self.device)

        mem_dim = self.agent.actor.mem_dim
        hidden_in = torch.zeros((len(batch), mem_dim), dtype=torch.float32, device=self.device)
        for i, tr in enumerate(batch):
            if len(tr) >= 6 and tr[5] is not None:
                hidden_in[i] = tr[5].to(self.device)

        # IMPORTANT: critic expects EXACT d_gnn from config, not encoder output dim
        d_gnn = self.encoder.d_gnn
        N = states.size(1)

        state_flat = states[:, :, :d_gnn].contiguous().view(states.size(0), N * d_gnn)
        next_state_flat = next_states[:, :, :d_gnn].contiguous().view(next_states.size(0), N * d_gnn)

        # SAFETY ASSERT (leave this in)
        assert state_flat.shape[1] == d_gnn * N, (
            f"Critic input mismatch: got {state_flat.shape[1]}, "
            f"expected {d_gnn * N}"
        )

        
        critic_loss = self.agent.update_critic(
            state_flat,
            actions,
            rewards,
            next_state_flat,
            dones
        )


        actor_loss = self.agent.update_actor(states, hidden_in=hidden_in)

        self.agent.update_targets()

        self.logger.log_internal(
            {
                "step": step,
                "phase": "train",
                "a_loss": actor_loss,
                "c_loss": critic_loss,
                "w_mean": actions.mean().item(),
                "w_std": actions.std().item(),
            }
        )

    # ------------------------------------------------------------------
    # META WEEKLY UPDATE
    # ------------------------------------------------------------------
    def _handle_weekly_update(self, date, date_str, step_count, allow_learning: bool = True, phase: str = "train"):
        weekly_return = 0.0
        meta_loss_val = 0.0
        if self.current_week_num != -1:
            weekly_return = (self.portfolio.nav / self.week_start_nav) - 1.0
            if allow_learning and step_count > self.meta_warmup_steps:
                # REINFORCE update using previous log_prob
                if self.prev_log_prob is not None:
                    meta_loss_val = self.meta_trainer.update_policy(
                        weekly_return,
                        self.prev_log_prob,
                        entropy=self.prev_entropy,
                    )

            self.prev_week_stats = np.array(
                [
                    weekly_return,
                    self.rolling_metrics["vol"],
                    self.rolling_metrics["cvar"],
                    self.rolling_metrics["mdd"],
                ]
            )

        self.current_week_num = date.isocalendar()[1]
        self.week_start_nav = self.portfolio.nav

        macro_vals = self._get_weekly_macros(date)
        meta_state = np.concatenate([macro_vals, self.prev_week_stats, self.prev_action])
        meta_tensor = torch.tensor(meta_state, dtype=torch.float32).unsqueeze(0).to(self.device)

        log_prob_val = 0.0
        if step_count < self.meta_warmup_steps:
            w = np.array([1.0, 0.0, 0.0, 0.0])
            rho = self.meta_rho_min
            self.prev_log_prob = None
            self.prev_entropy = None
        else:
            if allow_learning:
                # Sample action during train for exploration
                w_t, rho_t, log_prob, entropy = self.meta_trainer.agent.get_action(meta_tensor)
                w_sample = w_t.squeeze(0).detach().cpu().numpy()
                rho_sample = float(rho_t.item())

                # Smoothly transition from warmup policy to sampled policy.
                if self.meta_transition_steps > 0 and step_count < (self.meta_warmup_steps + self.meta_transition_steps):
                    alpha = (step_count - self.meta_warmup_steps + 1) / float(self.meta_transition_steps)
                    alpha = float(np.clip(alpha, 0.0, 1.0))
                else:
                    alpha = 1.0

                w_default = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
                w = (1.0 - alpha) * w_default + alpha * w_sample
                w = w / (w.sum() + 1e-12)
                rho = (1.0 - alpha) * self.meta_rho_min + alpha * rho_sample

                self.prev_log_prob = log_prob
                self.prev_entropy = entropy
                log_prob_val = log_prob.item()
                entropy_val = entropy.item()
            else:
                # Deterministic action during test/inference
                with torch.no_grad():
                    w_t, rho_t = self.meta_trainer.agent(meta_tensor)
                w = w_t.squeeze(0).cpu().numpy()
                rho = float(rho_t.item())
                self.prev_log_prob = None
                self.prev_entropy = None
                entropy_val = 0.0

        if step_count < self.meta_warmup_steps:
            entropy_val = 0.0

        rho_before_clip = float(rho)
        rho = float(np.clip(rho_before_clip, self.meta_rho_min, self.meta_rho_max))
        rho_clipped = abs(rho_before_clip - rho) > 1e-12

        self.current_w = w
        self.current_rho = rho
        self.prev_action = np.concatenate([w, [rho]])

        self.logger.log_meta(
            {
                "date": date_str,
                "step": step_count,
                "phase": phase,
                "rho": rho,
                "w_ret": w[0],
                "w_vol": w[1],
                "w_cvar": w[2],
                "w_mdd": w[3],
                "log_prob": log_prob_val,
                "entropy": entropy_val,
                "weekly_return": weekly_return,
                "meta_loss": meta_loss_val,
                "rho_clipped": rho_clipped,
            }
        )

    def _build_adjacency(self, X_day: np.ndarray, valid_mask: np.ndarray = None) -> torch.Tensor:
        """
        Build inter-stock adjacency from historical close returns within the window.
        Falls back to identity graph if required inputs are unavailable.
        """
        n = X_day.shape[0]
        if self.graph_mode == "identity" or self.close_idx is None:
            return torch.eye(n, dtype=torch.float32, device=self.device)

        if valid_mask is None:
            valid_mask = np.ones((n,), dtype=bool)
        else:
            valid_mask = np.asarray(valid_mask, dtype=bool).reshape(-1)
            if valid_mask.shape[0] != n:
                valid_mask = np.ones((n,), dtype=bool)

        close_window = X_day[:, :, self.close_idx].astype(np.float64)
        if close_window.shape[1] < 3:
            return torch.eye(n, dtype=torch.float32, device=self.device)

        a = np.eye(n, dtype=np.float32)
        valid_idx = np.flatnonzero(valid_mask)
        if valid_idx.size <= 1:
            return torch.tensor(a, dtype=torch.float32, device=self.device)

        valid_close = close_window[valid_idx]
        rets = np.diff(valid_close, axis=1) / (valid_close[:, :-1] + 1e-8)
        rets = np.nan_to_num(rets, nan=0.0, posinf=0.0, neginf=0.0)

        std = np.std(rets, axis=1)
        active_idx = valid_idx[std > 1e-12]
        if active_idx.size <= 1:
            return torch.tensor(a, dtype=torch.float32, device=self.device)

        active_pos = [int(np.where(valid_idx == idx)[0][0]) for idx in active_idx]
        active_rets = rets[active_pos]
        corr = np.corrcoef(active_rets)
        corr = np.nan_to_num(corr, nan=0.0, posinf=0.0, neginf=0.0)
        if self.graph_abs_corr:
            corr = np.abs(corr)

        sub_a = np.zeros_like(corr, dtype=np.float32)
        if self.graph_mode == "corr_threshold":
            sub_a[corr >= self.graph_corr_thr] = 1.0
        else:
            # default: knn
            k = max(1, min(self.graph_k, active_idx.size - 1))
            for i in range(active_idx.size):
                order = np.argsort(-corr[i])
                neighbors = [j for j in order if j != i][:k]
                sub_a[i, neighbors] = 1.0

        np.fill_diagonal(sub_a, 1.0)
        row_sum = sub_a.sum(axis=1, keepdims=True)
        row_sum[row_sum == 0.0] = 1.0
        sub_a = sub_a / row_sum

        for local_i, global_i in enumerate(active_idx):
            a[global_i, active_idx] = sub_a[local_i]

        return torch.tensor(a, dtype=torch.float32, device=self.device)

    # ------------------------------------------------------------------
    # UTILITIES
    # ------------------------------------------------------------------
    def _compute_rolling_metrics(self):
        if len(self.ret_history) < 5:
            return {"vol": 0.0, "cvar": 0.0, "mdd": 0.0}

        arr = np.array(self.ret_history)
        vol = np.std(arr)
        worst = np.sort(arr)[: max(1, int(0.05 * len(arr)))]
        cvar = -np.mean(worst)

        cum = np.cumprod(1 + arr)
        peak = np.maximum.accumulate(cum)
        mdd = np.max((peak - cum) / peak)

        return {"vol": vol, "cvar": cvar, "mdd": mdd}

    def _get_weekly_macros(self, date):
        idx = self.macro_df.index.asof(date - pd.Timedelta(days=1))
        if pd.isna(idx):
            return np.zeros(len(self.meta_cols), dtype=np.float32)
        row = self.macro_df.loc[idx]
        return np.array([row.get(c, 0.0) for c in self.meta_cols], dtype=np.float32)
