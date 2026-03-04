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
from rl_layer.reward_function import compute_reward


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

        # RL transition state
        self.prev_state = None
        self.prev_action_weights = None
        self.prev_nav = self.portfolio.nav

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
    def step_daily(self, X_day: np.ndarray, date_str: str, step_count: int):
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
            else:
                self.logger.error(f"Input assets {N_input} < Model assets {self.N_stocks}. Crash likely.")

        # ---- Encode state
        N = X_day.shape[0]
        A = torch.eye(N).to(self.device)
        X_tensor = torch.tensor(X_day, dtype=torch.float32).unsqueeze(0).to(self.device)

        with torch.no_grad():
            z = self.encoder(X_tensor, A)            # (1, N, d)
            scores = self.agent.actor(z).squeeze(0) # (N,)

        scores_np = scores.cpu().numpy()
        soft_weights = torch.softmax(scores, dim=0).cpu().numpy()

        # ---- Weekly meta update
        if week_num != self.current_week_num:
            self._handle_weekly_update(date, date_str, step_count)

        # ---- Reward from previous action
        if self.prev_state is not None:
            current_nav = self.portfolio.nav
            daily_ret = (current_nav / self.prev_nav) - 1.0

            self.ret_history.append(daily_ret)
            self.rolling_metrics = self._compute_rolling_metrics()

            reward = compute_reward(
                {
                    "ret": daily_ret,
                    "vol": self.rolling_metrics["vol"],
                    "cvar": self.rolling_metrics["cvar"],
                    "mdd": self.rolling_metrics["mdd"],
                },
                self.current_w,
                self.current_rho,
            )

            # ✅ STORE REAL ACTIONS (SOFT WEIGHTS)
            self.replay.add(
                (
                    self.prev_state.cpu(),
                    self.prev_action_weights,
                    reward,
                    z.cpu(),
                    False,
                )
            )

            if len(self.replay) >= self.cfg.get("batch_size", 32):
                self._update_agent(step_count)

            self.logger.log_performance(
                {
                    "date": date_str,
                    "step": step_count,
                    "nav": current_nav,
                    "cash": self.portfolio.cash,
                    "cash_pct": self.portfolio.cash / self.portfolio.total_value,
                    "ret": daily_ret,
                    "reward": reward,
                    "vol": self.rolling_metrics["vol"],
                    "mdd": self.rolling_metrics["mdd"],
                }
            )

        # ---- Top-K execution (non-differentiable, OK)
        ranked_idx = np.argsort(-scores_np)
        ranked_list = [self.tickers[i] for i in ranked_idx]
        selected_tickers = self.selector.update(ranked_list)
        selected_idx = [self.tickers.index(t) for t in selected_tickers]

        alloc = {t: 0.0 for t in self.tickers}
        sel_scores = scores_np[selected_idx]
        exps = np.exp(sel_scores - sel_scores.max())
        sel_w = exps / exps.sum()

        for i, t in enumerate(selected_tickers):
            alloc[t] = float(sel_w[i])

        idx_open = self.feature_cols.index("Open")
        prices = {self.tickers[i]: float(X_day[i, -1, idx_open]) for i in range(N)}

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
            "allocations": executed_allocs,
            "nav": self.portfolio.nav,
            "cash": self.portfolio.cash
        })

        # ---- Save transition pointers
        self.prev_state = z
        self.prev_action_weights = soft_weights

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


        actor_loss = self.agent.update_actor(states)

        self.agent.update_targets()

        self.logger.log_internal(
            {
                "step": step,
                "a_loss": actor_loss,
                "c_loss": critic_loss,
                "w_mean": actions.mean().item(),
                "w_std": actions.std().item(),
            }
        )

    # ------------------------------------------------------------------
    # META WEEKLY UPDATE
    # ------------------------------------------------------------------
    def _handle_weekly_update(self, date, date_str, step_count):
        if self.current_week_num != -1:
            weekly_return = (self.portfolio.nav / self.week_start_nav) - 1.0
            if step_count > self.meta_warmup_steps:
                # REINFORCE update using previous log_prob
                if self.prev_log_prob is not None:
                    self.meta_trainer.update_policy(weekly_return, self.prev_log_prob)

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
            rho = 0.05
            self.prev_log_prob = None
        else:
            # Sample action
            w_t, rho_t, log_prob = self.meta_trainer.agent.get_action(meta_tensor)
            w = w_t.squeeze(0).detach().cpu().numpy()
            rho = float(rho_t.item())
            self.prev_log_prob = log_prob # Keep graph attached? Yes, for next update? 
            # Wait, REINFORCE needs log_prob attached to the graph.
            # But we update it AFTER the week ends.
            # So we must NOT detach it.
            log_prob_val = log_prob.item()

        rho = float(np.clip(rho, 0.05, 0.30))

        self.current_w = w
        self.current_rho = rho
        self.prev_action = np.concatenate([w, [rho]])

        self.logger.log_meta(
            {
                "date": date_str,
                "step": step_count,
                "rho": rho,
                "w_ret": w[0],
                "w_vol": w[1],
                "w_cvar": w[2],
                "w_mdd": w[3],
                "log_prob": log_prob_val,
                "entropy": 0.0,
            }
        )

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
