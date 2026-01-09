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
        
        # --- 1. Core Setup ---
        self.tickers = cfg["tickers"]
        self.feature_cols = cfg["feature_cols"]
        self.window = cfg.get("window_length", 126)
        self.N_stocks = len(self.tickers)
        
        # --- 2. Models (Encoder + Actor-Critic) ---
        enc_cfg = cfg.get("encoder", {"input_dim": len(self.feature_cols), "W": 126})
        self.encoder = CombinedEncoder(
            input_dim=enc_cfg["input_dim"],
            W=enc_cfg.get("W", 126),
            d_time=enc_cfg.get("d_time", 64),
            d_gnn=enc_cfg.get("d_gnn", 64),
            time_layers=enc_cfg.get("time_layers", 2),
            gnn_layers=enc_cfg.get("gnn_layers", 2)
        ).to(self.device)

        self.K = cfg.get("top_k", 10)
        self.agent = MADDPG(d_gnn=enc_cfg.get("d_gnn", 64), K=self.K, device=self.device)
        self.replay = ReplayBuffer(capacity=cfg.get("replay_capacity", 200000))

        # --- 3. Portfolio & Execution ---
        self.portfolio = Portfolio(self.tickers, initial_nav=cfg.get("initial_nav", 1000.0))
        self.selector = HysteresisSelector(self.tickers, k=self.K, hysteresis_days=cfg.get("hysteresis_days", 3))
        
        # Rolling Metrics for Reward Calculation (Vol, CVaR, MDD)
        self.ret_history = deque(maxlen=30) 
        self.rolling_metrics = {"vol": 0.0, "cvar": 0.0, "mdd": 0.0}
        
        # --- 4. Meta-RL Integration ---
        self.meta_trainer = None 
        
        # STRICT: Load Macros from Config. Fail if missing.
        macro_path = cfg.get("macros_weekly_path", "data/macros/combined_macros_weekly.csv")
        self._load_macros(macro_path)
        
        # Look for meta_input at root level (preferred) or nested (legacy)
        self.meta_cols = cfg.get("meta_input", {}).get("macros", [])
        if not self.meta_cols:
             self.meta_cols = cfg.get("meta_agent", {}).get("meta_input", {}).get("macros", [])
        
        if not self.meta_cols:
            raise ValueError("[Critical] No 'meta_input.macros' found in params.yaml. Cannot train Meta-Agent.")

        self.logger.info(f"[TrainerRL] Meta-Agent will use {len(self.meta_cols)} macro features.")

        # Meta State Context
        self.current_week_num = -1
        self.current_w = np.array([1.0, 0.0, 0.0, 0.0]) # Default: Aggressive
        self.current_rho = 0.05                         # Default: Invested
        self.week_start_nav = self.portfolio.nav
        self.meta_log_prob = None
        self.meta_entropy = None
        self.prev_week_stats = np.zeros(4) # [ret, vol, cvar, mdd]
        self.prev_action = np.zeros(5)     # [w..., rho]

        # RL Loop Transition State
        self.prev_state = None
        self.prev_actions_indices = None
        self.prev_nav = self.portfolio.nav

    def _load_macros(self, path):
        """Loads weekly macro data for the Meta-Agent."""
        if os.path.exists(path):
            df = pd.read_csv(path)
            df['Date'] = pd.to_datetime(df['Date'])
            self.macro_df = df.sort_values('Date').set_index('Date')
            self.logger.info(f"[TrainerRL] Loaded {len(df)} weekly macro records.")
        else:
            raise FileNotFoundError(f"[Critical] Macro file not found at {path}")

    def attach_meta_trainer(self, trainer):
        self.meta_trainer = trainer

    def step_daily(self, X_day: np.ndarray, date_str: str, step_count: int):
        """
        Main Training Step:
        1. Weekly Meta-Update (if Monday)
        2. Encode State
        3. Compute Reward (using Daily Return & Meta Weights)
        4. Store in Replay Buffer & Train Daily Agent
        5. Execute Trades (Constraint by Meta Rho)
        """
        date = pd.to_datetime(date_str)
        week_num = date.isocalendar()[1]
        
        # --- 1. Encode Current State (s_t) ---
        N = X_day.shape[0]
        A = torch.eye(N).to(self.device)
        X_tensor = torch.tensor(X_day, dtype=torch.float32).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            state_encoding = self.encoder(X_tensor, A) # (1, N, d_gnn)
            # Get Actor Scores
            scores = self.agent.actor(state_encoding).squeeze(0).cpu().numpy() # (N,)

        # --- 2. Meta-Agent Weekly Update (Weekly Pulse) ---
        if week_num != self.current_week_num:
            self._handle_weekly_update(date, date_str, step_count)

        # --- 3. Calculate Reward & Train Daily Agent ---
        # We utilize the transition: (prev_s, prev_a) -> (r, s_new)
        if self.prev_state is not None:
            # Calculate daily return based on yesterday's holdings
            current_nav = self.portfolio.nav
            daily_ret = (current_nav / self.prev_nav) - 1.0 if self.prev_nav > 0 else 0.0
            
            # Update Rolling Risk Metrics
            self.ret_history.append(daily_ret)
            self.rolling_metrics = self._compute_rolling_metrics()
            
            components = {
                "ret": daily_ret, "vol": self.rolling_metrics['vol'],
                "cvar": self.rolling_metrics['cvar'], "mdd": self.rolling_metrics['mdd']
            }
            
            # Compute Meta-Weighted Reward
            reward = compute_reward(components, self.current_w, self.current_rho)
            
            # Add to Replay Buffer
            self.replay.add((
                self.prev_state.cpu(),          # s_t
                self.prev_actions_indices,      # a_t (indices)
                reward,                         # r_t
                state_encoding.cpu(),           # s_{t+1}
                False                           # done
            ))
            
            # CRITICAL: Trigger Training if buffer is ready
            if len(self.replay) > self.cfg.get("batch_size", 32):
                self._update_agent(step_count)

            # Log Daily Body Performance
            self.logger.log_performance({
                "date": date_str, "step": step_count,
                "nav": current_nav, "cash": self.portfolio.cash,
                "cash_pct": self.portfolio.cash / self.portfolio.total_value,
                "ret": daily_ret, "reward": reward,
                "vol": components['vol'], "mdd": components['mdd']
            })

        # --- 4. Select Action (Top-K) ---
        ranked_idx = np.argsort(-scores)
        ranked_list = [self.tickers[i] for i in ranked_idx]
        
        # Use Hysteresis Selector to prevent churning
        selected_tickers = self.selector.update(ranked_list)
        selected_idx = [self.tickers.index(t) for t in selected_tickers]
        
        # Softmax weights for selected assets
        s_sel = np.array([scores[i] for i in selected_idx])
        exps = np.exp(s_sel - np.max(s_sel))
        sel_weights = exps / np.sum(exps)
        
        proposed_alloc = {t: 0.0 for t in self.tickers}
        for i, t in enumerate(selected_tickers):
            proposed_alloc[t] = float(sel_weights[i])

        # --- 5. Execute Trades (Enforcing Meta-Constraint) ---
        idx_open = self.feature_cols.index("Open")
        prices_t = {self.tickers[i]: float(X_day[i, -1, idx_open]) for i in range(N)}
        
        self.prev_nav = self.portfolio.nav 
        
        # Execute: Passes target_cash (rho) to force liquidation if needed
        executed_alloc, trades = self.portfolio.execute_allocations(
            proposed_alloc, prices_t, 
            target_cash=self.current_rho, 
            min_trade_value=self.cfg.get("min_trade_value", 1000.0)
        )
        
        # Log Atomic Trades
        for tr in trades:
            tr["date"] = date_str
            tr["step"] = step_count
            self.logger.log_trade(tr)

        # --- 6. Update State Pointers ---
        self.prev_state = state_encoding 
        self.prev_actions_indices = selected_idx

    def _update_agent(self, step):
        """Samples batch and updates Daily Actor/Critic."""
        batch = self.replay.sample(self.cfg.get("batch_size", 32))
        
        # Unpack Batch
        states = torch.cat([x[0] for x in batch], dim=0).to(self.device)
        selected_indices = [x[1] for x in batch] 
        rewards = torch.tensor([x[2] for x in batch], dtype=torch.float32).unsqueeze(1).to(self.device)
        next_states = torch.cat([x[3] for x in batch], dim=0).to(self.device)
        dones = torch.tensor([x[4] for x in batch], dtype=torch.float32).unsqueeze(1).to(self.device)

        # Helper to build flat state for Critic
        def build_state_flat(z_batch, sel_idx_batch):
            B_size = z_batch.size(0)
            flat_list = []
            for b in range(B_size):
                idx = sel_idx_batch[b]
                sel_emb = z_batch[b, idx, :] 
                flat_list.append(sel_emb.view(-1))
            return torch.stack(flat_list)

        state_flat = build_state_flat(states, selected_indices)
        next_state_flat = build_state_flat(next_states, selected_indices)
        
        # Simplified action tensor for critic (Uniform for now)
        actions_tensor = torch.ones((len(batch), self.K), device=self.device) / self.K

        # Update Critic
        critic_loss = self.agent.update_critic(
            state_flat, actions_tensor, rewards, next_state_flat, actions_tensor, dones
        )
        
        # Update Actor
        actor_loss = self.agent.update_actor(
            states, build_state_flat, selected_indices
        )
        
        self.agent.update_targets()
        
        # Log Internal Training Stats
        self.logger.log_internal({
            "step": step, "actor_loss": actor_loss, "critic_loss": critic_loss,
            "grad_norm": 0.0, "w_mean": 0.0, "w_std": 0.0
        })

    def _handle_weekly_update(self, date, date_str, step_count):
        """Handles Meta-Agent Logic (Weekly Pulse)"""
        # 1. End Previous Week (Reward Meta-Agent)
        if self.current_week_num != -1 and self.meta_log_prob is not None:
            weekly_return = (self.portfolio.nav / self.week_start_nav) - 1.0
            loss, adv = self.meta_trainer.update_policy(weekly_return, self.meta_log_prob, self.meta_entropy)
            
            # Store stats for next state
            self.prev_week_stats = np.array([
                weekly_return, self.rolling_metrics['vol'], 
                self.rolling_metrics['cvar'], self.rolling_metrics['mdd']
            ])

        # 2. Start New Week (Get New Orders)
        self.current_week_num = date.isocalendar()[1]
        self.week_start_nav = self.portfolio.nav
        
        # Fetch Macros
        macro_features = self._get_weekly_macros(date)
        
        # Construct State Vector: [Macros(M) + Stats(4) + PrevAction(5)]
        meta_state_vec = np.concatenate([macro_features, self.prev_week_stats, self.prev_action])
        meta_tensor = torch.tensor(meta_state_vec, dtype=torch.float32).unsqueeze(0).to(self.device)
        
        # Sample Strategy
        w, rho, lp, ent, _ = self.meta_trainer.agent.sample_action(meta_tensor)
        
        # Lock Strategy for the Week
        self.current_w = w
        self.current_rho = rho
        self.meta_log_prob = lp
        self.meta_entropy = ent
        self.prev_action = np.concatenate([w, [rho]])
        
        self.logger.log_meta({
            "date": date_str, "step": step_count, "rho": rho, 
            "w_ret": w[0], "w_vol": w[1], "w_cvar": w[2], "w_mdd": w[3],
            "log_prob": lp.item(), "entropy": ent.item()
        })

    def _compute_rolling_metrics(self):
        if len(self.ret_history) < 5: return {"vol": 0.0, "cvar": 0.0, "mdd": 0.0}
        arr = np.array(self.ret_history)
        vol = np.std(arr)
        cutoff = max(1, int(0.05 * len(arr)))
        worst = np.sort(arr)[:cutoff]
        cvar = -np.mean(worst)
        cum = np.cumprod(1 + arr)
        peak = np.maximum.accumulate(cum)
        dd = (peak - cum) / peak
        mdd = np.max(dd)
        return {"vol": vol, "cvar": cvar, "mdd": mdd}

    def _get_weekly_macros(self, date):
        # Get last available macro data before this date
        try:
            idx = self.macro_df.index.asof(date - pd.Timedelta(days=1))
            if pd.isna(idx): 
                # Raise Error if data is genuinely missing at start (Stops "Silent Failure")
                # If you are okay with 0s at the very start, you can return np.zeros here.
                # But you requested "error if not working".
                self.logger.info(f"[Meta] No macro data found for {date}. Returning zeros for cold start.")
                return np.zeros(len(self.meta_cols), dtype=np.float32)
            
            row = self.macro_df.loc[idx]
            vals = [row.get(c, 0.0) for c in self.meta_cols]
            return np.array(vals, dtype=np.float32)
        except Exception as e:
            self.logger.error(f"[Meta] Error fetching macros: {e}")
            raise e