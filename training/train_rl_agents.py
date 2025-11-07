# training/train_rl_agents.py
"""
RL Trainer: Handles daily RL updates, replay buffer, and checkpoint saving.
Compatible with run_trainer.py full training loop.
"""

import os
import numpy as np
import torch
from tqdm import tqdm

from dataset.dataset_windows import build_windows_from_paths
from graph.graph_builder import build_adj_for_window_from_parquet_dfs
from timeseries_models.encoders import CombinedEncoder

from rl_layer.actor_critic import MADDPG
from rl_layer.replay_buffer import ReplayBuffer
from rl_layer.rebalance_manager import HysteresisSelector, allocations_to_shares
from rl_layer.execution import Portfolio
from rl_layer.reward_function import compute_reward

from proj_logging.logger import append_csv, log_info

class TrainerRL:
    def __init__(self, cfg, logger=None, ckpt_mgr=None, device="cpu"):
        self.cfg = cfg
        self.device = device
        self.logger = logger
        self.ckpt_mgr = ckpt_mgr

        self.tickers = cfg["tickers"]
        self.window = cfg.get("window_length", 126)
        self.K = cfg.get("top_k", 10)

        # Model components
        enc_cfg = cfg.get("encoder", {"input_dim": 26, "W": 126, "d_time": 64, "d_gnn": 64})
        self.encoder = CombinedEncoder(
            input_dim=enc_cfg["input_dim"],
            W=enc_cfg.get("W", 126),
            d_time=enc_cfg.get("d_time", 64),
            d_gnn=enc_cfg.get("d_gnn", 64),
            time_layers=enc_cfg.get("time_layers", 2),
            gnn_layers=enc_cfg.get("gnn_layers", 2)
        ).to(self.device)

        self.agent = MADDPG(d_gnn=enc_cfg.get("d_gnn", 64), K=self.K, device=self.device)
        self.replay = ReplayBuffer(capacity=cfg.get("replay_capacity", 200000))
        self.portfolio = Portfolio(self.tickers, initial_nav=cfg.get("initial_nav", 1.0))
        self.selector = HysteresisSelector(self.tickers, k=self.K, hysteresis_days=cfg.get("hysteresis_days", 3))

        self.logs_dir = cfg.get("logs_dir", "logging")
        os.makedirs(self.logs_dir, exist_ok=True)

        self.checkpoint_dir = cfg.get("checkpoint_dir", "checkpoints")
        os.makedirs(self.checkpoint_dir, exist_ok=True)

        self.global_step = 0

    # ---------------------------------------------
    # Core daily RL update
    # ---------------------------------------------
    def step_daily(self, X_day, date):
        """
        Performs one daily RL update step:
          - Encode time series & graph
          - Select portfolio
          - Execute allocations
          - Compute reward & store transition
        """
        X_t = torch.tensor(X_day, dtype=torch.float32).unsqueeze(0).to(self.device)
        N = X_t.shape[1]
        A = torch.eye(N).to(self.device)
        with torch.no_grad():
            z_t = self.encoder(X_t, A).squeeze(0).cpu().numpy()

        # Get actor scores
        z_tensor = torch.tensor(z_t, dtype=torch.float32).unsqueeze(0).to(self.device)
        with torch.no_grad():
            scores = self.agent.actor(z_tensor).squeeze(0).cpu().numpy()  # (N,)

        ranked_idx = np.argsort(-scores)
        ranked_list = [self.tickers[i] for i in ranked_idx.tolist()]
        selected = self.selector.update(ranked_list)

        # Build allocations
        proposed = {tkr: 0.0 for tkr in self.tickers}
        if len(selected) > 0:
            s_sel = np.array([scores[self.tickers.index(tk)] for tk in selected])
            w = s_sel / (s_sel.sum() + 1e-12)
            investable = 1.0 - self.cfg.get("min_cash", 0.05)
            for i, tk in enumerate(selected):
                proposed[tk] = float(w[i] * investable)

        prices_t = {tkr: 1.0 for tkr in self.tickers}  # placeholder
        shares, executed_alloc = allocations_to_shares(
            proposed, self.portfolio.allocations, prices_t, NAV=self.portfolio.nav,
            per_asset_delta_thresh=self.cfg.get("per_asset_delta_thresh", 0.01),
            turnover_thresh=self.cfg.get("turnover_thresh", 0.02),
            min_trade_value=self.cfg.get("min_trade_value", 1000.0),
            cap=self.cfg.get("max_asset_weight", 0.20),
            min_cash=self.cfg.get("min_cash", 0.05),
            lot_size=self.cfg.get("lot_size", 1)
        )

        self.portfolio.execute_allocations(executed_alloc, prices_t)
        port_ret = self.portfolio.apply_open_to_open(prices_t, prices_t)
        components = {"ret": port_ret, "vol": 0.0, "cvar": 0.0, "mdd": 0.0}
        w_meta = np.array([0.5, 0.1666, 0.1666, 0.1666])
        rew = compute_reward(components, w_meta)

        # Store in replay buffer
        d = z_t.shape[1]
        state_flat = z_t.flatten()[:self.K * d]
        action_vec = np.array([executed_alloc.get(tk, 0.0) for tk in selected[:self.K]])
        next_state_flat = np.copy(state_flat)
        self.replay.add((state_flat, action_vec, rew, next_state_flat, False))

        append_csv(os.path.join(self.logs_dir, "training_logs.csv"), {
            "date": str(date), "nav": self.portfolio.nav, "reward": rew, "selected": ",".join(selected)
        })
        log_info(f"[RLTrainer] {date} | NAV={self.portfolio.nav:.4f} | Reward={rew:.6f}")

        self.global_step += 1
        if self.global_step % 50 == 0 and self.ckpt_mgr:
            self.ckpt_mgr.save(
                step=self.global_step,
                actor=self.agent.actor,
                critic=self.agent.critic
            )

    def is_weekly_update(self, date):
        # update every 5 trading days
        return (self.global_step % self.cfg.get("weekly_update_every_days", 5)) == 0

    def save_all(self):
        if self.ckpt_mgr:
            self.ckpt_mgr.save(
                step=self.global_step,
                actor=self.agent.actor,
                critic=self.agent.critic
            )

    # ---------------------------------------------
    # Smoke Test (for quick dry run)
    # ---------------------------------------------
    def run_smoke(self, parquet_paths, min_date=None, max_days=20):
        X, dates = build_windows_from_paths(parquet_paths, self.cfg["feature_cols"], W=self.window)
        T = len(dates)
        for t in range(min(max_days, T-1)):
            self.step_daily(X[t], dates[t])
        print("SMOKE RUN COMPLETE")
