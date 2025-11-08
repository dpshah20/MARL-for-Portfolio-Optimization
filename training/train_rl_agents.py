# training/train_rl_agents.py
import os, time, json
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
from proj_logging.logger import append_csv, append_jsonl, log_info
from training.checkpoints import CheckpointManager


class TrainerRL:
    def __init__(self, cfg: dict,logger=None,ckpt_mgr=None, device: str = "cpu"):
        self.cfg = cfg
        self.device = device
        self.tickers = cfg["tickers"]
        self.feature_cols = cfg["feature_cols"]
        self.window = cfg.get("window_length", 126)
        self.logger = logger
        self.ckpt_mgr = ckpt_mgr
        # Encoder
        enc_cfg = cfg.get("encoder", {"input_dim": len(self.feature_cols), "W": 126})
        self.encoder = CombinedEncoder(
            input_dim=enc_cfg["input_dim"],
            W=enc_cfg.get("W", 126),
            d_time=enc_cfg.get("d_time", 64),
            d_gnn=enc_cfg.get("d_gnn", 64),
            time_layers=enc_cfg.get("time_layers", 2),
            gnn_layers=enc_cfg.get("gnn_layers", 2)
        ).to(self.device)

        # Agent + buffers
        self.K = cfg.get("top_k", 10)
        self.agent = MADDPG(d_gnn=enc_cfg.get("d_gnn", 64), K=self.K, device=self.device)
        self.replay = ReplayBuffer(capacity=cfg.get("replay_capacity", 200000))

        # Portfolio and selector
        self.portfolio = Portfolio(self.tickers, initial_nav=cfg.get("initial_nav", 1.0))
        self.selector = HysteresisSelector(self.tickers, k=self.K,
                                           hysteresis_days=cfg.get("hysteresis_days", 3))

        # Paths
        self.logs_dir = cfg.get("logs_dir", "logs")
        os.makedirs(self.logs_dir, exist_ok=True)
        self.exec_log_path = os.path.join(self.logs_dir, "execution_logs.jsonl")
        self.train_log_path = os.path.join(self.logs_dir, "portfolio_logs.csv")
        self.checkpoint_dir = cfg.get("checkpoint_dir", "checkpoints")
        os.makedirs(self.checkpoint_dir, exist_ok=True)

        self.prev_nav = self.portfolio.nav

    # ---------------------------------------------------------------------- #
    def step_daily(self, X_day: np.ndarray, date):
        """Run one training step (day)."""
        N = X_day.shape[0]

        # build adjacency (simplified identity)
        A = torch.eye(N).to(self.device)
        X_t = torch.tensor(X_day, dtype=torch.float32).unsqueeze(0).to(self.device)

        # encode
        with torch.no_grad():
            z_t = self.encoder(X_t, A).squeeze(0).cpu().numpy()

        # actor scores
        z_tensor = torch.tensor(z_t, dtype=torch.float32).unsqueeze(0).to(self.device)
        with torch.no_grad():
            scores = self.agent.actor(z_tensor).squeeze(0).cpu().numpy()

        ranked_idx = np.argsort(-scores)
        ranked_list = [self.tickers[i] for i in ranked_idx.tolist()]
        selected = self.selector.update(ranked_list)

        # propose allocations
        proposed = {tkr: 0.0 for tkr in self.tickers}
        s_sel = np.array([scores[self.tickers.index(t)] for t in selected])
        if len(selected) > 0:
            w = s_sel / (s_sel.sum() + 1e-12)
            investable = 1.0 - self.cfg.get("min_cash", 0.05)
            for i, tk in enumerate(selected):
                proposed[tk] = float(w[i] * investable)

        # --- Execution prices from features (last day Open)
        idx_open = self.feature_cols.index("Open")
        prices_t = {self.tickers[i]: float(X_day[i, -1, idx_open]) for i in range(N)}

        # --- Execute trades
        executed_alloc = self.portfolio.execute_allocations(
            proposed, prices_t,
            lot_size=self.cfg.get("lot_size", 1),
            min_trade_value=self.cfg.get("min_trade_value", 1000.0),
            cap_per_asset=self.cfg.get("max_asset_weight", 0.20),
            min_cash=self.cfg.get("min_cash", 0.05)
        )

        nav_before = self.prev_nav
        # simulate next-day open (for now assume +random noise)
        next_prices = {t: prices_t[t] * (1.0 + np.random.normal(0, 0.002)) for t in prices_t}
        port_ret = self.portfolio.apply_open_to_open(prices_t, next_prices)
        nav_after = self.portfolio.nav
        self.prev_nav = nav_after

        # compute reward
        components = {"ret": port_ret, "vol": 0.0, "cvar": 0.0, "mdd": 0.0}
        w_meta = np.array([0.5, 0.1666, 0.1666, 0.1666])
        rew = compute_reward(components, w_meta)

        # log summary to CSV
        append_csv(self.train_log_path, {
            "date": str(date),
            "nav_before": nav_before,
            "nav_after": nav_after,
            "return": port_ret,
            "reward": rew,
            "selected": ",".join(selected)
        })

        # log detailed execution per stock
        for i, tk in enumerate(self.tickers):
            rec = {
                "date": str(date),
                "ticker": tk,
                "open_price": prices_t[tk],
                "features": X_day[i, -1, :].tolist(),
                "proposed_alloc": proposed[tk],
                "executed_alloc": executed_alloc[tk],
                "nav_before": nav_before,
                "nav_after": nav_after,
                "reward": rew
            }
            append_jsonl(self.exec_log_path, rec)

        log_info(f"[RLTrainer] {date} | NAV={nav_after:.4f} | Reward={rew:.6f}")
