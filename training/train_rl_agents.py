# training/train_rl_agents.py
import os, time
import numpy as np
import torch
from tqdm import tqdm
from dataset.dataset_windows import build_windows_from_paths
from feature_pipeline.graph_builder import build_adj_for_window_from_parquet_dfs
from feature_pipeline.encoders import CombinedEncoder
from rl_layer.actor_critic import MADDPG
from rl_layer.replay_buffer import ReplayBuffer
from rl_layer.rebalance_manager import HysteresisSelector, allocations_to_shares
from rl_layer.execution import Portfolio
from rl_layer.reward_function import compute_reward
from logging.logger import append_csv, log_info
from training.checkpoints import save_all, load_all

class TrainerRL:
    def __init__(self, cfg: dict, device: str = "cpu"):
        self.cfg = cfg
        self.device = device
        self.tickers = cfg["tickers"]
        self.window = cfg.get("window_length", 126)
        # encoder
        enc_cfg = cfg.get("encoder", {"input_dim":26, "W":126, "d_time":64, "d_gnn":64})
        self.encoder = CombinedEncoder(input_dim=enc_cfg["input_dim"], W=enc_cfg.get("W",126),
                                       d_time=enc_cfg.get("d_time",64), d_gnn=enc_cfg.get("d_gnn",64),
                                       time_layers=enc_cfg.get("time_layers",2), gnn_layers=enc_cfg.get("gnn_layers",2)).to(self.device)
        # MADDPG
        self.K = cfg.get("top_k", 10)
        self.agent = MADDPG(d_gnn=enc_cfg.get("d_gnn",64), K=self.K, device=self.device)
        # replay
        self.replay = ReplayBuffer(capacity=cfg.get("replay_capacity",200000))
        # portfolio
        self.portfolio = Portfolio(self.tickers, initial_nav=cfg.get("initial_nav", 1.0))
        # selector
        self.selector = HysteresisSelector(self.tickers, k=self.K, hysteresis_days=cfg.get("hysteresis_days",3))
        # bookkeeping
        self.logs_dir = cfg.get("logs_dir","logs")
        os.makedirs(self.logs_dir, exist_ok=True)
        self.checkpoint_dir = cfg.get("checkpoint_dir","checkpoints")
        os.makedirs(self.checkpoint_dir, exist_ok=True)

    def run_smoke(self, parquet_paths, min_date=None, max_days=20):
        # build windows generator -> we will use build_windows_from_paths to load all windows for simplicity
        X, dates = build_windows_from_paths(parquet_paths, self.cfg["feature_cols"], W=self.window)
        T = len(dates)
        # precompute embeddings for all windows to make z_next simple
        embeddings = []
        adjacencies = []
        import torch
        for t in range(T):
            X_t = torch.tensor(X[t], dtype=torch.float32).unsqueeze(0).to(self.device)
            # adjacency build
            # build slice_dates to pass to graph builder: we will derive from dfs; but here we skip and use knn on price returns computed in graph_builder
            # simplified: use build_adj_for_window_from_parquet_dfs requires dfs; to keep smoke simple, create identity adjacency
            N = X_t.shape[1]
            A = torch.eye(N).to(self.device)
            with torch.no_grad():
                z = self.encoder(X_t, A)  # (1,N,d_gnn)
            embeddings.append(z.squeeze(0).cpu().numpy())
            adjacencies.append(A.cpu().numpy())
        # iterate a few days
        for t in range(min(max_days, T-1)):
            z_t = embeddings[t]  # (N,d)
            z_next = embeddings[t+1]
            # actor scores (use actor on torch)
            z_tensor = torch.tensor(z_t, dtype=torch.float32).unsqueeze(0).to(self.device)
            with torch.no_grad():
                scores = self.agent.actor(z_tensor).squeeze(0).cpu().numpy()  # (N,)
            ranked_idx = np.argsort(-scores)
            ranked_list = [self.tickers[i] for i in ranked_idx.tolist()]
            selected = self.selector.update(ranked_list)
            # propose allocations proportional to scores among selected
            proposed = {tkr:0.0 for tkr in self.tickers}
            if len(selected)>0:
                s_sel = np.array([scores[self.tickers.index(tk)] for tk in selected])
                w = s_sel / (s_sel.sum()+1e-12)
                investable = 1.0 - self.cfg.get("min_cash", 0.05)
                for i,tk in enumerate(selected):
                    proposed[tk] = float(w[i] * investable)
            # get open prices approximated by Close values in window's last day feature Close column index
            # To keep smoke simple assume open prices are all 1.0
            prices_t = {tkr:1.0 for tkr in self.tickers}
            # compute shares via allocations_to_shares
            shares, executed_alloc = allocations_to_shares(proposed, self.portfolio.allocations, prices_t, NAV=self.portfolio.nav,
                                                          per_asset_delta_thresh=self.cfg.get("per_asset_delta_thresh",0.01),
                                                          turnover_thresh=self.cfg.get("turnover_thresh",0.02),
                                                          min_trade_value=self.cfg.get("min_trade_value",1000.0),
                                                          cap=self.cfg.get("max_asset_weight",0.20),
                                                          min_cash=self.cfg.get("min_cash",0.05),
                                                          lot_size=self.cfg.get("lot_size",1))
            # execute
            self.portfolio.execute_allocations(executed_alloc, prices_t)
            # next prices approximated by same prices -> zero return
            port_ret = self.portfolio.apply_open_to_open(prices_t, prices_t)
            # reward
            components = {"ret": port_ret, "vol":0.0, "cvar":0.0, "mdd":0.0}
            w_meta = np.array([0.5,0.1666,0.1666,0.1666])
            rew = compute_reward(components, w_meta)
            # store transition: state_flat, action_vec, reward, next_state_flat, done
            # build state_flat = flatten embeddings of selected K in fixed ordering (pad zeros)
            K = self.K
            d = z_t.shape[1]
            state_flat = np.zeros(K*d, dtype=np.float32)
            action_vec = np.zeros(K, dtype=np.float32)
            sel_idx = [self.tickers.index(x) for x in selected]
            for i,idx in enumerate(sel_idx[:K]):
                state_flat[i*d:(i+1)*d] = z_t[idx]
                action_vec[i] = executed_alloc[selected[i]]
            next_state_flat = np.zeros_like(state_flat)
            for i,idx in enumerate(sel_idx[:K]):
                next_state_flat[i*d:(i+1)*d] = z_next[idx]
            self.replay.add( (state_flat, action_vec, rew, next_state_flat, False) )
            append_csv(os.path.join(self.logs_dir, "portfolio_logs.csv"),
                       {"date": str(t), "nav": self.portfolio.nav, "return": port_ret, "selected": ",".join(selected)})
            log_info(f"Smoke day {t} nav:{self.portfolio.nav:.4f} selected:{len(selected)}")
        print("SMOKE RUN COMPLETE")
