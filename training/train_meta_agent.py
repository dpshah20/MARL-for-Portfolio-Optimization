# training/train_meta_agent.py
import os
import torch
import numpy as np

from rl_layer.meta_agent import MetaAgent

class MetaTrainer:
    def __init__(self, cfg, logger=None, ckpt_mgr=None, device="cpu"):
        self.cfg = cfg
        self.logger = logger
        self.ckpt_mgr = ckpt_mgr
        self.device = device

        in_dim = cfg.get("meta_input_dim", cfg.get("encoder", {}).get("d_gnn", 64))
        hidden = cfg.get("meta_hidden_dim", 128)
        out_dim = cfg.get("meta_output_dim", 4)

        self.meta_agent = MetaAgent(input_dim=in_dim, hidden_dim=hidden, output_dim=out_dim).to(self.device)
        self.optimizer = torch.optim.Adam(self.meta_agent.parameters(), lr=cfg.get("meta_lr", 3e-4))
        self.epochs = cfg.get("meta_epochs", 5)
        self.save_dir = cfg.get("checkpoint_dir", "checkpoints/meta_agent")
        os.makedirs(self.save_dir, exist_ok=True)

    def train(self, X: np.ndarray, returns: np.ndarray, epochs: int = None, batch_size: int = 32):
        if epochs is None:
            epochs = self.epochs
        self.meta_agent.train()
        X_t = torch.tensor(X, dtype=torch.float32).to(self.device)
        Y_t = torch.tensor(returns, dtype=torch.float32).to(self.device)

        N = len(X_t)
        if N == 0:
            if self.logger:
                self.logger.warning("[MetaTrainer] empty meta dataset - skipping")
            return

        for epoch in range(epochs):
            idx = np.random.permutation(N)
            for start in range(0, N, batch_size):
                batch_idx = idx[start:start+batch_size]
                xb = X_t[batch_idx]
                yb = Y_t[batch_idx]

                rho, w = self.meta_agent(xb)
                # simple supervised objective: predict reward_stats if provided (we do mse on yb)
                # if yb shape differs, adapt loss accordingly
                pred = torch.cat([rho, w], dim=1) if w is not None else rho
                # if dimensions mismatch, reduce to mse on first columns
                tgt = yb
                if pred.shape != tgt.shape:
                    # align shapes by padding/trunc
                    L = min(pred.shape[1], tgt.shape[1])
                    loss = torch.nn.functional.mse_loss(pred[:, :L], tgt[:, :L])
                else:
                    loss = torch.nn.functional.mse_loss(pred, tgt)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
            if self.logger:
                self.logger.info(f"[MetaTrainer] epoch {epoch+1}/{epochs} loss={loss.item():.6f}")

    def save(self, path=None):
        if path is None:
            path = os.path.join(self.save_dir, "meta_agent.pt")
        torch.save(self.meta_agent.state_dict(), path)
        if self.logger:
            self.logger.info(f"[MetaTrainer] saved meta_agent to {path}")
