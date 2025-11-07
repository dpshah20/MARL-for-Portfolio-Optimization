# training/train_meta_agent.py
import torch
import torch.optim as optim
import numpy as np

class MetaTrainer:
    def __init__(self, meta_agent, lr=3e-4, device="cpu"):
        self.agent = meta_agent.to(device)
        self.opt = optim.Adam(self.agent.parameters(), lr=lr)
        self.device = device

    def train(self, X: np.ndarray, returns: np.ndarray, epochs: int = 10, batch_size: int = 32):
        X = torch.tensor(X, dtype=torch.float32).to(self.device)
        R = torch.tensor(returns, dtype=torch.float32).to(self.device)
        N = len(X)
        for e in range(epochs):
            idx = np.random.permutation(N)
            for start in range(0, N, batch_size):
                batch_idx = idx[start:start+batch_size]
                xb = X[batch_idx]
                rb = R[batch_idx]
                rho, w = self.agent(xb)
                # surrogate objective: favor rho when future returns positive
                adv = rb - rb.mean()
                entropy = - (w * (w+1e-12).log()).sum(dim=1)
                obj = (adv * rho).mean() + 0.01 * (adv * entropy).mean()
                loss = -obj
                self.opt.zero_grad()
                loss.backward()
                self.opt.step()
        return
