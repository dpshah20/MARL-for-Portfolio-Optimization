# rl_layer/critic_distributional.py
"""
Quantile Regression distributional critic (QR).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class QuantileCritic(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, hidden: int = 512, Nq: int = 51):
        super().__init__()
        self.Nq = Nq
        self.net = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden//2),
            nn.ReLU()
        )
        self.quantile_head = nn.Linear(hidden//2, Nq)

    def forward(self, state_flat, actions):
        x = torch.cat([state_flat, actions], dim=-1)
        h = self.net(x)
        q = self.quantile_head(h)
        return q

def quantile_huber_loss(predictions, targets, taus, kappa=1.0):
    # predictions: (B,Nq), targets: (B,Nq), taus: (Nq,)
    diff = targets.unsqueeze(-2) - predictions.unsqueeze(-1)  # (B,Nq,Nq)
    abs_diff = diff.abs()
    huber = torch.where(abs_diff <= kappa, 0.5 * diff**2, kappa * (abs_diff - 0.5 * kappa))
    tau = taus.view(1, -1, 1)
    loss = (torch.abs(tau - (diff.detach() < 0).float()) * huber).mean()
    return loss
