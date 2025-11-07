# rl_layer/meta_agent.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class MetaAgent(nn.Module):
    def __init__(self, input_dim: int, hidden: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.LayerNorm(hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden//2),
            nn.ReLU()
        )
        self.head_rho = nn.Linear(hidden//2, 1)
        self.head_w = nn.Linear(hidden//2, 4)

    def forward(self, x):
        h = self.net(x)
        rho = torch.sigmoid(self.head_rho(h)) * 1.0 + 0.5  # maps to (0.5,1.5)
        logits = self.head_w(h)
        w = F.softmax(logits, dim=-1)
        return rho.squeeze(-1), w
