# rl_layer/meta_agent.py
"""
Meta-agent module:
Learns reward-weighting vector (ρ_t) and component weights W_meta
based on macroeconomic and portfolio state features.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class MetaAgent(nn.Module):
    def __init__(self, input_dim: int = 64, hidden_dim: int = 128, output_dim: int = 4):
        """
        input_dim: dimension of macro + RL context features
        hidden_dim: hidden size for MLP
        output_dim: number of reward components (ret, vol, cvar, mdd)
        """
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.rho_head = nn.Linear(hidden_dim, 1)         # scalar ρ_t
        self.weights_head = nn.Linear(hidden_dim, output_dim)  # reward component weights

    def forward(self, x):
        """
        x: (B, input_dim)
        Returns:
          - rho: (B, 1), scaling factor for reward weighting
          - w: (B, output_dim), normalized softmax weights for reward components
        """
        h = F.relu(self.fc1(x))
        h = F.relu(self.fc2(h))
        rho = torch.sigmoid(self.rho_head(h)) * 1.5  # restrict between (0,1.5)
        w_raw = self.weights_head(h)
        w = F.softmax(w_raw, dim=-1)
        return rho, w
