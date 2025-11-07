# rl_layer/actor_critic.py
"""
MADDPG actor + centralized distributional critic trainer.
Actor outputs per-node score in (0,1). For actor updates we use a differentiable
soft-top-k approximation (softmax with temperature + masking) to produce soft allocations
for backprop. Actual execution uses hard top-k + rebalance_manager.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from rl_layer.critic_distributional import QuantileCritic, quantile_huber_loss

class ActorNet(nn.Module):
    def __init__(self, in_dim: int, hidden: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden//2),
            nn.ReLU(),
            nn.Linear(hidden//2, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        # x: (B,N,d)
        single = False
        if x.dim()==2:
            single = True
            x = x.unsqueeze(0)
        B, N, d = x.shape
        x_flat = x.view(B*N, d)
        out = self.net(x_flat).view(B, N)
        if single:
            return out.squeeze(0)
        return out

class MADDPG:
    def __init__(self, d_gnn: int, K: int = 10, Nq: int = 51,
                 actor_lr: float = 1e-4, critic_lr: float = 1e-4,
                 gamma: float = 0.99, tau: float = 0.005, device: str = "cpu"):
        self.device = device
        self.d_gnn = d_gnn
        self.K = K
        self.actor = ActorNet(d_gnn).to(device)
        self.actor_target = ActorNet(d_gnn).to(device)
        self.actor_target.load_state_dict(self.actor.state_dict())

        state_dim = K * d_gnn
        action_dim = K
        self.critic = QuantileCritic(state_dim, action_dim, hidden=512, Nq=Nq).to(device)
        self.critic_target = QuantileCritic(state_dim, action_dim, hidden=512, Nq=Nq).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())

        self.actor_opt = optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_opt = optim.Adam(self.critic.parameters(), lr=critic_lr)

        self.Nq = Nq
        taus = (torch.arange(Nq, dtype=torch.float32) + 0.5) / Nq
        self.taus = taus.to(device)
        self.gamma = gamma
        self.tau = tau

    def soft_update(self, src, tgt):
        for p_s, p_t in zip(src.parameters(), tgt.parameters()):
            p_t.data.copy_(self.tau * p_s.data + (1.0 - self.tau) * p_t.data)

    def update_targets(self):
        self.soft_update(self.actor, self.actor_target)
        self.soft_update(self.critic, self.critic_target)

    def infer_actor(self, z_batch):
        return self.actor(z_batch)

    def infer_actor_target(self, z_batch):
        return self.actor_target(z_batch)

    def critic_loss(self, state_flat, actions, reward, next_state_flat, next_actions, done):
        q_pred = self.critic(state_flat, actions)  # (B,Nq)
        with torch.no_grad():
            q_next = self.critic_target(next_state_flat, next_actions)
            target = reward.unsqueeze(-1) + (1.0 - done.unsqueeze(-1)) * (self.gamma * q_next)
        loss = quantile_huber_loss(q_pred, target, self.taus)
        return loss

    def update_critic(self, state_flat, actions, reward, next_state_flat, next_actions, done):
        self.critic_opt.zero_grad()
        loss = self.critic_loss(state_flat, actions, reward, next_state_flat, next_actions, done)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 1.0)
        self.critic_opt.step()
        return loss.item()

    def update_actor(self, z_batch, state_flat_builder, selected_idx_batch):
        """
        z_batch: (B,N,d)
        state_flat_builder: function mapping z_batch and actor scores -> state_flat for critic (handles K selection and ordering)
        selected_idx_batch: list of lists with indices of selected K (used for mapping)
        We'll compute soft allocations for actor update using softmax temperature trick:
           - convert actor scores -> soft allocation across full N
           - mask to top-K via top-k soft selection (differentiable approx)
        For simplicity, we approximate by taking softmax(scores / temp) and then zeroing small entries via top-k mask (non-differentiable).
        A better approach would use differentiable top-k; here we implement softmax + scaling restricted to top-K indices per sample
        """
        self.actor_opt.zero_grad()
        # produce scores
        scores = self.actor(z_batch)  # (B,N)
        # softmax across N to get soft weights
        temp = 0.05
        soft_w = F.softmax(scores / temp, dim=1)  # (B,N)
        # for each batch element, zero-out weights not in selected_idx_batch (non-diff op) -> still provides useful gradient via soft_w for top entries
        batch_actions = []
        for b in range(z_batch.size(0)):
            sel_idx = selected_idx_batch[b]
            if len(sel_idx)==0:
                act = torch.zeros(self.K, device=self.device)
            else:
                vec = soft_w[b, sel_idx]  # (K,)
                # normalize to investable fraction (we assume investable=1 - min_cash)
                investable = 1.0
                act = vec / (vec.sum()+1e-12) * investable
                # ensure length K (pad if fewer selected)
                if len(sel_idx) < self.K:
                    pad = torch.zeros(self.K - len(sel_idx), device=self.device)
                    act = torch.cat([act, pad], dim=0)
            batch_actions.append(act.unsqueeze(0))
        actions_tensor = torch.cat(batch_actions, dim=0)  # (B,K)
        # build state_flat from z_batch using provided builder
        state_flat = state_flat_builder(z_batch, selected_idx_batch)  # returns (B, K*d_gnn)
        # get critic quantiles -> take mean across quantiles as surrogate Q
        q = self.critic(state_flat, actions_tensor)  # (B,Nq)
        q_mean = q.mean(dim=1)  # (B,)
        # maximize q_mean -> minimize -q_mean
        loss = -q_mean.mean()
        loss.backward()
        self.actor_opt.step()
        return loss.item()
