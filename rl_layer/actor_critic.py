import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from rl_layer.critic_distributional import QuantileCritic, quantile_huber_loss

class ActorNet(nn.Module):
    def __init__(self, in_dim, hidden=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1)
        )

    def forward(self, x):
        if x.dim() == 2:
            x = x.unsqueeze(0)
        B, N, d = x.shape
        out = self.net(x.view(B * N, d)).view(B, N)
        return out


class MADDPG:
    def __init__(
        self,
        d_gnn,
        n_assets,
        actor_lr=1e-4,
        critic_lr=1e-4,
        gamma=0.99,
        tau=0.005,
        Nq=51,
        device="cpu",
    ):
        self.device = device
        self.gamma = gamma
        self.tau = tau
        self.Nq = Nq
        self.n_assets = n_assets
        self.d_gnn = d_gnn

        self.actor = ActorNet(d_gnn).to(device)
        self.actor_target = ActorNet(d_gnn).to(device)
        self.actor_target.load_state_dict(self.actor.state_dict())

        self.critic = QuantileCritic(
            state_dim=d_gnn * n_assets,
            action_dim=n_assets,
            hidden=512,
            Nq=Nq,
        ).to(device)

        self.critic_target = QuantileCritic(
            state_dim=d_gnn * n_assets,
            action_dim=n_assets,
            hidden=512,
            Nq=Nq,
        ).to(device)

        self.critic_target.load_state_dict(self.critic.state_dict())

        self.actor_opt = optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_opt = optim.Adam(self.critic.parameters(), lr=critic_lr)

        taus = (torch.arange(Nq) + 0.5) / Nq
        self.taus = taus.to(device)

    def update_critic(self, s, a, r, s2, done):
        q = self.critic(s, a)
        with torch.no_grad():
            q2 = self.critic_target(s2, a)
            target = r + (1 - done) * self.gamma * q2
        loss = quantile_huber_loss(q, target, self.taus)

        self.critic_opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 1.0)
        self.critic_opt.step()
        return loss.item()

    def update_actor(self, z):
        """
        z: (B, N, D_enc) from encoder
        Critic expects ONLY first d_gnn dims per stock
        """
        self.actor_opt.zero_grad()

        scores = self.actor(z)              # (B, N)
        actions = F.softmax(scores, dim=1)  # (B, N)

        # 🔑 Slice encoder output to match critic expectation
        B, N, _ = z.shape
        z_core = z[:, :, :self.actor.net[0].in_features]  # = d_gnn

        state_flat = z_core.contiguous().view(B, N * z_core.size(-1))

        q = self.critic(state_flat, actions)
        loss = -q.mean()

        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 1.0)
        self.actor_opt.step()

        return loss.item()


    def update_targets(self):
        for p, pt in zip(self.actor.parameters(), self.actor_target.parameters()):
            pt.data.copy_(self.tau * p.data + (1 - self.tau) * pt.data)
        for p, pt in zip(self.critic.parameters(), self.critic_target.parameters()):
            pt.data.copy_(self.tau * p.data + (1 - self.tau) * pt.data)
