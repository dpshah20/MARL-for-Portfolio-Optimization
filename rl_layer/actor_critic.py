import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from rl_layer.critic_distributional import QuantileCritic, quantile_huber_loss

class ActorNet(nn.Module):
    def __init__(self, in_dim, hidden=128, mem_dim=64):
        super().__init__()
        self.in_dim = in_dim
        self.mem_dim = mem_dim

        self.memory_cell = nn.GRUCell(in_dim, mem_dim)
        self.net = nn.Sequential(
            nn.Linear(in_dim + mem_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1)
        )

    def init_hidden(self, batch_size, device=None):
        if device is None:
            device = next(self.parameters()).device
        return torch.zeros(batch_size, self.mem_dim, device=device)

    def forward(self, x, hidden=None):
        if x.dim() == 2:
            x = x.unsqueeze(0)
        B, N, d = x.shape
        if hidden is None:
            hidden = self.init_hidden(B, device=x.device)

        market_context = x.mean(dim=1)
        next_hidden = self.memory_cell(market_context, hidden)

        hidden_expand = next_hidden.unsqueeze(1).expand(-1, N, -1)
        actor_in = torch.cat([x, hidden_expand], dim=-1)
        out = self.net(actor_in.view(B * N, d + self.mem_dim)).view(B, N)
        return out, next_hidden


class MADDPG:
    def __init__(
        self,
        d_gnn,
        n_assets,
        actor_lr,
        critic_lr,
        gamma,
        tau,
        Nq,
        actor_hidden,
        actor_mem_dim,
        critic_hidden,
        device="cpu",
    ):
        self.device = device
        self.gamma = gamma
        self.tau = tau
        self.Nq = Nq
        self.n_assets = n_assets
        self.d_gnn = d_gnn

        self.actor = ActorNet(d_gnn, hidden=actor_hidden, mem_dim=actor_mem_dim).to(device)
        self.actor_target = ActorNet(d_gnn, hidden=actor_hidden, mem_dim=actor_mem_dim).to(device)
        self.actor_target.load_state_dict(self.actor.state_dict())

        self.critic = QuantileCritic(
            state_dim=d_gnn * n_assets,
            action_dim=n_assets,
            hidden=critic_hidden,
            Nq=Nq,
        ).to(device)

        self.critic_target = QuantileCritic(
            state_dim=d_gnn * n_assets,
            action_dim=n_assets,
            hidden=critic_hidden,
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

    def update_actor(self, z, hidden_in=None):
        """
        z: (B, N, D_enc) from encoder
        Critic expects ONLY first d_gnn dims per stock
        """
        self.actor_opt.zero_grad()

        scores, _ = self.actor(z, hidden=hidden_in)  # (B, N)
        actions = F.softmax(scores, dim=1)  # (B, N)

        # 🔑 Slice encoder output to match critic expectation
        B, N, _ = z.shape
        z_core = z[:, :, :self.d_gnn]

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
