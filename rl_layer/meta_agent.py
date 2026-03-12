import torch
import torch.nn as nn
import torch.nn.functional as F

class MetaAgent(nn.Module):
    """
    Continuous Meta-Agent
    Outputs:
      w   : (B,4) risk weights (softmax)
      rho : (B,)  cash fraction (sigmoid)
    """
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
        rho_min: float = 0.05,
        rho_max: float = 0.30,
        init_std_w: float = 0.35,
        init_std_rho: float = 0.20,
    ):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh()
        )
        self.head_w = nn.Linear(hidden_dim, 4)
        self.head_rho = nn.Linear(hidden_dim, 1)

        # Learnable exploration scales for stable policy-gradient updates.
        self.log_std_w = nn.Parameter(torch.full((4,), float(torch.log(torch.tensor(init_std_w)))))
        self.log_std_rho = nn.Parameter(torch.tensor(float(torch.log(torch.tensor(init_std_rho)))))

        self.rho_min = float(rho_min)
        self.rho_max = float(rho_max)

    def _scale_rho(self, rho_unit: torch.Tensor) -> torch.Tensor:
        """Map rho from unit interval to configured bounds."""
        return self.rho_min + (self.rho_max - self.rho_min) * rho_unit

    def forward(self, state):
        h = self.backbone(state)
        logits_w = self.head_w(h)
        rho_logit = self.head_rho(h).squeeze(-1)

        w = torch.softmax(logits_w, dim=-1)
        rho = self._scale_rho(torch.sigmoid(rho_logit))
        return w, rho

    def get_action(self, state):
        """
        Sample action for REINFORCE.
        w: Sample from Dirichlet (approximated via Softmax + Noise)
        rho: Sample from Beta (approximated via Sigmoid + Noise)
        """
        h = self.backbone(state)
        logits_w = self.head_w(h)
        rho_logit = self.head_rho(h).squeeze(-1)

        std_w = torch.clamp(F.softplus(self.log_std_w), min=1e-3, max=2.0)
        std_rho = torch.clamp(F.softplus(self.log_std_rho), min=1e-3, max=2.0)

        # Sample logits then squash to simplex.
        dist_w = torch.distributions.Normal(logits_w, std_w.unsqueeze(0))
        action_logits_w = dist_w.rsample()
        w = torch.softmax(action_logits_w, dim=-1)
        log_prob_w = dist_w.log_prob(action_logits_w).sum(dim=-1)
        entropy_w = dist_w.entropy().sum(dim=-1)

        # Sample rho-logit, map to bounded rho interval.
        dist_rho = torch.distributions.Normal(rho_logit, std_rho)
        action_logit_rho = dist_rho.rsample()
        rho_unit = torch.sigmoid(action_logit_rho)
        rho = self._scale_rho(rho_unit)
        log_prob_rho = dist_rho.log_prob(action_logit_rho)
        entropy_rho = dist_rho.entropy()

        total_log_prob = log_prob_w + log_prob_rho
        total_entropy = entropy_w + entropy_rho
        return w, rho, total_log_prob, total_entropy

