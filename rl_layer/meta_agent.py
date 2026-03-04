import torch
import torch.nn as nn

class MetaAgent(nn.Module):
    """
    Continuous Meta-Agent
    Outputs:
      w   : (B,4) risk weights (softmax)
      rho : (B,)  cash fraction (sigmoid)
    """
    def __init__(self, input_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 5)  # 4 weights + 1 rho
        )

    def forward(self, state):
        out = self.net(state)
        w = torch.softmax(out[:, :4], dim=-1)
        rho = torch.sigmoid(out[:, 4])
        return w, rho

    def get_action(self, state):
        """
        Sample action for REINFORCE.
        w: Sample from Dirichlet (approximated via Softmax + Noise)
        rho: Sample from Beta (approximated via Sigmoid + Noise)
        """
        out = self.net(state)
        
        # 1. Weights (w) - Add Gumbel noise for exploration on simplex
        logits_w = out[:, :4]
        # Gumbel noise: -log(-log(U))
        u = torch.rand_like(logits_w)
        gumbel = -torch.log(-torch.log(u + 1e-10) + 1e-10)
        w_sample = torch.softmax(logits_w + gumbel, dim=-1)
        
        # Log prob of softmax sample is complex, we'll use a simplified proxy
        # or just use the logits' log_softmax as the "action distribution"
        # For REINFORCE, we need log_prob(action). 
        # If we treat the output as parameters of a distribution:
        # Let's assume a Dirichlet distribution parameterized by exp(logits).
        # But for simplicity, let's just add Gaussian noise to logits and take softmax.
        
        # Alternative: Use simple Gaussian exploration on the logits
        dist_w = torch.distributions.Normal(logits_w, 0.5) # Fixed std dev for exploration
        action_logits_w = dist_w.sample()
        w = torch.softmax(action_logits_w, dim=-1)
        log_prob_w = dist_w.log_prob(action_logits_w).sum(dim=-1)
        
        # 2. Cash (rho) - Beta distribution or Normal on logit
        logit_rho = out[:, 4]
        dist_rho = torch.distributions.Normal(logit_rho, 0.5)
        action_logit_rho = dist_rho.sample()
        rho = torch.sigmoid(action_logit_rho)
        log_prob_rho = dist_rho.log_prob(action_logit_rho)
        
        return w, rho, log_prob_w + log_prob_rho

