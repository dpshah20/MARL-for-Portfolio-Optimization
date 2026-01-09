import torch
import torch.optim as optim
import numpy as np

class MetaTrainer:
    """
    Trainer for the Meta-Agent using REINFORCE (Policy Gradient).
    Updates the policy based on the realized Weekly Return compared to a Baseline.
    """
    def __init__(self, meta_agent, lr=1e-3, gamma=0.99, entropy_coef=0.05):
        self.agent = meta_agent
        self.optimizer = optim.Adam(self.agent.parameters(), lr=lr)
        self.gamma = gamma
        self.entropy_coef = entropy_coef
        
        # Baseline for Variance Reduction (EMA of weekly returns)
        # This helps the agent distinguish "good" weeks from "lucky" weeks.
        self.baseline = 0.0
        self.alpha = 0.1  # Smoothing factor for baseline

    def update_policy(self, weekly_return: float, log_prob: torch.Tensor, entropy: torch.Tensor):
        """
        Performs one REINFORCE update step at the end of the week.
        
        Args:
            weekly_return (float): The actual return achieved this week (NAV_end / NAV_start - 1).
            log_prob (Tensor): The log probability of the action that was taken on Monday.
            entropy (Tensor): The entropy of the policy distribution (for exploration).
            
        Returns:
            loss (float): The total training loss.
            advantage (float): How much better the return was compared to the baseline.
        """
        # 1. Update Baseline (Exponential Moving Average)
        if self.baseline == 0.0:
            self.baseline = weekly_return
        else:
            self.baseline = (1 - self.alpha) * self.baseline + self.alpha * weekly_return

        # 2. Compute Advantage
        # If return > baseline, advantage is positive -> Increase prob of action.
        # If return < baseline, advantage is negative -> Decrease prob of action.
        advantage = weekly_return - self.baseline

        # 3. Compute Loss
        # Policy Gradient Loss: - (Advantage * log_prob)
        # Entropy Loss: - (Entropy_Coef * Entropy) -> Encourages higher entropy (exploration)
        policy_loss = -(advantage * log_prob)
        entropy_loss = - (self.entropy_coef * entropy)
        
        total_loss = policy_loss + entropy_loss

        # 4. Backpropagation
        self.optimizer.zero_grad()
        total_loss.backward()
        
        # Clip gradients to prevent instability
        torch.nn.utils.clip_grad_norm_(self.agent.parameters(), 1.0)
        
        self.optimizer.step()

        return total_loss.item(), advantage