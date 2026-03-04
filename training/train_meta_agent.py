import torch
import torch.optim as optim

class MetaTrainer:
    """
    Deterministic Meta-Agent trainer.
    Optimizes weekly return directly.
    """
    def __init__(self, meta_agent, lr=1e-3):
        self.agent = meta_agent
        self.optimizer = optim.Adam(self.agent.parameters(), lr=lr)

    def update_policy(self, weekly_return: float, log_prob: torch.Tensor):
        """
        REINFORCE update: loss = -log_prob * return
        """
        # Normalize return? Maybe not for now.
        # We want to maximize return, so minimize -return.
        # With log_prob: minimize -(log_prob * return)
        
        loss = -(log_prob * weekly_return)
        
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.agent.parameters(), 1.0)
        self.optimizer.step()
        return loss.item()
