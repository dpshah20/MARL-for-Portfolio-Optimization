import torch
import torch.optim as optim
import math

class MetaTrainer:
    """
    Deterministic Meta-Agent trainer.
    Optimizes weekly return directly.
    """
    def __init__(
        self,
        meta_agent,
        lr=1e-3,
        entropy_coef: float = 1e-3,
        baseline_momentum: float = 0.95,
        eps: float = 1e-8,
    ):
        self.agent = meta_agent
        self.optimizer = optim.Adam(self.agent.parameters(), lr=lr)
        self.entropy_coef = float(entropy_coef)
        self.baseline_momentum = float(baseline_momentum)
        self.eps = float(eps)

        # Running return moments for variance-reduced policy gradients.
        self.ret_ema = 0.0
        self.ret_var_ema = 1.0
        self._initialized = False

    def update_policy(self, weekly_return: float, log_prob: torch.Tensor, entropy: torch.Tensor = None):
        """
        Advantage-normalized REINFORCE with entropy regularization.
        """
        ret = float(weekly_return)

        if not self._initialized:
            self.ret_ema = ret
            self.ret_var_ema = 1.0
            self._initialized = True

        delta = ret - self.ret_ema
        self.ret_ema = self.baseline_momentum * self.ret_ema + (1.0 - self.baseline_momentum) * ret
        self.ret_var_ema = self.baseline_momentum * self.ret_var_ema + (1.0 - self.baseline_momentum) * (delta ** 2)

        advantage = (ret - self.ret_ema) / math.sqrt(self.ret_var_ema + self.eps)
        adv_tensor = torch.tensor(advantage, dtype=log_prob.dtype, device=log_prob.device)

        pg_obj = log_prob * adv_tensor
        if entropy is not None:
            obj = pg_obj + self.entropy_coef * entropy
        else:
            obj = pg_obj

        loss = -obj.mean()
        
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.agent.parameters(), 1.0)
        self.optimizer.step()
        return loss.item()
