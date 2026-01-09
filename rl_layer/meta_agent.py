import torch
import torch.nn as nn
from torch.distributions import Categorical
import itertools

class MetaAgent(nn.Module):
    """
    Meta-Agent module:
    Learns to select a high-level strategy (w, rho) based on macroeconomic state features.
    Output is a Categorical distribution over discrete (w, rho) pairs.
    """
    def __init__(self, input_dim: int, hidden_dim: int = 128):
        super().__init__()
        
        # --- 1. Define Discrete Action Candidates ---
        
        # W candidates: [w_ret, w_vol, w_cvar, w_mdd]
        # These represent distinct risk profiles the agent can choose.
        self.w_candidates = torch.tensor([
            [1.0, 0.0, 0.0, 0.0],  # 0: Aggressive (Pure Return maximization)
            [0.6, 0.2, 0.1, 0.1],  # 1: Growth (Focus on return but punish extreme risks)
            [0.4, 0.2, 0.2, 0.2],  # 2: Balanced (Standard portfolio approach)
            [0.1, 0.3, 0.3, 0.3],  # 3: Defensive (Prioritize stability over growth)
            [0.0, 0.2, 0.4, 0.4]   # 4: Ultra-Defensive (Minimize Tail Risk & Drawdown)
        ], dtype=torch.float32)
        
        # Rho candidates: Cash Fraction
        # 0.05 = 5% Cash (Fully Invested), 0.90 = 90% Cash (Market Exit)
        self.rho_candidates = torch.tensor(
            [0.05, 0.25, 0.50, 0.75, 0.90], 
            dtype=torch.float32
        )

        # Create Cartesian Product of actions (Total = 5 x 5 = 25 combinations)
        # Example Action 0: (Aggressive, 5% Cash)
        # Example Action 24: (Ultra-Defensive, 90% Cash)
        self.action_combinations = list(itertools.product(
            range(len(self.w_candidates)), 
            range(len(self.rho_candidates))
        ))
        self.output_dim = len(self.action_combinations)

        # --- 2. Policy Network ---
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),  # Tanh often works better for policy networks than ReLU
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, self.output_dim) 
            # Output logits for Categorical distribution (Softmax applied implicitly by Categorical)
        )

    def forward(self, state):
        """
        state: (B, input_dim)
        Returns: logits (B, output_dim)
        """
        return self.net(state)

    def sample_action(self, state):
        """
        Samples a discrete action (w, rho) based on the state.
        
        Returns: 
            w_selected (np.array): The chosen weight vector
            rho_selected (float): The chosen cash fraction
            log_prob (Tensor): Log probability of the action (for REINFORCE update)
            entropy (Tensor): Entropy of the distribution (for exploration bonus)
            action_idx (int): The integer index of the chosen action
        """
        logits = self.forward(state)
        dist = Categorical(logits=logits)
        
        action_idx = dist.sample()
        log_prob = dist.log_prob(action_idx)
        entropy = dist.entropy()

        # Decode Action Index -> (W_idx, Rho_idx)
        # We use .item() to get standard python ints/floats from tensors
        w_idx, rho_idx = self.action_combinations[action_idx.item()]
        
        w_selected = self.w_candidates[w_idx].numpy()
        rho_selected = self.rho_candidates[rho_idx].item()
        
        return w_selected, rho_selected, log_prob, entropy, action_idx.item()