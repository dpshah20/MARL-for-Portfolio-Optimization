# training/train_meta_agent.py
import os
import torch
import numpy as np
from rl_layer.meta_agent import MetaAgent
from training.checkpoints import CheckpointManager

class MetaTrainer:
    def __init__(self, cfg, logger=None, ckpt_mgr: CheckpointManager = None, device="cpu"):
        self.cfg = cfg
        self.logger = logger
        self.ckpt_mgr = ckpt_mgr
        self.device = device

        # initialize meta agent
        self.meta_agent = MetaAgent(
            input_dim=cfg.get("meta_input_dim", 64),
            hidden_dim=cfg.get("meta_hidden_dim", 128),
            output_dim=cfg.get("meta_output_dim", 4)
        ).to(device)

        # optimizer & hyperparams
        self.lr = cfg.get("meta_lr", 3e-4)
        self.optimizer = torch.optim.Adam(self.meta_agent.parameters(), lr=self.lr)
        self.epochs = cfg.get("meta_epochs", 10)
        self.save_dir = cfg.get("checkpoint_dir", "checkpoints/meta_agent")
        os.makedirs(self.save_dir, exist_ok=True)

    def step_weekly(self, meta_states, reward_stats):
        """
        Perform one weekly meta-agent training step.
        meta_states: np.ndarray (N, d_meta)
        reward_stats: np.ndarray (N, num_components)
        """
        self.meta_agent.train()
        x = torch.tensor(meta_states, dtype=torch.float32).to(self.device)
        y = torch.tensor(reward_stats, dtype=torch.float32).to(self.device)

        preds = self.meta_agent(x)
        loss = torch.nn.functional.mse_loss(preds, y)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        if self.logger:
            self.logger.info(f"[MetaTrainer] Weekly meta update complete | Loss={loss.item():.6f}")

    def save_all(self):
        if self.ckpt_mgr:
            self.ckpt_mgr.save(
                step=0,  # meta agent step index not tracked yet
                actor=None,
                critic=None,
                meta_agent=self.meta_agent
            )
