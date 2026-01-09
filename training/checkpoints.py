import os
import torch
import logging

class CheckpointManager:
    def __init__(self, checkpoint_dir="checkpoints", logger=None):
        self.checkpoint_dir = checkpoint_dir
        self.logger = logger
        os.makedirs(self.checkpoint_dir, exist_ok=True)

    def save(self, step, actor=None, critic=None, meta_agent=None):
        """
        Saves state dicts for all active agents.
        """
        path = os.path.join(self.checkpoint_dir, f"ckpt_step_{step}.pt")
        state = {"step": step}
        
        if actor:
            state["actor_state"] = actor.state_dict()
        if critic:
            state["critic_state"] = critic.state_dict()
        if meta_agent:
            state["meta_state"] = meta_agent.state_dict()
            
        torch.save(state, path)
        # Only log if logger is provided to avoid spamming if not set up
        if self.logger:
            self.logger.info(f"Saved checkpoint to {path}")

    def load(self, path=None, actor=None, critic=None, meta_agent=None):
        """
        Loads state dicts into provided models.
        If path is None, loads the latest checkpoint in the directory.
        """
        # 1. Determine Path
        if path is None:
            return self.load_latest(actor, critic, meta_agent)
            
        if not os.path.exists(path):
            if self.logger:
                self.logger.warning(f"Checkpoint {path} not found.")
            print(f"Checkpoint {path} not found.")
            return 0
            
        # 2. Load State
        try:
            state = torch.load(path, map_location="cpu") # Safe load to CPU first
            step = state.get("step", 0)
            
            if actor and "actor_state" in state:
                actor.load_state_dict(state["actor_state"])
            if critic and "critic_state" in state:
                critic.load_state_dict(state["critic_state"])
            if meta_agent and "meta_state" in state:
                meta_agent.load_state_dict(state["meta_state"])
                
            if self.logger:
                self.logger.info(f"Loaded checkpoint from step {step}")
            else:
                print(f"Loaded checkpoint from step {step}")
                
            return step
        except Exception as e:
            if self.logger:
                self.logger.error(f"Failed to load checkpoint: {e}")
            print(f"Failed to load checkpoint: {e}")
            return 0

    def load_latest(self, actor=None, critic=None, meta_agent=None):
        """Finds and loads the checkpoint with the highest step number."""
        import glob
        files = glob.glob(os.path.join(self.checkpoint_dir, "ckpt_step_*.pt"))
        if not files:
            if self.logger:
                self.logger.info("No checkpoints found to resume.")
            return 0
            
        # Parse step numbers
        # Filename format: ckpt_step_123.pt
        try:
            latest_file = max(files, key=lambda x: int(x.split("_")[-1].split(".")[0]))
            return self.load(latest_file, actor, critic, meta_agent)
        except Exception as e:
            if self.logger:
                self.logger.error(f"Error finding latest checkpoint: {e}")
            return 0