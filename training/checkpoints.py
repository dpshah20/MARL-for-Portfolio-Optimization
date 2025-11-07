import os
import torch
import json


class CheckpointManager:
    """
    Handles saving and loading of model, critic, and meta-agent checkpoints.
    Keeps the latest few checkpoints to avoid storage bloat.
    """

    def __init__(self, save_dir="checkpoints", prefix="rl_agent", max_to_keep=5):
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
        self.prefix = prefix
        self.max_to_keep = max_to_keep
        self.saved_checkpoints = []

    def save(self, step, actor, critic, meta_agent=None, replay_buffer=None):
        """
        Save model state dictionaries and metadata.
        """
        ckpt_path = os.path.join(self.save_dir, f"{self.prefix}_step_{step}.pt")
        payload = {
            "step": step,
            "actor": actor.state_dict() if actor else None,
            "critic": critic.state_dict() if critic else None,
            "meta_agent": meta_agent.state_dict() if meta_agent else None,
        }
        torch.save(payload, ckpt_path)
        self.saved_checkpoints.append(ckpt_path)

        # Manage retention (keep only last N)
        if len(self.saved_checkpoints) > self.max_to_keep:
            oldest = self.saved_checkpoints.pop(0)
            if os.path.exists(oldest):
                os.remove(oldest)

        # Optionally save replay buffer metadata
        if replay_buffer:
            buffer_meta = {"size": len(replay_buffer), "capacity": replay_buffer.capacity}
            with open(os.path.join(self.save_dir, f"{self.prefix}_buffer_meta.json"), "w") as f:
                json.dump(buffer_meta, f, indent=2)

        print(f"[CheckpointManager] ✅ Saved checkpoint: {ckpt_path}")

    def load_latest(self, actor=None, critic=None, meta_agent=None):
        """
        Load the most recent checkpoint from the directory.
        Returns the step number to resume from.
        """
        ckpts = [f for f in os.listdir(self.save_dir) if f.endswith(".pt")]
        if not ckpts:
            print("[CheckpointManager] ⚠️ No checkpoints found. Starting fresh.")
            return 0

        ckpts.sort(key=lambda x: os.path.getmtime(os.path.join(self.save_dir, x)))
        latest_path = os.path.join(self.save_dir, ckpts[-1])
        payload = torch.load(latest_path, map_location="cpu")

        if actor and payload.get("actor"):
            actor.load_state_dict(payload["actor"])
        if critic and payload.get("critic"):
            critic.load_state_dict(payload["critic"])
        if meta_agent and payload.get("meta_agent"):
            meta_agent.load_state_dict(payload["meta_agent"])

        print(f"[CheckpointManager] 🔄 Loaded checkpoint: {latest_path}")
        return payload.get("step", 0)

    def save_actor(self, step, actor):
        torch.save(actor.state_dict(),
                      os.path.join(self.save_dir, f"actor_{step}.pth"))

    def save_critic(self, step, critic):
        torch.save(critic.state_dict(),
                      os.path.join(self.save_dir, f"critic_{step}.pth"))

    def save_meta_agent(self, step, meta_agent):
        torch.save(meta_agent.state_dict(),
                      os.path.join(self.save_dir, f"meta_{step}.pth"))

    def load(self, step, actor=None, critic=None, meta_agent=None):
        if actor is not None:
            actor.load_state_dict(torch.load(
                os.path.join(self.save_dir, f"actor_{step}.pth")))
        if critic is not None:
            critic.load_state_dict(torch.load(
                os.path.join(self.save_dir, f"critic_{step}.pth")))
        if meta_agent is not None:
            meta_agent.load_state_dict(torch.load(
                os.path.join(self.save_dir, f"meta_{step}.pth")))

    def load_latest(self, actor=None, critic=None, meta_agent=None):
        files = os.listdir(self.save_dir)
        if not files:
            return

        steps = []
        for f in files:
            if f.startswith("actor_") and f.endswith(".pth"):
                try:
                    step = int(f.split("_")[1].split(".")[0])
                    steps.append(step)
                except:
                    continue

        if steps:
            latest = max(steps)
            self.load(latest, actor, critic, meta_agent)
