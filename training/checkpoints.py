# training/checkpoints.py
import os
import torch
import json

class CheckpointManager:
    def __init__(self, save_dir="checkpoints", max_to_keep=5):
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
        self.max_to_keep = max_to_keep
        self._saved = []

    def save(self, step, actor=None, critic=None, meta_agent=None, replay_buffer=None):
        fname = os.path.join(self.save_dir, f"ckpt_step_{step}.pt")
        payload = {"step": step}
        if actor is not None:
            payload["actor"] = actor.state_dict()
        if critic is not None:
            payload["critic"] = critic.state_dict()
        if meta_agent is not None:
            payload["meta_agent"] = meta_agent.state_dict()
        torch.save(payload, fname)
        self._saved.append(fname)
        # rotate
        if len(self._saved) > self.max_to_keep:
            old = self._saved.pop(0)
            try:
                os.remove(old)
            except:
                pass
        # optional replay metadata
        if replay_buffer is not None:
            meta = {"size": len(replay_buffer), "capacity": getattr(replay_buffer, "capacity", None)}
            with open(os.path.join(self.save_dir, "replay_meta.json"), "w") as f:
                json.dump(meta, f, indent=2)
        print(f"[CheckpointManager] Saved {fname}")

    def load_latest(self, actor=None, critic=None, meta_agent=None, device="cpu"):
        files = [f for f in os.listdir(self.save_dir) if f.endswith(".pt")]
        if not files:
            print("[CheckpointManager] No checkpoint found.")
            return 0
        files.sort(key=lambda f: os.path.getmtime(os.path.join(self.save_dir, f)))
        latest = os.path.join(self.save_dir, files[-1])
        data = torch.load(latest, map_location=device)
        if actor is not None and "actor" in data:
            actor.load_state_dict(data["actor"])
        if critic is not None and "critic" in data:
            critic.load_state_dict(data["critic"])
        if meta_agent is not None and "meta_agent" in data:
            meta_agent.load_state_dict(data["meta_agent"])
        print(f"[CheckpointManager] Loaded {latest}")
        return data.get("step", 0)
