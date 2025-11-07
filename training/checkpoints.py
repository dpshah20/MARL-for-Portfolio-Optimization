# training/checkpoints.py
import os
import torch
import pickle
import json

def save_all(path: str, state: dict):
    os.makedirs(path, exist_ok=True)
    for k,v in state.items():
        p = os.path.join(path, f"{k}.pt")
        try:
            torch.save(v, p)
        except Exception:
            with open(os.path.join(path, f"{k}.pkl"), "wb") as f:
                pickle.dump(v, f)
    # trainer_state if present
    if "trainer_state" in state:
        with open(os.path.join(path, "trainer_state.json"), "w") as f:
            json.dump(state["trainer_state"], f, indent=2)

def load_all(path: str, device: str = "cpu"):
    assert os.path.exists(path)
    items = {}
    for fname in os.listdir(path):
        fp = os.path.join(path, fname)
        key, ext = os.path.splitext(fname)
        if ext == ".pt":
            items[key] = torch.load(fp, map_location=device)
        elif ext == ".pkl":
            import pickle
            with open(fp,"rb") as f:
                items[key] = pickle.load(f)
        elif ext == ".json":
            import json
            with open(fp,"r") as f:
                items[key] = json.load(f)
    return items
