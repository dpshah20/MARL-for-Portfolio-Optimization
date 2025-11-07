# rl_layer/replay_buffer.py
import random, pickle, os
import numpy as np

class ReplayBuffer:
    def __init__(self, capacity: int = 200_000):
        self.capacity = capacity
        self.buffer = []
        self.pos = 0

    def add(self, transition):
        if len(self.buffer) < self.capacity:
            self.buffer.append(transition)
        else:
            self.buffer[self.pos] = transition
        self.pos = (self.pos + 1) % self.capacity

    def sample(self, batch_size: int):
        batch = random.sample(self.buffer, min(batch_size, len(self.buffer)))
        return batch

    def __len__(self):
        return len(self.buffer)

    def save(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump({"capacity": self.capacity, "buffer": self.buffer, "pos": self.pos}, f)

    def load(self, path):
        with open(path, "rb") as f:
            data = pickle.load(f)
        self.capacity = data["capacity"]
        self.buffer = data["buffer"]
        self.pos = data["pos"]
