import torch
from collections import namedtuple, deque
import random
import numpy as np
from typing import Tuple, Deque, NamedTuple

class ReplayMemory:
    def __init__(self, action_size: int, buffer_size: int, batch_size: int, seed: int) -> None:
        self.action_size: int = action_size
        self.memory: Deque[NamedTuple] = deque(maxlen=buffer_size)
        self.batch_size: int = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
        self.seed: None = random.seed(seed)

    def add(self, state: np.ndarray, action: int, reward: float, next_state: np.ndarray, done: bool) -> None:
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)

    def sample(self, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        experiences = random.sample(self.memory, k=self.batch_size)
        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).long().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(device)
        return (states, actions, rewards, next_states, dones)

    def __len__(self) -> int:
        return len(self.memory)