import torch
import torch.nn as nn
from torch import Tensor

class DQN(nn.Module):
    def __init__(self, state_size: int, action_size: int, seed: int) -> None:
        super(DQN, self).__init__()
        self.seed: torch.Generator = torch.manual_seed(seed)
        self.fc1: nn.Linear = nn.Linear(state_size, 128)
        self.fc2: nn.Linear = nn.Linear(128, 128)
        self.fc3: nn.Linear = nn.Linear(128, action_size)

    def forward(self, state: Tensor) -> Tensor:
        x: Tensor = torch.relu(self.fc1(state))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)