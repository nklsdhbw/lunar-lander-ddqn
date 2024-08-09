import random
import torch
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from ReplayMemory import ReplayMemory
from dqn import DQN
from typing import Tuple

class Agent:
    def __init__(
        self,
        state_size: int,
        action_size: int,
        seed: int,
        device: torch.device,
        gamma: float = 0.99,
        tau: float = 1e-3,
        batch_size: int = 64,
        update_every: int = 4
    ) -> None:
        self.state_size: int = state_size
        self.action_size: int = action_size
        self.seed: None = random.seed(seed)
        self.device: torch.device = device

        self.gamma: float = gamma
        self.tau: float = tau
        self.batch_size: int = batch_size
        self.update_every: int = update_every

        self.DQN_policy: DQN = DQN(state_size, action_size, seed).to(device)
        self.DQN_target: DQN = DQN(state_size, action_size, seed).to(device)
        self.optimizer: optim.Adam = optim.Adam(self.DQN_policy.parameters(), lr=5e-4)

        self.memory: ReplayMemory = ReplayMemory(action_size, buffer_size=int(1e5), batch_size=batch_size, seed=seed)
        self.t_step: int = 0

    def step(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool
    ) -> None:
        self.memory.add(state, action, reward, next_state, done)
        self.t_step = (self.t_step + 1) % self.update_every
        if self.t_step == 0:
            if len(self.memory) > self.batch_size:
                experiences: Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor] = self.memory.sample(device=self.device)
                self.learn(experiences)

    def act(self, state: np.ndarray, eps: float = 0.0) -> int:
        state = np.array(state, dtype=np.float32)  # Ensure state is float32
        if state.ndim == 1:
            state = np.expand_dims(state, axis=0)
        state = torch.from_numpy(state).to(self.device)
        self.DQN_policy.eval()
        with torch.no_grad():
            action_values = self.DQN_policy(state)
        self.DQN_policy.train()

        if random.random() > eps:
            return np.argmax(action_values.cpu().data.numpy())
        else:
            return random.choice(np.arange(self.action_size))

    def learn(self, experiences: Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]) -> None:
        states, actions, rewards, next_states, dones = experiences
        Q_targets_next = self.DQN_target(next_states).detach().max(1)[0].unsqueeze(1)
        Q_targets = rewards + (self.gamma * Q_targets_next * (1 - dones))
        Q_expected = self.DQN_policy(states).gather(1, actions)

        loss = F.mse_loss(Q_expected, Q_targets)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.soft_update(self.DQN_policy, self.DQN_target)

    def soft_update(self, local_model: nn.Module, target_model: nn.Module) -> None:
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(self.tau * local_param.data + (1.0 - self.tau) * target_param.data)