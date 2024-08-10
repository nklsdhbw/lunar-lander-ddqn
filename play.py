import torch
import gymnasium as gym
import numpy as np
from agent import Agent
import time
from typing import Any
import os

def play(agent: Agent, env: gym.Env, n_episodes: int = 5) -> None:
    for i_episode in range(1, n_episodes + 1):
        state, _ = env.reset()
        done: bool = False
        total_reward: float = 0.0
        while not done:
            action: int = agent.act(state)
            state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            total_reward += reward
            env.render()
            time.sleep(0.02)
        print(f'Episode {i_episode}\tTotal Reward: {total_reward:.2f}')
    env.close()

if __name__ == "__main__":
    if not os.path.exists('dqn_policy.pth'):
        raise FileNotFoundError("Model file not found. Please run 'python train.py' first as shown in the README.md file.")
        
    env: gym.Env = gym.make('LunarLander-v2', render_mode="human")
    state_size: int = env.observation_space.shape[0]
    action_size: int = env.action_space.n
    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    agent: Agent = Agent(state_size, action_size, device=device, seed=0)
    
    agent.DQN_policy.load_state_dict(torch.load('dqn_policy.pth', map_location=device, weights_only=True))
    
    play(agent, env)