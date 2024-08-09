import torch
import gymnasium as gym
import numpy as np
from agent import Agent
import time

def play(agent, env, n_episodes=5):
    for i_episode in range(1, n_episodes + 1):
        state, _ = env.reset()
        done = False
        total_reward = 0
        while not done:
            action = agent.act(state)
            state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            total_reward += reward
            env.render()
            time.sleep(0.02)
        print(f'Episode {i_episode}\tTotal Reward: {total_reward:.2f}')
    env.close()

if __name__ == "__main__":
    env = gym.make('LunarLander-v2', render_mode="human")
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    agent = Agent(state_size, action_size, device=device, seed=0)
    
    agent.DQN_policy.load_state_dict(torch.load('checkpoint.pth', map_location=device, weights_only=True))
    
    play(agent, env)