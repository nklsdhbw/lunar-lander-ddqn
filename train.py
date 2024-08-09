import gymnasium as gym
import numpy as np
from collections import deque
from agent import Agent
import torch
from sklearn.model_selection import ParameterGrid
import argparse
from typing import List, Tuple, Dict, Any

env = gym.make("LunarLander-v2")
state_size: int = env.observation_space.shape[0]
action_size: int = env.action_space.n
device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
agent: Agent = Agent(state_size, action_size, device=device, seed=0, gamma=0.99, tau=1e-3, batch_size=64, update_every=4)

def dqn(
    n_episodes: int = 2000, 
    max_t: int = 1000, 
    eps_start: float = 1.0, 
    eps_end: float = 0.01, 
    eps_decay: float = 0.995
) -> Tuple[List[float], List[Dict[str, Any]], int]:
    scores: List[float] = []
    params: List[Dict[str, Any]] = []
    scores_window: deque = deque(maxlen=100)
    eps: float = eps_start
    for i_episode in range(1, n_episodes + 1):
        state, _ = env.reset()
        score: float = 0.0
        for t in range(max_t):
            action: int = agent.act(state, eps)
            next_state, reward, terminated, truncated, _ = env.step(action) 
            done: bool = terminated or truncated 
            agent.step(state, action, reward, next_state, done)
            state = next_state
            score += reward
            if done:
                break
        scores_window.append(score)
        scores.append(score)
        eps = max(eps_end, eps_decay * eps)
        print(f"\rEpisode {i_episode}\tAverage Score: {np.mean(scores_window):.2f}", end="")
        if i_episode % 100 == 0:
            print(f"\rEpisode {i_episode}\tAverage Score: {np.mean(scores_window):.2f}")
        if np.mean(scores_window) >= 250.0:
            print(f"\nEnvironment solved in {i_episode - 100} episodes!\tAverage Score: {np.mean(scores_window):.2f}")
            torch.save(agent.DQN_policy.state_dict(), "checkpoint.pth")
            break
    if np.mean(scores_window) < 250.0:    
        print(f"\nEnvironment not solved in {i_episode} episodes.\tAverage Score: {np.mean(scores_window):.2f}")
        print("Model will not be saved.")
    return scores, params, i_episode

if __name__ == "__main__":
    parser: argparse.ArgumentParser = argparse.ArgumentParser(description="Run the ddqn for LunarLander.")

    group = parser.add_mutually_exclusive_group(required=False)
    group.add_argument("--grid_search", action="store_true", help="Specify whether to perform grid search.")

    parser.add_argument("--gamma", type=float, help="Discount factor.")
    parser.add_argument("--batch_size", type=int, help="Batch size.")
    parser.add_argument("--update_every", type=int, help="Update frequency.")
    parser.add_argument("--tau", type=float, help="Soft update parameter.")
    parser.add_argument("--eps_decay", type=float, help="Epsilon decay rate.")

    args: argparse.Namespace = parser.parse_args()

    if args.grid_search:
        param_grid: Dict[str, List[Any]] = {
            "gamma": [0.999, 0.99, 0.95],
            "batch_size": [64, 128],
            "update_every": [4, 2],
            "tau": [1e-3, 1e-2],
            "eps_decay": [0.995, 0.99]
        }

        grid: ParameterGrid = ParameterGrid(param_grid)
        episodes: List[int] = []
        params_total: List[Dict[str, Any]] = []
        for params in list(grid)[:2]:
            agent = Agent(
                state_size=state_size,
                action_size=action_size,
                seed=0,
                device=device,
                gamma=params["gamma"],
                tau=params["tau"],
                batch_size=params["batch_size"],
                update_every=params["update_every"],
            )
            scores, params_used, episodes_taken = dqn(eps_decay=params["eps_decay"])
            episodes.append(episodes_taken)
            params_total.append(params_used)
        min_episode_idx: int = int(np.argmin(episodes))
        print(f"Best parameters: {params_total[min_episode_idx]}\tEpisodes taken: {episodes[min_episode_idx]}")

    else:
        yellow: str = "\033[93m"
        reset: str = "\033[0m"

        defaults: Dict[str, Any] = {
            "gamma": 0.99,
            "batch_size": 64,
            "update_every": 4,
            "tau": 1e-3,
            "eps_decay": 0.995
        }
        
        warnings: List[str] = []

        for param, default_value in defaults.items():
            if getattr(args, param) is None:
                warnings.append(f"{param} not set, using default value: {default_value}")

        if warnings:
            print(f"{yellow}WARNING: The following parameters were not set and will use default values:{reset}")
            for warning in warnings:
                print(f"{yellow}{warning}{reset}")

        agent = Agent(
                state_size=state_size,
                action_size=action_size,
                seed=0,
                device=device,
                gamma=args.gamma if args.gamma is not None else defaults["gamma"],
                tau=args.tau if args.tau is not None else defaults["tau"],
                batch_size=args.batch_size if args.batch_size is not None else defaults["batch_size"],
                update_every=args.update_every if args.update_every is not None else defaults["update_every"],
            )
        scores, params_used, episodes_taken = dqn(eps_decay=args.eps_decay if args.eps_decay is not None else defaults["eps_decay"])