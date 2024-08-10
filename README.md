# 🕹️ Deep Q-Learning on OpenAI Gym's LunarLander Environment

## Project Overview

This project implements a Double Deep Q-Network (Double DQN) to solve the LunarLander-v2 environment from OpenAI's Gymnasium. The project includes key components like a replay memory buffer, Double DQN agent, and the training process to optimize the agent's performance in landing the lunar module successfully.

## 📂 Directory Structure

```

project-root/

│
├── .gitgignore
│
├── agent.py
│
├── dqn.py
│
├── ReplayMemory.py
│
├── train.py
│
├── play.py
│
├── dqn_policy.pth
│
└── requirements.txt

```

## 📄 Files Description

1. **agent.py**:

- Contains the implementation of the Double DQN agent, including methods for action selection, network updates, and interaction with the environment.

2. **dqn.py**:

- Defines the neural network architecture used by the Double DQN agent, including layers and forward pass logic.

3. **ReplayMemory.py**:

- Implements the experience replay memory, which stores experiences (state, action, reward, next state) to be sampled during training.

4. **train.py**:

- Handles the training loop, where the Double DQN agent is trained over multiple episodes to learn the optimal policy for the LunarLander-v2 environment. The trained model is saved as `dqn_policy.pth`.

5. **play.py**:

- Loads the trained model from `dqn_policy.pth` and runs it to evaluate its performance in the LunarLander-v2 environment.

6. **dqn_policy.pth**:

- A saved checkpoint file containing the trained model's weights, allowing the agent to be loaded and evaluated without retraining.

7. **requirements.txt**:

- Lists the required Python packages for the project, including `gymnasium[box2d]` for the environment and `torchvision` for neural network operations.

---

## ⚙️ Installation

### 🐍 Install Python 3.12

Download and install [Python3.12](https://www.python.org/downloads/).

### ⬇️ Clone the repository to your local machine

```sh
git  clone  https://github.com/nklsdhbw/lunar-lander-ddqn.git
```

and change to the project directory.

### 🔨 Create a virtual environment using `venv`:

```sh
python3.12  -m  venv  .venv
```

### 🚀 Activate the virtual environment:

- On Windows:

```sh
.venv\Scripts\activate
```

- On Unix or MacOS:

```sh
source  .venv/bin/activate
```

### 📦 Install the required packages using pip:

```sh
pip  install  -r  requirements.txt
```

---

## Usage

1. ### Training the Double DQN Agent:

To train the Double DQN agent on the LunarLander-v2 environment, run the `train.py` script:

```sh
python  train.py
```

This will train the agent and save the best model weights in `dqn_policy.pth`.

You can also pass arguments to specify the hyperparameters as shown in the following example:

```python
python train.py --gamma 0.95
```

**Arguments for** **`train.py`:**

If the hyperparameters are not set, the default values will be used.

* `--gamma`: Specify the discount factor (default: 0.99).
* `--batch_size`: Specify the batch size (default: 64).
* `--update_every`: Specify how often to update the network (default: 4).
* `--tau`: Specify the soft update parameter (default: 1e-3).
* `--eps_decay`: Specify the epsilon decay rate (default: 0.995).
* `--n_episodes`: Specify the number of episodes the agent will be trained on (default: 2000)

**Note:** You'll also be warned about all hyperparameters that are not set

2. ### Optimize Hyperparameters of the Double DQN Agent

To perform a grid search for hyperparameter optimizitaiton, use the `--grid_search` flag as shown in the following example:

```python
python train.py --model nn --grid_search
```

**Note:** All other flags set when using the `--grid_search` flag will be ignored.

3. ### Running the Trained Agent:

To evaluate the trained Double DQN agent, run the `play.py` script:

```sh
python  play.py
```

This will load the model from `dqn_policy.pth` and run the agent in the LunarLander-v2 environment.

---

## Acknowledgments

This project utilizes the LunarLander-v2 environment from OpenAI's Gymnasium and leverages PyTorch for implementing the Double Deep Q-Network. The experience replay buffer is based on commonly used techniques in reinforcement learning to improve learning stability.
