"""REINFORCE policy gradient algorithm from scratch."""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical

from utils import normalize_state


class PolicyNetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim),
        )

    def forward(self, x):
        return self.net(x)


def compute_returns(rewards, gamma=0.99):
    """Compute discounted returns and normalize."""
    returns = []
    G = 0.0
    for r in reversed(rewards):
        G = r + gamma * G
        returns.insert(0, G)

    returns = torch.tensor(returns, dtype=torch.float32)
    returns = (returns - returns.mean()) / (returns.std() + 1e-8)
    return returns


def train_reinforce(env, episodes=1000, lr=1e-3, gamma=0.99, seed=42, verbose=True):
    """Train a REINFORCE agent."""
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    model = PolicyNetwork(state_dim, action_dim)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    reward_history = []

    for ep in range(episodes):
        state, _ = env.reset(seed=seed + ep)
        done = False

        log_probs = []
        rewards = []

        while not done:
            state_n = normalize_state(state, env)
            state_t = torch.tensor(state_n, dtype=torch.float32)

            logits = model(state_t)
            dist = Categorical(logits=logits)
            action = dist.sample()
            log_prob = dist.log_prob(action)

            next_state, reward, terminated, truncated, info = env.step(action.item())
            done = terminated or truncated

            log_probs.append(log_prob)
            rewards.append(reward)
            state = next_state

        returns = compute_returns(rewards, gamma)

        loss = 0.0
        for log_prob, G in zip(log_probs, returns):
            loss += -log_prob * G

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        ep_return = sum(rewards)
        reward_history.append(ep_return)

        if verbose and (ep + 1) % 50 == 0:
            print(f"REINFORCE | Episode {ep+1} | Return = {ep_return:.3f}")

    return model, reward_history
