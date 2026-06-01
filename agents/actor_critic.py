"""Actor-Critic algorithm from scratch."""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical

from utils import normalize_state


class ActorCriticNetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
        )
        self.actor = nn.Linear(128, action_dim)
        self.critic = nn.Linear(128, 1)

    def forward(self, x):
        features = self.shared(x)
        logits = self.actor(features)
        value = self.critic(features)
        return logits, value


def train_actor_critic(
    env,
    episodes=1000,
    lr=1e-3,
    gamma=0.99,
    entropy_coef=0.01,
    seed=42,
    verbose=True,
):
    """Train an Actor-Critic agent."""
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    model = ActorCriticNetwork(state_dim, action_dim)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    reward_history = []

    for ep in range(episodes):
        state, _ = env.reset(seed=seed + ep)
        done = False
        ep_return = 0.0

        while not done:
            state_n = normalize_state(state, env)
            state_t = torch.tensor(state_n, dtype=torch.float32)

            logits, value = model(state_t)
            dist = Categorical(logits=logits)
            action = dist.sample()

            next_state, reward, terminated, truncated, info = env.step(action.item())
            done = terminated or truncated

            with torch.no_grad():
                next_state_n = normalize_state(next_state, env)
                next_state_t = torch.tensor(next_state_n, dtype=torch.float32)
                _, next_value = model(next_state_t)
                reward_t = torch.tensor([reward], dtype=torch.float32)
                td_target = reward_t if done else reward_t + gamma * next_value.view(-1)

            advantage = td_target - value.view(-1)

            actor_loss = -dist.log_prob(action) * advantage.detach()
            critic_loss = F.mse_loss(value.view(-1), td_target)
            entropy = dist.entropy().mean()

            loss = actor_loss + 0.5 * critic_loss - entropy_coef * entropy

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            state = next_state
            ep_return += reward

        reward_history.append(ep_return)

        if verbose and (ep + 1) % 50 == 0:
            print(f"Actor-Critic | Episode {ep+1} | Return = {ep_return:.3f}")

    return model, reward_history
