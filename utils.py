"""Shared utilities: normalization, evaluation, plotting."""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# State normalization
# ---------------------------------------------------------------------------

def normalize_state(state, env):
    """Min-max normalize state to [0, 1] using env bounds."""
    low = env.observation_space.low
    high = env.observation_space.high
    return (state - low) / (high - low + 1e-8)


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate_policy(env_fn, action_fn, episodes=100, seed=42):
    """Evaluate a policy over multiple episodes.

    Args:
        env_fn: callable that returns a new environment instance.
        action_fn: callable (state, env) -> action.
        episodes: number of evaluation episodes.
        seed: base seed for reproducibility.

    Returns:
        dict with reward, latency, action distribution stats.
    """
    episode_rewards = []
    episode_avg_latencies = []
    all_latencies = []
    action_counts = None

    for ep in range(episodes):
        env = env_fn()
        state, _ = env.reset(seed=seed + 2000 + ep)

        if action_counts is None:
            action_counts = np.zeros(env.action_space.n, dtype=np.int64)

        done = False
        rewards = []
        latencies = []

        while not done:
            action = int(action_fn(state, env))
            state, reward, terminated, truncated, info = env.step(action)

            rewards.append(reward)
            if "latency" in info:
                latencies.append(info["latency"])
            if action < len(action_counts):
                action_counts[action] += 1

            done = terminated or truncated

        env.close()
        episode_rewards.append(float(np.sum(rewards)))
        if latencies:
            episode_avg_latencies.append(float(np.mean(latencies)))
            all_latencies.extend(latencies)

    result = {
        "episode_rewards": np.array(episode_rewards, dtype=np.float32),
        "avg_reward": float(np.mean(episode_rewards)),
        "action_counts": action_counts,
    }

    if all_latencies:
        all_latencies = np.array(all_latencies, dtype=np.float32)
        total_actions = max(1, int(action_counts.sum()))
        result.update(
            {
                "all_latencies": all_latencies,
                "avg_latency": float(np.mean(all_latencies)),
                "p95_latency": float(np.percentile(all_latencies, 95)),
                "device_pct": float(action_counts[0] / total_actions * 100),
                "edge_pct": float(action_counts[1] / total_actions * 100),
            }
        )

    return result


def evaluate_sb3_model(model, env_fn, episodes=100, deterministic=True, seed=3000):
    """Evaluate a Stable-Baselines3 model."""
    rewards = []
    action_counts = None

    for ep in range(episodes):
        env = env_fn()
        state, _ = env.reset(seed=seed + ep)

        if action_counts is None:
            action_counts = np.zeros(env.action_space.n, dtype=np.int64)

        done = False
        ep_return = 0.0

        while not done:
            action, _ = model.predict(state, deterministic=deterministic)
            action = int(np.asarray(action).item())
            state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            ep_return += reward
            action_counts[action] += 1

        rewards.append(ep_return)
        env.close()

    return {
        "avg_reward": float(np.mean(rewards)),
        "std_reward": float(np.std(rewards)),
        "action_counts": action_counts,
    }


# ---------------------------------------------------------------------------
# Debug helpers
# ---------------------------------------------------------------------------

def inspect_action_distribution(model, env, episodes=20):
    """Print action distribution for a trained SB3 model."""
    counts = np.zeros(env.action_space.n, dtype=np.int64)
    for ep in range(episodes):
        state, _ = env.reset(seed=1000 + ep)
        done = False
        while not done:
            action, _ = model.predict(state, deterministic=False)
            action = int(np.asarray(action).item())
            counts[action] += 1
            state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

    pct = counts / counts.sum() * 100
    print(f"Action counts: {counts}")
    print(f"Action pct:    {pct}")
    return counts, pct


def compare_agent_to_greedy(model, env, policy_greedy, n_samples=1000):
    """Compare agent actions against greedy baseline."""
    mismatch = 0
    rows = []

    for i in range(n_samples):
        state, _ = env.reset(seed=5000 + i)
        agent_action, _ = model.predict(state, deterministic=True)
        agent_action = int(np.asarray(agent_action).item())
        greedy_action = policy_greedy(state, env)

        if agent_action != greedy_action:
            mismatch += 1
            t_device, t_edge = env._compute_latencies(state)
            rows.append(
                {
                    "state": state,
                    "agent_action": agent_action,
                    "greedy_action": greedy_action,
                    "t_device": t_device,
                    "t_edge": t_edge,
                    "gap": abs(t_device - t_edge),
                }
            )

    print(f"Mismatch rate: {mismatch / n_samples:.3f}")
    return rows


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def moving_average(values, window=20):
    return pd.Series(values).rolling(window=window, min_periods=1).mean().to_numpy()


def plot_curve(values, title="Learning Curve", ylabel="Episode Reward", window=20):
    x = np.arange(1, len(values) + 1)
    plt.figure(figsize=(10, 4))
    plt.plot(x, values, alpha=0.35, label="Raw")
    plt.plot(x, moving_average(values, window), linewidth=2, label="Moving average")
    plt.xlabel("Episode")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.show()


# ---------------------------------------------------------------------------
# Comparison table
# ---------------------------------------------------------------------------

def build_comparison_table(results_dict, greedy_latency=None):
    """Build a pandas DataFrame comparing methods.

    Args:
        results_dict: {name: evaluate_policy result dict}
        greedy_latency: avg latency of greedy baseline for gap calculation.
    """
    rows = []
    for name, result in results_dict.items():
        row = {
            "Method": name,
            "Avg Reward ↑": result["avg_reward"],
            "Device %": result.get("device_pct", None),
            "Edge %": result.get("edge_pct", None),
        }
        if "avg_latency" in result:
            row["Avg Latency ↓"] = result["avg_latency"]
        if "p95_latency" in result:
            row["P95 Latency ↓"] = result["p95_latency"]
        if greedy_latency is not None and "avg_latency" in result:
            row["Gap to Greedy ↓"] = result["avg_latency"] - greedy_latency
        rows.append(row)

    df = pd.DataFrame(rows)
    if "Avg Latency ↓" in df.columns:
        df = df.sort_values(by="Avg Latency ↓")
    return df
