"""Train and evaluate Stable-Baselines3 agents (A2C, PPO, DQN)."""

import numpy as np

from envs.edge_offloading import EdgeOffloadingEnv
from baselines.policies import (
    policy_random,
    policy_device_only,
    policy_edge_only,
    policy_greedy_latency,
)
from train import train_a2c, train_ppo, train_dqn
from utils import (
    evaluate_policy,
    evaluate_sb3_model,
    build_comparison_table,
    inspect_action_distribution,
)


def main():
    env_fn = lambda: EdgeOffloadingEnv()

    # Baselines
    print("=" * 60)
    print("Evaluating baselines ...")
    print("=" * 60)

    baseline_results = {}
    for name, policy_fn in [
        ("Random", policy_random),
        ("Device-only", policy_device_only),
        ("Edge-only", policy_edge_only),
        ("Greedy latency", policy_greedy_latency),
    ]:
        baseline_results[name] = evaluate_policy(env_fn, policy_fn, episodes=100)
        res = baseline_results[name]
        print(
            f"{name:15} | "
            f"Avg Reward: {res['avg_reward']:.3f} | "
            f"Avg Latency: {res.get('avg_latency', float('nan')):.2f}ms | "
            f"P95: {res.get('p95_latency', float('nan')):.2f}ms"
        )

    # A2C
    print("\n" + "=" * 60)
    print("Training A2C ...")
    print("=" * 60)
    a2c_model = train_a2c(total_timesteps=100_000)

    # PPO
    print("\n" + "=" * 60)
    print("Training PPO ...")
    print("=" * 60)
    ppo_model = train_ppo(total_timesteps=200_000)

    # DQN
    print("\n" + "=" * 60)
    print("Training DQN ...")
    print("=" * 60)
    dqn_model = train_dqn(total_timesteps=200_000)

    # Evaluate RL agents
    print("\n" + "=" * 60)
    print("Evaluating RL agents ...")
    print("=" * 60)

    rl_results = {}
    for name, model in [("A2C", a2c_model), ("PPO", ppo_model), ("DQN", dqn_model)]:
        res = evaluate_sb3_model(model, env_fn, episodes=100)
        rl_results[name] = res
        print(f"{name} | Avg Reward: {res['avg_reward']:.3f} ± {res['std_reward']:.3f}")
        inspect_action_distribution(model, EdgeOffloadingEnv(), episodes=20)

    # Comparison
    print("\n" + "=" * 60)
    print("Final Comparison")
    print("=" * 60)

    all_results = {**baseline_results}
    # Merge RL results into the same format
    for name, res in rl_results.items():
        all_results[name] = {
            "avg_reward": res["avg_reward"],
            "action_counts": res["action_counts"],
        }

    greedy_latency = baseline_results["Greedy latency"]["avg_latency"]
    df = build_comparison_table(all_results, greedy_latency=greedy_latency)
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
