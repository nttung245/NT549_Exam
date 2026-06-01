"""Run the full Edge Offloading experiment: baselines + RL agents."""

import numpy as np

from envs.edge_offloading import EdgeOffloadingEnv
from baselines.policies import (
    policy_random,
    policy_device_only,
    policy_edge_only,
    policy_greedy_latency,
)
from agents.reinforce import train_reinforce
from agents.actor_critic import train_actor_critic
from utils import (
    evaluate_policy,
    build_comparison_table,
    plot_curve,
    normalize_state,
)

import torch


def main():
    env_fn = lambda: EdgeOffloadingEnv()

    # -----------------------------------------------------------------------
    # 1. Baselines
    # -----------------------------------------------------------------------
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

    # -----------------------------------------------------------------------
    # 2. REINFORCE
    # -----------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("Training REINFORCE ...")
    print("=" * 60)

    reinforce_model, reinforce_rewards = train_reinforce(
        EdgeOffloadingEnv(), episodes=500, lr=1e-3, gamma=0.99
    )
    plot_curve(reinforce_rewards, title="REINFORCE Learning Curve")

    def reinforce_action_fn(state, env):
        from utils import normalize_state as ns
        state_n = ns(state, env)
        state_t = torch.tensor(state_n, dtype=torch.float32)
        with torch.no_grad():
            logits = reinforce_model(state_t)
        return int(torch.argmax(logits).item())

    reinforce_eval = evaluate_policy(env_fn, reinforce_action_fn, episodes=100)
    print(f"REINFORCE | Avg Reward: {reinforce_eval['avg_reward']:.3f}")

    # -----------------------------------------------------------------------
    # 3. Actor-Critic
    # -----------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("Training Actor-Critic ...")
    print("=" * 60)

    ac_model, ac_rewards = train_actor_critic(
        EdgeOffloadingEnv(), episodes=500, lr=1e-3, gamma=0.99, entropy_coef=0.01
    )
    plot_curve(ac_rewards, title="Actor-Critic Learning Curve")

    def ac_action_fn(state, env):
        from utils import normalize_state as ns
        state_n = ns(state, env)
        state_t = torch.tensor(state_n, dtype=torch.float32)
        with torch.no_grad():
            logits, _ = ac_model(state_t)
        return int(torch.argmax(logits).item())

    ac_eval = evaluate_policy(env_fn, ac_action_fn, episodes=100)
    print(f"Actor-Critic | Avg Reward: {ac_eval['avg_reward']:.3f}")

    # -----------------------------------------------------------------------
    # 4. Comparison table
    # -----------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("Comparison")
    print("=" * 60)

    all_results = {**baseline_results}
    all_results["REINFORCE"] = reinforce_eval
    all_results["Actor-Critic"] = ac_eval

    greedy_latency = baseline_results["Greedy latency"]["avg_latency"]
    df = build_comparison_table(all_results, greedy_latency=greedy_latency)
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
