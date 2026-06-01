"""Baseline policies for the Edge Offloading environment."""


def policy_random(state, env):
    """Select a random action."""
    return env.action_space.sample()


def policy_device_only(state, env):
    """Always process on device."""
    return 0


def policy_edge_only(state, env):
    """Always offload to edge."""
    return 1


def policy_greedy_latency(state, env):
    """Choose the action with lower instantaneous latency."""
    t_device, t_edge = env._compute_latencies(state)
    return 0 if t_device <= t_edge else 1
