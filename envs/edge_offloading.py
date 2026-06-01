import numpy as np
import gymnasium as gym
from gymnasium import spaces


class EdgeOffloadingEnv(gym.Env):
    """
    Edge Offloading Environment.

    State:  [input_size, workload, uplink_rate, edge_queue]
    Action: 0 = process on device, 1 = offload to edge
    Reward: -selected_latency / 1000
    """

    metadata = {"render_modes": ["human"]}

    def __init__(self, max_steps=50):
        super().__init__()

        self.f_device = 1.0  # GHz
        self.f_edge = 5.0  # GHz
        self.max_steps = max_steps
        self.step_count = 0

        self.action_space = spaces.Discrete(2)

        low = np.array([1.0, 0.1, 2.0, 0.0], dtype=np.float32)
        high = np.array([20.0, 3.0, 20.0, 200.0], dtype=np.float32)
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)

        self.state = None

    def _sample_state(self):
        input_size = self.np_random.uniform(1.0, 20.0)
        workload = self.np_random.uniform(0.1, 3.0)
        uplink_rate = self.np_random.uniform(2.0, 20.0)
        edge_queue = self.np_random.uniform(0.0, 200.0)
        return np.array([input_size, workload, uplink_rate, edge_queue], dtype=np.float32)

    def _compute_latencies(self, state):
        input_size, workload, uplink_rate, edge_queue = state

        t_device = (workload / self.f_device) * 1000
        t_upload = (input_size / uplink_rate) * 1000
        t_edge = t_upload + edge_queue + (workload / self.f_edge) * 1000

        return t_device, t_edge

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.step_count = 0
        self.state = self._sample_state()
        return self.state, {}

    def step(self, action):
        assert self.action_space.contains(action), "Invalid action"

        t_device, t_edge = self._compute_latencies(self.state)

        if action == 0:
            latency = t_device
            action_name = "device"
        else:
            latency = t_edge
            action_name = "edge"

        reward = -latency / 1000.0

        self.step_count += 1
        terminated = False
        truncated = self.step_count >= self.max_steps

        info = {
            "latency": latency,
            "latency_device": t_device,
            "latency_edge": t_edge,
            "action": action,
            "action_name": action_name,
            "is_greedy_action": action == (0 if t_device <= t_edge else 1),
        }

        self.state = self._sample_state()
        return self.state, reward, terminated, truncated, info

    def render(self):
        t_device, t_edge = self._compute_latencies(self.state)
        print(
            f"state={self.state} | "
            f"t_device={t_device:.1f}ms | t_edge={t_edge:.1f}ms"
        )
