import numpy as np
import gymnasium as gym
from gymnasium import spaces


class CustomEnv(gym.Env):
    """Generic skeleton for custom Gymnasium environments."""

    metadata = {"render_modes": ["human"]}

    def __init__(self, max_steps=100):
        super().__init__()

        self.max_steps = max_steps
        self.step_count = 0

        # Continuous state, 4 dimensions
        self.observation_space = spaces.Box(
            low=np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32),
            high=np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float32),
            dtype=np.float32,
        )

        # Discrete actions: 0, 1, 2
        self.action_space = spaces.Discrete(3)

        self.state = None

    def _sample_state(self):
        return self.np_random.uniform(
            low=self.observation_space.low,
            high=self.observation_space.high,
        ).astype(np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.step_count = 0
        self.state = self._sample_state()
        return self.state, {}

    def step(self, action):
        assert self.action_space.contains(action), "Invalid action"

        reward = self._compute_reward(self.state, action)

        self.step_count += 1
        terminated = False
        truncated = self.step_count >= self.max_steps

        info = {"action": action, "step_count": self.step_count}

        self.state = self._sample_state()
        return self.state, reward, terminated, truncated, info

    def _compute_reward(self, state, action):
        # Override with real reward logic
        return 0.0

    def render(self):
        print(f"state = {self.state}")
