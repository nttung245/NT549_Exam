# RL A-Z Exam Flow + Code Mẫu

> Mục tiêu: đây là bản ghi chú offline để tự định nghĩa, triển khai và phân tích một bài toán Reinforcement Learning (RL) từ đầu đến cuối, đặc biệt cho các bài dạng **custom environment**, **thiết kế reward**, **so sánh baseline**, **chọn thuật toán**, **debug agent bị bias/collapse**, và **chỉnh hyperparameter**.

---

## 0. Flow tổng quát khi gặp một bài RL

Khi đọc đề, đi theo thứ tự sau:

```text
1. Hiểu bài toán thực tế
   ↓
2. Xác định agent là ai
   ↓
3. Xác định environment là gì
   ↓
4. Xác định state / observation
   ↓
5. Xác định action space
   ↓
6. Xác định transition dynamics
   ↓
7. Xác định reward function
   ↓
8. Xác định episode: reset, done, truncated
   ↓
9. Xây baseline chính
   ↓
10. Chọn thuật toán RL
   ↓
11. Train + log metrics
   ↓
12. Evaluate deterministic / stochastic
   ↓
13. Debug nếu agent bias hoặc reward không tăng
   ↓
14. So sánh và kết luận
```

Cách trả lời miệng:

> Em không bắt đầu bằng thuật toán ngay. Em định nghĩa MDP trước: state là gì, action là gì, reward tối ưu mục tiêu nào, episode kết thúc khi nào. Sau đó em tạo baseline để biết mức hiệu năng tối thiểu, rồi mới chọn thuật toán phù hợp với action space và độ phức tạp của môi trường.

---

# PHẦN A — DEFINE BÀI TOÁN RL

## 1. Công thức MDP chuẩn

Một bài toán RL thường được mô hình hóa bằng MDP:

```text
MDP = (S, A, P, R, gamma)
```

Trong đó:

| Thành phần | Ý nghĩa | Câu hỏi cần trả lời |
|---|---|---|
| S | State space | Agent quan sát được gì? |
| A | Action space | Agent được phép làm gì? |
| P | Transition | Sau action thì môi trường thay đổi thế nào? |
| R | Reward | Hành động đó tốt hay xấu? |
| gamma | Discount factor | Có coi trọng tương lai không? |

Cách nói trong bài thi:

> Với bài toán này, em mô hình hóa dưới dạng MDP. State mô tả trạng thái hiện tại của hệ thống, action là quyết định của agent, reward phản ánh mục tiêu tối ưu, còn transition được mô phỏng bởi môi trường sau mỗi bước.

---

## 2. Template define problem

```text
Problem:
- Real-world goal:
- Agent:
- Environment:
- State / Observation:
- Action:
- Reward:
- Episode start:
- Episode end:
- Main metric:
- Baselines:
- Candidate algorithms:
```

Ví dụ Edge Offloading:

```text
Problem:
- Real-world goal: Giảm độ trễ xử lý task.
- Agent: Bộ ra quyết định offloading.
- Environment: Hệ thống device-edge.
- State: [input_size, workload, uplink_rate, edge_queue].
- Action: 0 = xử lý trên device, 1 = offload lên edge.
- Reward: -latency / 1000.
- Episode start: sample task đầu tiên.
- Episode end: sau max_steps.
- Main metric: avg latency, p95 latency, avg reward.
- Baselines: random, device-only, edge-only, greedy latency.
- Candidate algorithms: REINFORCE, Actor-Critic, A2C, PPO, DQN.
```

---

# PHẦN B — CODE MẪU CUSTOM ENVIRONMENT

## 3. Skeleton Gymnasium Environment cơ bản

Dùng khi đề yêu cầu tự tạo môi trường.

```python
import numpy as np
import gymnasium as gym
from gymnasium import spaces

class CustomEnv(gym.Env):
    metadata = {"render_modes": ["human"]}

    def __init__(self, max_steps=100):
        super().__init__()

        self.max_steps = max_steps
        self.step_count = 0

        # Ví dụ state liên tục 4 chiều
        self.observation_space = spaces.Box(
            low=np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32),
            high=np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float32),
            dtype=np.float32
        )

        # Ví dụ action rời rạc: 0, 1, 2
        self.action_space = spaces.Discrete(3)

        self.state = None

    def _sample_state(self):
        return self.np_random.uniform(
            low=self.observation_space.low,
            high=self.observation_space.high
        ).astype(np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.step_count = 0
        self.state = self._sample_state()
        info = {}
        return self.state, info

    def step(self, action):
        assert self.action_space.contains(action), "Invalid action"

        reward = self._compute_reward(self.state, action)

        self.step_count += 1
        terminated = False
        truncated = self.step_count >= self.max_steps

        info = {
            "action": action,
            "step_count": self.step_count
        }

        self.state = self._sample_state()
        return self.state, reward, terminated, truncated, info

    def _compute_reward(self, state, action):
        # Viết reward thật ở đây
        return 0.0

    def render(self):
        print("state =", self.state)
```

Ghi nhớ:

```text
terminated = kết thúc tự nhiên do thành công/thất bại.
truncated  = kết thúc do giới hạn thời gian/số step.
```

---

## 4. Code mẫu Edge Offloading Environment

Đây là dạng rất hay gặp vì state/action/reward rõ.

```python
import numpy as np
import gymnasium as gym
from gymnasium import spaces

class EdgeOffloadingEnv(gym.Env):
    """
    State:
        [input_size, workload, uplink_rate, edge_queue]

    Action:
        0 = xử lý tại device
        1 = offload lên edge

    Reward:
        -selected_latency / 1000
    """

    metadata = {"render_modes": ["human"]}

    def __init__(self, max_steps=50):
        super().__init__()

        self.f_device = 1.0  # GHz
        self.f_edge = 5.0    # GHz
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

        return np.array(
            [input_size, workload, uplink_rate, edge_queue],
            dtype=np.float32
        )

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
            "action_name": action_name
        }

        self.state = self._sample_state()
        return self.state, reward, terminated, truncated, info
```

Cách giải thích:

```text
Device latency chỉ gồm thời gian xử lý local.
Edge latency gồm upload delay + queue delay + edge compute delay.
Reward = -latency/1000 để biến bài toán minimize latency thành maximize reward.
Chia 1000 để scale reward nhỏ hơn, giúp training ổn định hơn.
```

---

## 5. Code kiểm tra environment

```python
from gymnasium.utils.env_checker import check_env

env = EdgeOffloadingEnv()
check_env(env, skip_render_check=True)
print("Environment passed check_env")
```

Test rollout:

```python
env = EdgeOffloadingEnv()
obs, info = env.reset(seed=42)

for t in range(5):
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)

    print(
        f"step={t}, action={action}, reward={reward:.4f}, "
        f"latency={info['latency']:.2f}ms, action_name={info['action_name']}"
    )

    if terminated or truncated:
        break
```

Nếu bị lỗi thường là:

```text
- observation trả về sai dtype, nên dùng np.float32.
- reward không phải float.
- action không thuộc action_space.
- reset không trả về (obs, info).
- step không trả về đủ 5 giá trị: obs, reward, terminated, truncated, info.
```

---

# PHẦN C — BASELINE POLICIES

## 6. Vì sao phải có baseline?

Câu trả lời:

> Baseline giúp biết thuật toán RL có thật sự học được gì không. Nếu RL chỉ bằng random hoặc chỉ bằng device-only thì chưa đủ thuyết phục. Greedy thường là baseline mạnh vì nó dùng công thức trực tiếp để chọn action tốt nhất tại thời điểm hiện tại.

---

## 7. Code baseline policies

```python
def policy_random(state, env):
    return env.action_space.sample()

def policy_device_only(state, env):
    return 0

def policy_edge_only(state, env):
    return 1

def policy_greedy_latency(state, env):
    t_device, t_edge = env._compute_latencies(state)
    if t_device <= t_edge:
        return 0
    return 1
```

---

## 8. Code evaluation function dùng chung

```python
import numpy as np

SEED = 42

def evaluate_policy(env_fn, action_fn, episodes=100):
    episode_rewards = []
    episode_avg_latencies = []
    all_latencies = []
    action_counts = np.zeros(2, dtype=np.int64)

    for ep in range(episodes):
        env = env_fn()
        state, _ = env.reset(seed=SEED + 2000 + ep)

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

        if len(latencies) > 0:
            episode_avg_latencies.append(float(np.mean(latencies)))
            all_latencies.extend(latencies)

    result = {
        "episode_rewards": np.array(episode_rewards, dtype=np.float32),
        "avg_reward": float(np.mean(episode_rewards)),
        "action_counts": action_counts,
    }

    if len(all_latencies) > 0:
        all_latencies = np.array(all_latencies, dtype=np.float32)
        total_actions = max(1, int(action_counts.sum()))

        result.update({
            "all_latencies": all_latencies,
            "avg_latency": float(np.mean(all_latencies)),
            "p95_latency": float(np.percentile(all_latencies, 95)),
            "device_pct": float(action_counts[0] / total_actions * 100),
            "edge_pct": float(action_counts[1] / total_actions * 100),
        })

    return result
```

Chạy baseline:

```python
baseline_results = {}

baseline_results["Random"] = evaluate_policy(
    lambda: EdgeOffloadingEnv(),
    policy_random
)

baseline_results["Device-only"] = evaluate_policy(
    lambda: EdgeOffloadingEnv(),
    policy_device_only
)

baseline_results["Edge-only"] = evaluate_policy(
    lambda: EdgeOffloadingEnv(),
    policy_edge_only
)

baseline_results["Greedy latency"] = evaluate_policy(
    lambda: EdgeOffloadingEnv(),
    policy_greedy_latency
)

for name, res in baseline_results.items():
    print(
        f"{name:15} | "
        f"Avg Reward: {res['avg_reward']:.3f} | "
        f"Avg Latency: {res.get('avg_latency', float('nan')):.2f}ms | "
        f"P95: {res.get('p95_latency', float('nan')):.2f}ms"
    )
```

---

# PHẦN D — REWARD DESIGN

## 9. Nguyên tắc thiết kế reward

Reward tốt nên có các tính chất:

```text
1. Cùng hướng với mục tiêu thực tế.
2. Không quá lớn hoặc quá nhỏ.
3. Không tạo shortcut cho agent.
4. Có penalty cho hành vi nguy hiểm.
5. Có reward/penalty đủ sớm nếu episode dài.
6. Có thể giải thích bằng công thức.
```

---

## 10. Reward dạng minimize cost

Dùng cho latency, energy, distance, error.

```python
reward = -cost
```

Nếu cost quá lớn:

```python
reward = -cost / scale
```

Ví dụ:

```python
reward = -latency_ms / 1000.0
```

Câu trả lời nếu bị hỏi tại sao chia 1000:

```text
Latency đang tính bằng ms nên giá trị có thể lên đến hàng nghìn.
Nếu dùng trực tiếp reward = -latency thì gradient có thể lớn và khó ổn định.
Chia 1000 đưa reward về khoảng nhỏ hơn, ví dụ từ -3000 thành -3.
Việc này không đổi mục tiêu tối ưu, vì action nào có latency thấp hơn vẫn có reward cao hơn.
```

---

## 11. Reward dạng nhiều thành phần

Dùng khi bài có nhiều mục tiêu.

```python
reward = (
    w_main * r_main
    + w_safety * r_safety
    + w_energy * r_energy
    + w_progress * r_progress
    + terminal_bonus
    - terminal_penalty
)
```

Ví dụ robotics/navigation:

```python
distance_to_goal = np.linalg.norm(agent_pos - goal_pos)
collision = check_collision()
arrived = distance_to_goal < 0.2

r_progress = previous_distance - distance_to_goal
r_safety = -1.0 if collision else 0.0
r_arrival = 10.0 if arrived else 0.0
r_time = -0.01

reward = 2.0 * r_progress + r_safety + r_arrival + r_time
```

---

## 12. Reward clipping

Khi reward quá lớn:

```python
reward = np.clip(reward, -10.0, 10.0)
```

Dùng khi:

```text
- Reward có outlier.
- Training loss dao động mạnh.
- Value function khó học.
```

Không nên lạm dụng vì có thể làm mất thông tin độ lớn reward.

---

## 13. Reward normalization

```python
running_mean = 0.99 * running_mean + 0.01 * reward
running_var = 0.99 * running_var + 0.01 * (reward - running_mean) ** 2
reward_norm = (reward - running_mean) / (np.sqrt(running_var) + 1e-8)
```

Trong thực tế với Stable-Baselines3 có thể dùng `VecNormalize`.

---

## 14. Reward shaping đúng và sai

Reward shaping đúng:

```text
- Khuyến khích tiến gần mục tiêu.
- Phạt va chạm/nguy hiểm.
- Phạt tiêu tốn tài nguyên.
- Không làm thay đổi mục tiêu cuối.
```

Reward shaping sai:

```text
- Thưởng quá nhiều cho hành vi phụ.
- Agent farm reward mà không hoàn thành nhiệm vụ.
- Penalty quá lớn khiến agent không dám làm gì.
- Reward sparse quá khiến agent không học được.
```

Ví dụ lỗi:

```python
# Sai: agent có thể đứng gần goal để farm progress reward nếu không có done đúng.
reward = +1.0 if near_goal else 0.0
```

Sửa:

```python
if arrived:
    reward += 10.0
    terminated = True
```

---

# PHẦN E — CHỌN THUẬT TOÁN

## 15. Decision tree chọn thuật toán

```text
Action rời rạc nhỏ?
    → DQN, REINFORCE, Actor-Critic, A2C, PPO

Action rời rạc nhưng môi trường phức tạp?
    → PPO hoặc DQN

Action liên tục?
    → PPO, SAC, TD3

Cần ổn định, dễ dùng?
    → PPO

Cần value-based, discrete action?
    → DQN

Muốn học policy trực tiếp?
    → REINFORCE / Actor-Critic / A2C / PPO

Môi trường có partial observability?
    → Recurrent PPO, LSTM/GRU policy, frame stacking, sliding window
```

---

## 16. Bảng chọn nhanh

| Thuật toán | Action | Ưu điểm | Nhược điểm | Khi dùng |
|---|---:|---|---|---|
| Q-learning | Discrete nhỏ | Dễ hiểu | Không hợp state lớn | Bài tabular |
| DQN | Discrete | Mạnh hơn Q-table | Cần replay buffer | Game, discrete control |
| REINFORCE | Discrete/continuous | Dễ viết policy gradient | High variance | Bài học thuật |
| Actor-Critic | Discrete/continuous | Ổn định hơn REINFORCE | Cần tune critic | Bài vừa |
| A2C | Discrete/continuous | Practical, synchronous | Có thể kém PPO | Lab policy-based |
| PPO | Discrete/continuous | Rất ổn định | Nhiều hyperparameter | Default tốt |
| SAC | Continuous | Exploration tốt | Phức tạp | Continuous control |

---

# PHẦN F — CODE MẪU REINFORCE

## 17. REINFORCE từ scratch

REINFORCE học trực tiếp policy bằng return của cả episode.

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical

class PolicyNetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim)
        )

    def forward(self, x):
        logits = self.net(x)
        return logits

def normalize_state(state, env):
    low = env.observation_space.low
    high = env.observation_space.high
    return (state - low) / (high - low + 1e-8)

def compute_returns(rewards, gamma=0.99):
    returns = []
    G = 0.0

    for r in reversed(rewards):
        G = r + gamma * G
        returns.insert(0, G)

    returns = torch.tensor(returns, dtype=torch.float32)

    # Giảm variance
    returns = (returns - returns.mean()) / (returns.std() + 1e-8)
    return returns

def train_reinforce(env, episodes=1000, lr=1e-3, gamma=0.99):
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    model = PolicyNetwork(state_dim, action_dim)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    reward_history = []

    for ep in range(episodes):
        state, _ = env.reset(seed=42 + ep)
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

        if (ep + 1) % 50 == 0:
            print(f"Episode {ep+1} | Return = {ep_return:.3f}")

    return model, reward_history
```

Evaluate REINFORCE:

```python
def reinforce_action_fn(state, env):
    state_n = normalize_state(state, env)
    state_t = torch.tensor(state_n, dtype=torch.float32)

    with torch.no_grad():
        logits = reinforce_model(state_t)

    return int(torch.argmax(logits).item())

reinforce_model, reinforce_rewards = train_reinforce(EdgeOffloadingEnv())
reinforce_eval = evaluate_policy(lambda: EdgeOffloadingEnv(), reinforce_action_fn)
print(reinforce_eval)
```

Câu nhận xét:

```text
REINFORCE dễ triển khai nhưng update dựa trên return toàn episode nên variance cao.
Nếu reward dao động mạnh hoặc episode dài, học chậm và không ổn định.
```

---

# PHẦN G — CODE MẪU ACTOR-CRITIC

## 18. Actor-Critic từ scratch

Actor chọn action, Critic ước lượng state value.

```python
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical

class ActorCriticNetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()

        self.shared = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU()
        )

        self.actor = nn.Linear(128, action_dim)
        self.critic = nn.Linear(128, 1)

    def forward(self, x):
        features = self.shared(x)
        logits = self.actor(features)
        value = self.critic(features)
        return logits, value

def train_actor_critic(env, episodes=1000, lr=1e-3, gamma=0.99, entropy_coef=0.01):
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    model = ActorCriticNetwork(state_dim, action_dim)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    reward_history = []

    for ep in range(episodes):
        state, _ = env.reset(seed=42 + ep)
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

                if done:
                    td_target = reward_t
                else:
                    td_target = reward_t + gamma * next_value.view(-1)

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

        if (ep + 1) % 50 == 0:
            print(f"Episode {ep+1} | Return = {ep_return:.3f}")

    return model, reward_history
```

Evaluate Actor-Critic:

```python
ac_model, ac_rewards = train_actor_critic(
    EdgeOffloadingEnv(),
    episodes=1000,
    lr=1e-3,
    gamma=0.99,
    entropy_coef=0.01
)

def ac_action_fn(state, env):
    state_n = normalize_state(state, env)
    state_t = torch.tensor(state_n, dtype=torch.float32)

    with torch.no_grad():
        logits, _ = ac_model(state_t)

    return int(torch.argmax(logits).item())

ac_eval = evaluate_policy(lambda: EdgeOffloadingEnv(), ac_action_fn)
print(ac_eval)
```

Câu giải thích loss:

```text
actor_loss giúp policy tăng xác suất action có advantage dương.
critic_loss giúp value function dự đoán TD target chính xác hơn.
entropy bonus khuyến khích exploration, tránh policy collapse quá sớm.
```

---

# PHẦN H — CODE MẪU STABLE-BASELINES3

## 19. A2C

```python
from stable_baselines3 import A2C
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback
import numpy as np
import os

os.makedirs("./logs/a2c", exist_ok=True)

train_env = make_vec_env(lambda: EdgeOffloadingEnv(), n_envs=1, seed=42)
eval_env = make_vec_env(lambda: EdgeOffloadingEnv(), n_envs=1, seed=142)

eval_callback = EvalCallback(
    eval_env,
    best_model_save_path="./logs/a2c",
    log_path="./logs/a2c",
    eval_freq=500,
    deterministic=True,
    render=False
)

model = A2C(
    "MlpPolicy",
    train_env,
    learning_rate=7e-4,
    gamma=0.99,
    verbose=1,
    seed=42
)

model.learn(
    total_timesteps=100_000,
    callback=eval_callback
)

model.save("./logs/a2c/a2c_edge_offloading")
```

Evaluate A2C:

```python
def a2c_action_fn(state, env):
    action, _ = model.predict(state, deterministic=True)
    return int(np.asarray(action).item())

a2c_eval = evaluate_policy(lambda: EdgeOffloadingEnv(), a2c_action_fn)
print(a2c_eval)
```

---

## 20. PPO

PPO thường là lựa chọn mặc định tốt khi không chắc chọn gì.

```python
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback

train_env = make_vec_env(lambda: EdgeOffloadingEnv(), n_envs=4, seed=42)
eval_env = make_vec_env(lambda: EdgeOffloadingEnv(), n_envs=1, seed=142)

eval_callback = EvalCallback(
    eval_env,
    best_model_save_path="./logs/ppo",
    log_path="./logs/ppo",
    eval_freq=1000,
    deterministic=True,
    render=False
)

ppo_model = PPO(
    "MlpPolicy",
    train_env,
    learning_rate=3e-4,
    n_steps=2048,
    batch_size=64,
    n_epochs=10,
    gamma=0.99,
    gae_lambda=0.95,
    clip_range=0.2,
    ent_coef=0.01,
    vf_coef=0.5,
    verbose=1,
    seed=42
)

ppo_model.learn(
    total_timesteps=200_000,
    callback=eval_callback
)

ppo_model.save("./logs/ppo/ppo_edge_offloading")
```

Evaluate PPO:

```python
def ppo_action_fn(state, env):
    action, _ = ppo_model.predict(state, deterministic=True)
    return int(np.asarray(action).item())

ppo_eval = evaluate_policy(lambda: EdgeOffloadingEnv(), ppo_action_fn)
print(ppo_eval)
```

Giải thích tham số PPO:

| Tham số | Ý nghĩa |
|---|---|
| learning_rate | Tốc độ update trọng số |
| n_steps | Số step rollout trước mỗi lần update |
| batch_size | Kích thước minibatch khi update |
| n_epochs | Số lần lặp qua dữ liệu rollout |
| gamma | Mức coi trọng reward tương lai |
| gae_lambda | Làm mượt advantage |
| clip_range | Giới hạn policy update |
| ent_coef | Khuyến khích exploration |
| vf_coef | Trọng số critic loss |

---

## 21. DQN

DQN hợp với action rời rạc.

```python
from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import EvalCallback

train_env = EdgeOffloadingEnv()
eval_env = EdgeOffloadingEnv()

eval_callback = EvalCallback(
    eval_env,
    best_model_save_path="./logs/dqn",
    log_path="./logs/dqn",
    eval_freq=1000,
    deterministic=True,
    render=False
)

dqn_model = DQN(
    "MlpPolicy",
    train_env,
    learning_rate=1e-4,
    buffer_size=50_000,
    learning_starts=1_000,
    batch_size=64,
    gamma=0.99,
    train_freq=4,
    target_update_interval=10_000,
    exploration_fraction=0.2,
    exploration_initial_eps=1.0,
    exploration_final_eps=0.05,
    verbose=1,
    seed=42
)

dqn_model.learn(
    total_timesteps=200_000,
    callback=eval_callback
)

dqn_model.save("./logs/dqn/dqn_edge_offloading")
```

Evaluate DQN:

```python
def dqn_action_fn(state, env):
    action, _ = dqn_model.predict(state, deterministic=True)
    return int(np.asarray(action).item())

dqn_eval = evaluate_policy(lambda: EdgeOffloadingEnv(), dqn_action_fn)
print(dqn_eval)
```

Giải thích tham số DQN:

| Tham số | Ý nghĩa |
|---|---|
| buffer_size | Kích thước replay buffer |
| learning_starts | Số step random trước khi học |
| train_freq | Bao nhiêu step thì update |
| target_update_interval | Bao lâu update target network |
| exploration_fraction | Phần đầu training dùng để giảm epsilon |
| exploration_initial_eps | Epsilon ban đầu |
| exploration_final_eps | Epsilon cuối |

---

# PHẦN I — NORMALIZATION

## 22. Normalize state thủ công

Dùng khi feature có scale khác nhau.

```python
def normalize_state(state, env):
    low = env.observation_space.low
    high = env.observation_space.high
    return (state - low) / (high - low + 1e-8)
```

Vì sao cần:

```text
input_size có range 1-20
workload có range 0.1-3
edge_queue có range 0-200

Nếu không normalize, feature edge_queue có thể chi phối gradient mạnh hơn feature khác.
```

---

## 23. VecNormalize với SB3

```python
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

train_env = DummyVecEnv([lambda: EdgeOffloadingEnv()])
train_env = VecNormalize(
    train_env,
    norm_obs=True,
    norm_reward=True,
    clip_obs=10.0
)

model = PPO("MlpPolicy", train_env, verbose=1)
model.learn(total_timesteps=200_000)

train_env.save("./logs/vecnormalize.pkl")
model.save("./logs/ppo_norm")
```

Load để evaluate:

```python
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

eval_env = DummyVecEnv([lambda: EdgeOffloadingEnv()])
eval_env = VecNormalize.load("./logs/vecnormalize.pkl", eval_env)

eval_env.training = False
eval_env.norm_reward = False

model = PPO.load("./logs/ppo_norm", env=eval_env)

obs = eval_env.reset()
action, _ = model.predict(obs, deterministic=True)
```

Lỗi thường gặp:

```text
Train có normalize nhưng test/evaluate không normalize → policy hành động sai.
Train và test dùng khác scaler → kết quả không ổn định.
```

---

# PHẦN J — DEBUG AGENT BỊ BIAS / COLLAPSE

## 24. Agent luôn chọn một action

Ví dụ:

```text
Device % = 100%
Edge % = 0%
```

Có thể do:

```text
1. Reward thiên lệch thật: device thường tốt hơn edge.
2. Exploration quá thấp.
3. Entropy coefficient quá nhỏ.
4. Learning rate quá cao làm policy collapse sớm.
5. State không đủ thông tin để phân biệt khi nào edge tốt.
6. Reward scale làm critic học kém.
7. Evaluation dùng deterministic quá sớm.
```

Cách kiểm tra:

```python
def inspect_action_distribution(model, env, episodes=20):
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
    print("Action counts:", counts)
    print("Action pct:", pct)
```

---

## 25. So sánh action của policy với greedy

Dùng để xem agent sai ở vùng nào.

```python
def compare_agent_to_greedy(model, env, n_samples=1000):
    mismatch = 0
    rows = []

    for i in range(n_samples):
        state, _ = env.reset(seed=5000 + i)

        agent_action, _ = model.predict(state, deterministic=True)
        agent_action = int(np.asarray(agent_action).item())

        greedy_action = policy_greedy_latency(state, env)

        if agent_action != greedy_action:
            mismatch += 1
            t_device, t_edge = env._compute_latencies(state)
            rows.append({
                "state": state,
                "agent_action": agent_action,
                "greedy_action": greedy_action,
                "t_device": t_device,
                "t_edge": t_edge,
                "gap": abs(t_device - t_edge)
            })

    print("Mismatch rate:", mismatch / n_samples)

    return rows
```

Cách nhận xét:

```text
Nếu mismatch chủ yếu xảy ra khi t_device và t_edge gần nhau thì lỗi không nghiêm trọng.
Nếu mismatch xảy ra khi một action rõ ràng tốt hơn nhiều thì policy chưa học đúng.
```

---

## 26. Agent reward không tăng

Nguyên nhân và cách sửa:

| Triệu chứng | Nguyên nhân có thể | Cách chỉnh |
|---|---|---|
| Reward phẳng | lr quá thấp hoặc reward sparse | tăng lr, reward shaping |
| Reward dao động mạnh | lr quá cao | giảm lr |
| Policy chọn 1 action | thiếu exploration | tăng entropy/epsilon |
| Critic loss lớn | reward scale xấu | normalize reward |
| Train tốt test kém | overfit seed | random seed, eval nhiều episode |
| Không vượt baseline | thuật toán chưa phù hợp | thử PPO/DQN, tune reward |

---

## 27. Debug bằng info dictionary

Trong `step()`, nên trả thêm metric:

```python
info = {
    "latency": latency,
    "latency_device": t_device,
    "latency_edge": t_edge,
    "action": action,
    "action_name": action_name,
    "is_greedy_action": action == (0 if t_device <= t_edge else 1)
}
```

Khi evaluate:

```python
greedy_correct = []

while not done:
    action = action_fn(state, env)
    state, reward, terminated, truncated, info = env.step(action)

    greedy_correct.append(info["is_greedy_action"])

    done = terminated or truncated

print("Greedy agreement:", np.mean(greedy_correct))
```

---

# PHẦN K — HYPERPARAMETER CHEAT SHEET

## 28. Learning rate

```text
lr cao:
- học nhanh
- dễ dao động/collapse

lr thấp:
- học ổn định
- có thể học rất chậm
```

Gợi ý:

```text
PPO: 3e-4
A2C: 7e-4
DQN: 1e-4
Manual AC: 1e-3 hoặc 3e-4
```

Nếu loss nổ:

```python
learning_rate = 1e-4
```

Nếu reward không tăng:

```python
learning_rate = 3e-4  # hoặc tăng nhẹ
```

---

## 29. Gamma

```text
gamma gần 0: coi trọng reward trước mắt.
gamma gần 1: coi trọng reward dài hạn.
```

Gợi ý:

```text
Bài quyết định tức thời như Edge Offloading: gamma = 0.95 đến 0.99.
Bài long-horizon như navigation: gamma = 0.99 hoặc 0.995.
```

---

## 30. Entropy coefficient

Entropy giúp exploration.

```text
ent_coef cao:
- agent khám phá nhiều hơn
- học chậm hơn
- tránh collapse

ent_coef thấp:
- policy nhanh deterministic
- dễ kẹt vào một action
```

Gợi ý:

```text
PPO: ent_coef = 0.0, 0.001, 0.01
Manual AC: entropy_coef = 0.01
```

Nếu agent always action 0:

```python
ent_coef = 0.01  # PPO
entropy_coef = 0.02  # manual AC
```

Nếu policy quá random sau training:

```python
ent_coef = 0.001
```

---

## 31. PPO clip_range

```text
clip_range nhỏ:
- update thận trọng
- ổn định
- học chậm

clip_range lớn:
- update mạnh
- học nhanh
- dễ bất ổn
```

Gợi ý:

```text
clip_range = 0.1 đến 0.3
default thường dùng = 0.2
```

---

## 32. PPO n_steps, batch_size, n_epochs

```text
n_steps:
- số step thu thập trước mỗi lần update.

batch_size:
- kích thước mini-batch để update.

n_epochs:
- số lần học lại trên cùng rollout data.
```

Nếu `n_epochs` quá cao:

```text
Policy có thể overfit rollout cũ.
```

Nếu `n_epochs` quá thấp:

```text
Dữ liệu rollout chưa được tận dụng đủ.
```

Gợi ý:

```python
n_steps = 2048
batch_size = 64
n_epochs = 10
```

---

## 33. DQN epsilon

```text
epsilon cao: chọn action random nhiều.
epsilon thấp: chọn action theo Q-value nhiều.
```

Gợi ý:

```python
exploration_initial_eps = 1.0
exploration_final_eps = 0.05
exploration_fraction = 0.2
```

Nếu DQN bị bias 1 action:

```python
exploration_fraction = 0.4
exploration_final_eps = 0.1
```

Nếu DQN học chậm:

```python
learning_starts = 500
train_freq = 1
```

---

## 34. Target update interval trong DQN

```text
target_update_interval là số step sau đó copy online network sang target network.
```

Nếu update quá thường xuyên:

```text
Target thay đổi liên tục, training kém ổn định.
```

Nếu update quá chậm:

```text
Target quá cũ, học chậm.
```

Gợi ý:

```python
target_update_interval = 1000    # môi trường nhỏ
target_update_interval = 10000   # môi trường lớn hơn
```

---

# PHẦN L — EVALUATION VÀ SO SÁNH

## 35. Metrics cần có

```text
1. Avg Reward
2. Avg Latency / Cost
3. P95 Latency
4. Action distribution
5. Gap to Greedy
6. Success rate / crash rate nếu có
7. Episode length
8. Stability across seeds
```

---

## 36. Code tạo bảng so sánh

```python
import pandas as pd

comparison_rows = []

def add_result(method_name, result, greedy_latency=None):
    row = {
        "Method": method_name,
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

    comparison_rows.append(row)

greedy_latency = baseline_results["Greedy latency"]["avg_latency"]

for name, res in baseline_results.items():
    add_result(name, res, greedy_latency)

add_result("A2C", a2c_eval, greedy_latency)
add_result("PPO", ppo_eval, greedy_latency)

df = pd.DataFrame(comparison_rows)
df = df.sort_values(by="Avg Latency ↓")
print(df)
```

---

## 37. Code plot learning curve

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

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
```

---

## 38. Deterministic vs stochastic evaluation

```python
def evaluate_sb3_model(model, env_fn, episodes=100, deterministic=True):
    rewards = []
    action_counts = np.zeros(2, dtype=np.int64)

    for ep in range(episodes):
        env = env_fn()
        state, _ = env.reset(seed=3000 + ep)
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

    return {
        "avg_reward": float(np.mean(rewards)),
        "std_reward": float(np.std(rewards)),
        "action_counts": action_counts
    }
```

Cách nhận xét:

```text
Deterministic evaluation dùng action xác suất cao nhất, phù hợp khi deployment.
Stochastic evaluation vẫn sampling theo policy, phù hợp để kiểm tra exploration.
Nếu stochastic tốt hơn deterministic, có thể policy chưa đủ tự tin hoặc argmax đang chọn action lệch trong một số state.
```

---

# PHẦN M — CÁC CASE THƯỜNG GẶP TRONG ĐỀ THI

## 39. Case 1: Minimize latency

```text
State:
- task size
- workload
- bandwidth
- queue length

Action:
- local
- edge
- cloud

Reward:
- latency
- energy penalty nếu có
```

Code reward:

```python
reward = -latency_ms / 1000.0
```

Nếu thêm energy:

```python
reward = -(
    w_latency * latency_ms / 1000.0
    + w_energy * energy_joule
)
```

---

## 40. Case 2: Robot navigation

```text
State:
- position
- velocity
- distance to goal
- obstacle distance

Action:
- move left/right/up/down
- hoặc continuous velocity

Reward:
- progress toward goal
- collision penalty
- goal bonus
- time penalty
```

Code reward:

```python
distance = np.linalg.norm(agent_pos - goal_pos)
progress = prev_distance - distance

reward = 2.0 * progress - 0.01

if collision:
    reward -= 10.0
    terminated = True

if distance < goal_threshold:
    reward += 20.0
    terminated = True
```

---

## 41. Case 3: Trading

```text
State:
- price indicators
- moving averages
- inventory
- cash
- volatility

Action:
- buy
- sell
- hold

Reward:
- portfolio value change
- transaction cost penalty
- risk penalty
```

Code reward:

```python
portfolio_value = cash + shares * current_price
reward = (portfolio_value - prev_portfolio_value) / prev_portfolio_value

transaction_penalty = 0.001 * abs(trade_amount)
reward -= transaction_penalty
```

Cẩn thận:

```text
Không dùng future price trong state vì gây data leakage.
```

---

## 42. Case 4: Predictive maintenance

```text
State:
- sensor values
- RUL prediction
- health index
- fuel / cost / location nếu có

Action:
- continue
- reduce load
- maintenance
- reroute

Reward:
- task progress
- maintenance đúng lúc
- crash penalty
- unnecessary maintenance penalty
```

Code reward mẫu:

```python
reward = 0.0

# Khuyến khích hoàn thành nhiệm vụ
reward += progress_delta

# Phạt nguy hiểm khi RUL thấp mà vẫn tiếp tục
if rul < 20 and action == ACTION_CONTINUE:
    reward -= 2.0

# Thưởng maintenance đúng lúc
if action == ACTION_MAINTAIN:
    if 10 <= rul <= 40:
        reward += 5.0
    elif rul > 80:
        reward -= 3.0   # bảo trì quá sớm
    else:
        reward -= 5.0   # quá muộn

# Phạt crash mạnh
if crashed:
    reward -= 20.0
    terminated = True
```

---

## 43. Case 5: Scheduling / resource allocation

```text
State:
- queue length
- server load
- deadline
- task priority

Action:
- assign to server A/B/C
- reject/accept task
- allocate resource amount

Reward:
- throughput
- deadline miss penalty
- load imbalance penalty
```

Code reward:

```python
reward = 0.0

if task_finished:
    reward += task_priority

if missed_deadline:
    reward -= 5.0

reward -= 0.1 * load_imbalance
```

---

# PHẦN N — CÂU TRẢ LỜI MẪU KHI BỊ HỎI

## 44. Tại sao không dùng supervised learning?

```text
Supervised learning cần nhãn action đúng cho từng state.
Trong RL, agent không có nhãn action tối ưu sẵn mà học thông qua tương tác với môi trường và reward.
Vì vậy RL phù hợp khi mục tiêu là tối ưu chuỗi quyết định, đặc biệt khi action hiện tại ảnh hưởng đến trạng thái và reward tương lai.
```

---

## 45. Tại sao phải có baseline greedy?

```text
Greedy là heuristic mạnh vì nó dùng thông tin tức thời để chọn action tốt nhất ở từng bước.
Nó giúp kiểm tra agent RL có học được logic hợp lý không.
Nếu RL kém hơn random hoặc static baseline thì mô hình chưa đạt.
Nếu RL gần greedy thì chứng tỏ policy học được quan hệ giữa state và action.
```

---

## 46. Tại sao reward âm?

```text
Vì mục tiêu là minimize cost/latency, trong khi RL thường maximize reward.
Đặt reward = -cost giúp biến bài toán minimize thành maximize.
Cost càng nhỏ thì reward càng ít âm, tức là tốt hơn.
```

---

## 47. Tại sao agent bị bias chọn device?

```text
Có hai khả năng.
Một là dữ liệu/môi trường thật sự khiến device thường tốt hơn edge.
Hai là quá trình học bị collapse do exploration thấp, learning rate cao, reward scale chưa tốt hoặc state chưa đủ thông tin.
Em sẽ kiểm tra action distribution, so sánh với greedy policy, tăng entropy hoặc epsilon, normalize state/reward và đánh giá lại trên nhiều seed.
```

---

## 48. Tại sao dùng Actor-Critic thay vì REINFORCE?

```text
REINFORCE dùng return toàn episode nên variance cao.
Actor-Critic thêm critic để ước lượng value và dùng advantage cho update policy.
Nhờ vậy gradient ổn định hơn và học nhanh hơn trong nhiều bài toán.
```

---

## 49. Tại sao dùng PPO?

```text
PPO là policy gradient method ổn định vì nó giới hạn mức thay đổi policy qua clip objective.
Điều này giúp tránh update quá lớn làm policy collapse.
PPO cũng dùng được cho cả action rời rạc và liên tục nên thường là lựa chọn mặc định tốt.
```

---

## 50. Nếu reward tăng nhưng latency không giảm thì sao?

```text
Điều đó cho thấy reward chưa phản ánh đúng metric chính.
Có thể agent đang tối ưu một thành phần phụ trong reward.
Cần kiểm tra từng thành phần reward, log latency riêng, và điều chỉnh trọng số để reward aligned với objective thực tế.
```

---

# PHẦN O — CHECKLIST VIẾT CODE TRONG PHÒNG THI

## 51. Checklist environment

```text
[ ] import gymnasium, spaces, numpy
[ ] class Env(gym.Env)
[ ] __init__()
[ ] observation_space
[ ] action_space
[ ] _sample_state()
[ ] _compute_metric()
[ ] reset(seed=None, options=None)
[ ] step(action)
[ ] reward
[ ] terminated/truncated
[ ] info dict
[ ] check_env
```

---

## 52. Checklist training

```text
[ ] Set seed
[ ] Normalize state nếu feature khác scale
[ ] Có baseline
[ ] Train RL
[ ] Plot reward curve
[ ] Evaluate trên nhiều episode
[ ] So sánh avg reward / avg latency / action %
[ ] Debug nếu policy collapse
```

---

## 53. Checklist báo cáo/nhận xét

```text
[ ] Agent học mục tiêu gì?
[ ] Reward có khớp mục tiêu không?
[ ] Baseline nào mạnh nhất?
[ ] RL hơn/kém baseline nào?
[ ] Action distribution có hợp lý không?
[ ] Có dấu hiệu bias/collapse không?
[ ] Hyperparameter nào ảnh hưởng lớn?
[ ] Hạn chế môi trường là gì?
```

---

# PHẦN P — MINI TEMPLATE HOÀN CHỈNH

Dùng khi cần viết nhanh một bài từ đầu.

```python
import numpy as np
import gymnasium as gym
from gymnasium import spaces

class MyEnv(gym.Env):
    def __init__(self, max_steps=50):
        super().__init__()
        self.max_steps = max_steps
        self.step_count = 0

        self.observation_space = spaces.Box(
            low=np.array([0, 0], dtype=np.float32),
            high=np.array([1, 1], dtype=np.float32),
            dtype=np.float32
        )

        self.action_space = spaces.Discrete(2)
        self.state = None

    def _sample_state(self):
        return self.np_random.uniform(
            self.observation_space.low,
            self.observation_space.high
        ).astype(np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.step_count = 0
        self.state = self._sample_state()
        return self.state, {}

    def step(self, action):
        assert self.action_space.contains(action)

        # Example cost
        x1, x2 = self.state
        cost_action_0 = x1 + 0.5 * x2
        cost_action_1 = 0.5 * x1 + x2

        cost = cost_action_0 if action == 0 else cost_action_1
        reward = -cost

        self.step_count += 1
        terminated = False
        truncated = self.step_count >= self.max_steps

        info = {
            "cost": cost,
            "action": action
        }

        self.state = self._sample_state()
        return self.state, reward, terminated, truncated, info


def greedy_policy(state, env):
    x1, x2 = state
    cost_action_0 = x1 + 0.5 * x2
    cost_action_1 = 0.5 * x1 + x2
    return 0 if cost_action_0 <= cost_action_1 else 1


env = MyEnv()
obs, _ = env.reset(seed=42)

for _ in range(5):
    action = greedy_policy(obs, env)
    obs, reward, terminated, truncated, info = env.step(action)
    print(reward, info)
```

---

# PHẦN Q — KẾT LUẬN NGẮN ĐỂ VIẾT VÀO BÀI

```text
Bài toán được mô hình hóa dưới dạng MDP với state mô tả điều kiện hệ thống,
action là quyết định của agent, reward phản ánh trực tiếp mục tiêu tối ưu.
Trước khi huấn luyện RL, cần xây baseline để có mốc so sánh.
Với action rời rạc, có thể dùng DQN, A2C hoặc PPO; trong đó PPO thường ổn định hơn,
DQN phù hợp với discrete action, còn Actor-Critic minh họa rõ cơ chế policy-based.
Khi agent bị lệch về một action, cần kiểm tra action distribution, reward scale,
exploration, normalization và so sánh với greedy policy để xác định lỗi do môi trường
hay do quá trình huấn luyện.
```

---

# PHẦN R — BẢNG CHẨN ĐOÁN NHANH

| Vấn đề | Dấu hiệu | Cách xử lý nhanh |
|---|---|---|
| Reward không tăng | curve phẳng | tăng timesteps, chỉnh lr, shaping |
| Reward dao động | curve lên xuống mạnh | giảm lr, normalize reward |
| Agent chọn 1 action | action % lệch 100/0 | tăng entropy/epsilon |
| RL kém greedy nhiều | gap lớn | thêm training, kiểm tra state/reward |
| Train tốt test kém | eval seed khác giảm mạnh | nhiều seed, normalize đúng |
| Critic loss cao | value không học | reward scale, vf_coef, lr |
| DQN không học | Q-value kém | tăng buffer, learning_starts, epsilon |
| PPO collapse | entropy giảm nhanh | giảm lr, tăng ent_coef, giảm clip_range |
| Reward tốt metric xấu | tối ưu sai mục tiêu | sửa reward alignment |
| Edge/device bias | action lệch | so sánh với greedy, kiểm tra phân phối state |

---

## Ghi nhớ cuối

```text
Đừng nhảy vào thuật toán trước.
Luôn đi theo thứ tự:

MDP → Environment → Reward → Baseline → Algorithm → Evaluation → Debug.
```
