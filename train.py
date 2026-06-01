"""Training scripts for Stable-Baselines3 agents (A2C, PPO, DQN)."""

import os
import numpy as np
from stable_baselines3 import A2C, PPO, DQN
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback

from envs.edge_offloading import EdgeOffloadingEnv


def make_eval_callback(log_dir, eval_freq=1000):
    os.makedirs(log_dir, exist_ok=True)
    eval_env = make_vec_env(lambda: EdgeOffloadingEnv(), n_envs=1, seed=142)
    return EvalCallback(
        eval_env,
        best_model_save_path=log_dir,
        log_path=log_dir,
        eval_freq=eval_freq,
        deterministic=True,
        render=False,
    )


# ---------------------------------------------------------------------------
# A2C
# ---------------------------------------------------------------------------

def train_a2c(total_timesteps=100_000, seed=42):
    train_env = make_vec_env(lambda: EdgeOffloadingEnv(), n_envs=1, seed=seed)
    callback = make_eval_callback("./logs/a2c", eval_freq=500)

    model = A2C(
        "MlpPolicy",
        train_env,
        learning_rate=7e-4,
        gamma=0.99,
        verbose=1,
        seed=seed,
    )
    model.learn(total_timesteps=total_timesteps, callback=callback)
    model.save("./logs/a2c/a2c_edge_offloading")
    return model


# ---------------------------------------------------------------------------
# PPO
# ---------------------------------------------------------------------------

def train_ppo(total_timesteps=200_000, seed=42):
    train_env = make_vec_env(lambda: EdgeOffloadingEnv(), n_envs=4, seed=seed)
    callback = make_eval_callback("./logs/ppo", eval_freq=1000)

    model = PPO(
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
        seed=seed,
    )
    model.learn(total_timesteps=total_timesteps, callback=callback)
    model.save("./logs/ppo/ppo_edge_offloading")
    return model


# ---------------------------------------------------------------------------
# DQN
# ---------------------------------------------------------------------------

def train_dqn(total_timesteps=200_000, seed=42):
    train_env = EdgeOffloadingEnv()
    callback = make_eval_callback("./logs/dqn", eval_freq=1000)

    model = DQN(
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
        seed=seed,
    )
    model.learn(total_timesteps=total_timesteps, callback=callback)
    model.save("./logs/dqn/dqn_edge_offloading")
    return model
