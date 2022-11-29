import os
import sys
import numpy as np
from stable_baselines3.common.vec_env import DummyVecEnv, VecEnv

path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.insert(0, path)

from somo_rl.utils.load_env import load_env
from somo_rl.utils.load_run_config_file import load_run_config_file


def evaluate_policy(model, run_ID, env=None, n_eval_episodes=10, deterministic=True, render=False):
    _, run_config = load_run_config_file(run_ID)
    close = False
    if not env:
        env = load_env(run_config)
        close = True

    num_steps = int(run_config["max_episode_steps"])

    if not isinstance(env, VecEnv):
        if render:
            _ = env.reset(run_render=render)
        env = DummyVecEnv([lambda: env])

    n_envs = env.num_envs
    episode_rewards = []
    episode_z_rotations = []
    episode_count = 0
    while episode_count < n_eval_episodes:
        total_reward = np.zeros(n_envs)
        obs = env.reset()
        for step in range(num_steps):
            action, _ = model.predict(obs, deterministic=deterministic)
            obs, reward, _, info = env.step(action)
            total_reward += reward

            if render:
                env.render()

        for i in range(n_envs):
            episode_rewards.append(total_reward[i])
            if 'z_rotation_step' in info[i]:
                episode_z_rotations.append(info[i]['z_rotation_step'])
            else:
                episode_z_rotations.append(np.degrees(info[i]['z_rotation']))
            episode_count += 1

    mean_reward = np.mean(episode_rewards)
    std_reward = np.std(episode_rewards)
    mean_z_rotation = np.mean(episode_z_rotations)
    std_z_rotation = np.std(episode_z_rotations)

    if close:
        env.close()

    return mean_reward, std_reward, mean_z_rotation, std_z_rotation
