import os
import sys
import numpy as np
from copy import deepcopy

path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.insert(0, path)

from somo_rl.utils.load_run_config_file import load_run_config_file
from somo_rl.utils.load_env import load_env


def evaluate_policy(model, run_ID, n_eval_episodes=10, deterministic=True, render=False):

    _, run_config = load_run_config_file(run_ID)
    env = load_env(run_config)
    episode_rewards, episode_lengths, z_rotations = [], [], []
    num_steps = int(run_config["max_episode_steps"])
    while len(episode_rewards) < n_eval_episodes:
        obs = env.reset(run_render=render)
        actions = []
        observations = []
        rewards = []
        total_reward = 0
        observations.append(deepcopy(obs))
        for i in range(num_steps):
            action, _states = model.predict(obs, deterministic=deterministic)
            obs, reward, _dones, info = env.step(action)
            total_reward += reward
            actions.append(deepcopy(action))
            rewards.append(deepcopy(reward))

            if render:
                env.render()

        episode_rewards.append(total_reward)

        if 'z_rotation_step' in info:
            z_rotation = info['z_rotation_step']
        else:
            z_rotation = np.degrees(info['z_rotation'])

        z_rotations.append(z_rotation)

    mean_reward = np.mean(episode_rewards)
    std_reward = np.std(episode_rewards)
    mean_z_rotation = np.mean(z_rotations)
    std_z_rotation = np.std(z_rotations)
    env.close()

    return mean_reward, std_reward, mean_z_rotation, std_z_rotation
