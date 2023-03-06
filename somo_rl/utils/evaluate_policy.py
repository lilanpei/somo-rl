import os
import sys
import numpy as np
import torch as th
from torch import nn
import gym
import pybullet as p
from stable_baselines3.common.vec_env import DummyVecEnv, VecEnv

path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.insert(0, path)

from somo_rl.utils.load_env import load_env
from somo_rl.utils.load_run_config_file import load_run_config_file


def evaluate_policy(model, run_ID, env=None, n_eval_episodes=10, deterministic=True, render=False, vid_filename=None):
    _, run_config = load_run_config_file(run_ID)
    close = False
    if not env:
        env = load_env(run_config)
        close = True

    num_steps = int(run_config["max_episode_steps"])

    if not isinstance(env, VecEnv):
        if render:
            env.reset(run_render=render)
        env = DummyVecEnv([lambda: env])

    n_envs = env.num_envs
    episode_rewards = []
    episode_z_rotations = []
    episode_count = 0
    while episode_count < n_eval_episodes:
        total_reward = np.zeros(n_envs)
        obs = env.reset()
        if render and vid_filename and episode_count == 0:
            logIDvideo = p.startStateLogging(
                p.STATE_LOGGING_VIDEO_MP4, str(vid_filename)
            )
        for _ in range(num_steps):
            action, _ = model.predict(obs, deterministic=deterministic)
            obs, reward, _, info = env.step(action)
            total_reward += reward

            if render:
                env.render()

        if render and vid_filename and episode_count == 0:
            p.stopStateLogging(logIDvideo)

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

def evaluate_obs_model_rnn(agent, obs_model, run_ID, env=None, n_eval_episodes=10, deterministic=True, render=False):
    _, run_config = load_run_config_file(run_ID)
    print("@@@@@@ _evaluate_obs_model_rnn_")
    loss_fn = nn.MSELoss()
    device = "cuda" if th.cuda.is_available() else "cpu"
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
        obs = th.FloatTensor(env.reset())
        hidden = None
        for step in range(num_steps):
            action, _ = agent.predict(obs.cpu(), deterministic=deterministic)
            # if isinstance(agent.action_space, gym.spaces.Box):
            #     clipped_action = np.clip(action, agent.action_space.low, agent.action_space.high)
            input_obs_model = th.cat((obs.to(device), th.as_tensor(action).to(device)), -1)
            obs, hidden = obs_model(input_obs_model.view(1, n_envs, input_obs_model.shape[-1]), hidden)
            obs = th.squeeze(obs).view(n_envs, obs.shape[-1])
            obs_env, reward, _, info = env.step(action) # clipped_action
            print(f"Loss of obs_model_gru step: {step}, {loss_fn(obs, th.from_numpy(obs_env)):.6f}")
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

def evaluate_obs_model_mlp(agent, obs_model, run_ID, env=None, n_eval_episodes=10, deterministic=True, render=False):
    _, run_config = load_run_config_file(run_ID)
    print("@@@@@@ _evaluate_obs_model_mlp_")
    loss_fn = nn.MSELoss()
    device = "cuda" if th.cuda.is_available() else "cpu"
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
        obs = th.FloatTensor(env.reset())
        hidden = None
        for step in range(num_steps):
            action, _ = agent.predict(obs.cpu(), deterministic=deterministic)
            input_obs_model = th.cat((obs.to(device), th.as_tensor(action).to(device)), -1)
            obs = obs_model(input_obs_model)
            obs_env, reward, _, info = env.step(action)
            print(f"Loss of obs_model_mlp step: {step}, {loss_fn(obs, th.from_numpy(obs_env)):.6f}")
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
