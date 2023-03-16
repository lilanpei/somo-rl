import os
import sys
import argparse
import numpy as np
import torch as th
import pandas as pd
from datetime import datetime
from torch.optim.lr_scheduler import StepLR
from torch.utils.data.dataset import random_split
from torch import nn
from torch import optim
from torch.nn import Module
from torch.optim import Optimizer
import torch as th
from tqdm import tqdm
from stable_baselines3 import PPO, A2C
from stable_baselines3.common.utils import set_random_seed
from avalanche.benchmarks.utils import AvalancheTensorDataset, AvalancheDataset, AvalancheConcatDataset, AvalancheSubset, as_regression_dataset
from avalanche.benchmarks.generators import dataset_benchmark
from avalanche.training.supervised import Naive, Cumulative, EWC, JointTraining, Replay, AGEM, GEM, GSS_greedy, GDumb, SynapticIntelligence, MAS
from avalanche.training.plugins import ReplayPlugin, EWCPlugin
from avalanche.training.plugins.lr_scheduling import LRSchedulerPlugin
from avalanche.training.templates.supervised import SupervisedTemplate
from avalanche.benchmarks.utils.dataset_utils import ConstantSequence
from avalanche.training.storage_policy import (
    ExemplarsBuffer,
    BalancedExemplarsBuffer,
)

path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.insert(0, path)

from user_settings import EXPERIMENT_ABS_PATH, EXPERT_ABS_PATH
from somo_rl.utils import construct_policy_model
from somo_rl.utils.load_env import load_env
from somo_rl.utils.evaluate_policy import evaluate_policy
from somo_rl.utils.load_run_config_file import load_run_config_file
from user_settings import EXPERIMENT_ABS_PATH

target_z_rotation = {
    'cube'  : 225,
    'rect'  : 125,
    'bunny' : 150, #125
    'cross' : 180, #150
    'teddy' : 125
}
Length_Experience = 1000 # 1000 episodes = 100000 steps

class RewardPrioritySamplingBuffer(ExemplarsBuffer):
    """Buffer updated with reservoir sampling."""

    def __init__(self, max_size: int):
        """
        :param max_size:
        """
        # The algorithm follows
        # https://en.wikipedia.org/wiki/Reservoir_sampling
        # We sample a random uniform value in [0, 1] for each sample and
        # choose the `size` samples with higher values.
        # This is equivalent to a random selection of `size_samples`
        # from the entire stream.
        super().__init__(max_size)
        # INVARIANT: _buffer_weights is always sorted.
        self._buffer_weights = th.zeros(0)

    def update(self, strategy: "SupervisedTemplate", **kwargs):
        """Update buffer."""
        self.update_from_dataset(strategy.experience.dataset)

    def update_from_dataset(self, new_data: AvalancheDataset):
        """Update the buffer using the given dataset.

        :param new_data:
        :return:
        """
        new_weights = th.tensor(new_data.targets)# * th.rand(len(new_data))
        print(f"$$$$$$ new_weights update_from_dataset: {len(new_weights)}, {self.max_size}, {new_weights[:5]}")

        cat_weights = th.cat([new_weights, self._buffer_weights])
        print(f"$$$$$$ cat_weights update_from_dataset: {len(new_weights)}, {len(self._buffer_weights), {self.max_size}}")
        print("$$$$$$ self._buffer_weights update_from_dataset: ", len(self._buffer_weights), self._buffer_weights[:5])
        # cat_weights = new_weights
        cat_data = AvalancheConcatDataset([new_data, self.buffer])
        print(f"$$$$$$ cat_data update_from_dataset: {len(cat_data)}, {len(new_data)}, {len(self.buffer)}")
        sorted_weights, sorted_idxs = cat_weights.sort(descending=True)

        buffer_idxs = sorted_idxs[: self.max_size]
        self.buffer = AvalancheSubset(cat_data, buffer_idxs)
        self._buffer_weights = sorted_weights[: self.max_size]

    def resize(self, strategy, new_size):
        """Update the maximum size of the buffer."""
        print("$$$$$$ self.buffer resize: ", len(self.buffer), len(self._buffer_weights))
        self.max_size = new_size
        if len(self.buffer) <= self.max_size:
            return
        self.buffer = AvalancheSubset(self.buffer, th.arange(self.max_size))
        self._buffer_weights = self._buffer_weights[: self.max_size]
        print("$$$$$$ self._buffer_weights resize: ", len(self._buffer_weights), self._buffer_weights[:5])

class RewardPriorityExperienceBalancedBuffer(BalancedExemplarsBuffer):
    """Rehearsal buffer with samples balanced over experiences.

    The number of experiences can be fixed up front or adaptive, based on
    the 'adaptive_size' attribute. When adaptive, the memory is equally
    divided over all the unique observed experiences so far.
    """

    def __init__(
        self, max_size: int, adaptive_size: bool = True, num_experiences=None
    ):
        """
        :param max_size: max number of total input samples in the replay
            memory.
        :param adaptive_size: True if mem_size is divided equally over all
                              observed experiences (keys in replay_mem).
        :param num_experiences: If adaptive size is False, the fixed number
                                of experiences to divide capacity over.
        """
        super().__init__(max_size, adaptive_size, num_experiences)

    def update(self, strategy: "SupervisedTemplate", **kwargs):
        new_data = strategy.experience.dataset
        print("$$$$$$ current_experience update: ",  strategy.experience.current_experience)
        num_exps = strategy.clock.train_exp_counter + 1
        lens = self.get_group_lengths(num_exps)

        new_buffer = RewardPrioritySamplingBuffer(lens[-1])
        new_buffer.update_from_dataset(new_data)
        self.buffer_groups[num_exps - 1] = new_buffer

        for ll, b in zip(lens, self.buffer_groups.values()):
            b.resize(strategy, ll)

class Policy_rollout:
    def __init__(self, exp_abs_path, run_ID, render=False, debug=False):
        self.expert_dataset = None
        self.train_expert_dataset = None
        self.test_expert_dataset = None
        self.run_dir, self.run_config = load_run_config_file(run_ID=run_ID, exp_abs_path=exp_abs_path)
        self.env = load_env(self.run_config, render=False, debug=debug)
        self.render = render
        self.load_rollouts()

    def load_rollouts(self):
        saved_data_path = os.path.join(self.run_dir, f"expert_data_{self.run_config['object']}.npz")
        if os.path.isfile(saved_data_path):
            # expert_observations, expert_actions, episodic_rewards, episodic_z_rotation = np.load(saved_data_path)["expert_observations"], np.load(saved_data_path)["expert_actions"], np.load(saved_data_path)["episodic_rewards"], np.load(saved_data_path)["episodic_z_rotation"]
            expert_observations, expert_actions, episodic_rewards = np.load(saved_data_path)["expert_observations"], np.load(saved_data_path)["expert_actions"], np.load(saved_data_path)["episodic_rewards"]
            print(f"Using saved data from : {saved_data_path}")
            print(f"$$$$$$ load_rollouts - before reshape : {expert_observations.shape}, {expert_actions.shape}, {episodic_rewards.shape}")
            episodic_rewards =  np.repeat(episodic_rewards, expert_observations.shape[1])
            # episodic_z_rotation =  np.repeat(episodic_z_rotation, expert_observations.shape[1])
            expert_observations = np.reshape(expert_observations, (-1, expert_observations.shape[-1]))
            expert_actions = np.reshape(expert_actions, (-1, expert_actions.shape[-1]))
            print(f"$$$$$$ load_rollouts - after reshape : {expert_observations.shape}, {expert_actions.shape}, {episodic_rewards.shape}")
        else:
            if os.path.isdir(os.path.join(self.run_dir, "results/processed_data/")) and os.listdir(os.path.join(self.run_dir, "results/processed_data/")):
                print(f"@@@@@@ Preparing data from: {os.path.join(self.run_dir, 'results/processed_data/')}")
                actions_df_list = []
                observations_df_list = []
                for run in os.scandir(os.path.join(self.run_dir, "results/processed_data/")):
                    if os.path.isdir(run):
                        actions_df_list.append(pd.read_pickle(os.path.join(run.path, 'actions.pkl')).iloc[:, :8])
                        observations_df_list.append(pd.read_pickle(os.path.join(run.path, 'observations.pkl')))

                actions_df = pd.concat(actions_df_list)
                observations_df = pd.concat(observations_df_list)

                expert_observations, expert_actions = observations_df.to_numpy(), actions_df.to_numpy()
            else:
                models_dir = os.path.join(self.run_dir, f"models")
                alg = construct_policy_model.ALGS[self.run_config["alg"]]
                rl_agent= alg.load(os.path.join(models_dir, "best_model"))
                list_observations = []
                list_actions = []
                episodic_rewards = []
                episodic_z_rotation = []
                print(f"@@@@@@ Preparing data from the rl_agent and env from: {models_dir}")
                # for epi in tqdm(range(Length_Experience)):
                while len(list_actions) < Length_Experience:
                    episodic_list_observations = []
                    episodic_list_actions = []
                    obs = self.env.reset(run_render=False)#self.render)
                    if len(list_actions)==0: # save the observation after env.reset
                        th.save(obs, os.path.join(models_dir, "obs_tensor_env_reset"))
                    total_reward = 0
                    for _ in range(self.run_config["max_episode_steps"]):
                        episodic_list_observations.append(th.tensor(obs))
                        action, _ = rl_agent.predict(obs, deterministic=False)
                        obs, rewards, _, info = self.env.step(action)
                        episodic_list_actions.append(th.tensor(action))
                        total_reward += rewards
                    print(f"@@@@@@ {self.run_config['object']}, length: {len(list_actions)}, z_rotation: {info['z_rotation_step']}")
                    if info['z_rotation_step'] > target_z_rotation[self.run_config['object']]:
                        list_observations.append(th.stack(episodic_list_observations))
                        list_actions.append(th.stack(episodic_list_actions))
                        episodic_rewards.append(round(total_reward))
                        episodic_z_rotation.append(round(info['z_rotation_step']))
                print(f"$$$$$$ shape: {len(list_observations), {th.stack(list_observations).shape}}")
                expert_observations, expert_actions = (th.stack(list_observations)).detach().numpy(), (th.stack(list_actions)).detach().numpy()
                print(f"$$$$$$ shape: {expert_observations.shape}, {expert_actions.shape}")
                self.env.close()
            np.savez_compressed(
                os.path.join(self.run_dir, f"expert_data_{self.run_config['object']}"),
                expert_observations=expert_observations,
                expert_actions=expert_actions,
                episodic_rewards=np.array(episodic_rewards),
                episodic_z_rotation=np.array(episodic_z_rotation)
            )
            print(f"$$$$$$ load_rollouts - before reshape : {expert_observations.shape}, {expert_actions.shape}, {np.array(episodic_rewards).shape}")
            episodic_rewards =  np.repeat(episodic_rewards, expert_observations.shape[1])
            episodic_z_rotation =  np.repeat(episodic_z_rotation, expert_observations.shape[1])
            expert_observations = np.reshape(expert_observations, (-1, expert_observations.shape[-1]))
            expert_actions = np.reshape(expert_actions, (-1, expert_actions.shape[-1]))
            print(f"$$$$$$ load_rollouts - after reshape : {expert_observations.shape}, {expert_actions.shape}, {episodic_z_rotation.shape}")
        # self.expert_dataset = as_regression_dataset(AvalancheTensorDataset(th.tensor(expert_observations), th.tensor(expert_actions), targets=episodic_z_rotation))
        self.expert_dataset = as_regression_dataset(AvalancheTensorDataset(th.tensor(expert_observations), th.tensor(expert_actions), targets=episodic_rewards))
        print(f"size of expert_dataset_{self.run_config['object']}: {len(self.expert_dataset)}")


class ExMLBuffer:
    def __init__(self, experience, buffer_size=100):
        """ Buffer for ex-model strategies.
        :param experience:
        :param buffer_size:
        """
        self.experience = experience
        self.buffer_size = buffer_size
        self.buffer = []

    def __getitem__(self, item):
        return self.buffer[item]

    def __len__(self):
        return len(self.buffer)


class ReplayBuffer(ExMLBuffer):
    def __init__(self, experience, buffer_size=100):
        """ Buffer that stores a subsample of the experience's data.
        :param experience: `Experience` used to train `model`.
        :param buffer_size:
        """
        super().__init__(experience, buffer_size)

        data = self.experience.dataset
        rem_len = len(data) - buffer_size
        self.buffer, _ = random_split(data, [buffer_size, rem_len])
        t = ConstantSequence(self.experience.task_label, len(self.buffer))
        self.buffer = AvalancheDataset(self.buffer, task_labels=t)


class ReplayED(Naive):
    def __init__(self, model: Module, optimizer: Optimizer, buffer_size, criterion, train_mb_size, train_epochs, device, plugins):
        super().__init__(model=model, optimizer=optimizer, criterion=criterion, train_mb_size=train_mb_size, train_epochs=train_epochs, device=device, plugins=plugins)
        # buffer params.
        self.buffer_size = buffer_size

        # strategy state
        self.cum_data = []

    def train_dataset_adaptation(self):
        super().train_dataset_adaptation()
        buffer = self.make_buffer()
        self.adapted_dataset = buffer

    def make_buffer(self):
        numexp = len(self.cum_data) + 1
        bufsize = self.buffer_size // numexp
        curr_buffer = self.make_buffer_exp(bufsize)

        # subsample previous data
        for i, data in enumerate(self.cum_data):
            removed_els = len(data) - bufsize
            print(f"@@@@@@@@ i: {i}, len(data): {len(data)}, removed_els: {removed_els}")
            if removed_els > 0:
                data, _ = random_split(data, [bufsize, removed_els])
                print(f"$$$$$data: {len(data)}")
            self.cum_data[i] = data

        self.cum_data.append(curr_buffer)
        return AvalancheConcatDataset(self.cum_data)

    def make_buffer_exp(self, bufsize):
        return ReplayBuffer(self.experience, bufsize).buffer


class mymodel(nn.Module):
    def __init__(self, device, model):
        super(mymodel, self).__init__()
        self.student = model
        self.device = device
        self.model = model.policy.to(self.device)

    def forward(self, input):
        # A2C/PPO policy outputs actions, values, log_prob
        # SAC/TD3 policy outputs actions only
        if len(input.shape) > 2:
            # print("$$$$$$ input before reshape:", input.shape)
            x = input.view(-1, input.shape[-1])
            # print("$$$$$$ input after reshape:", x.shape)
        else:
            x = input

        if isinstance(self.student, (A2C, PPO)):
            action, _, _ = self.model(x)
        else:
            # SAC/TD3:
            action = self.model(x)

        if len(input.shape) > 2:
            return action.view(input.shape[0], input.shape[1], action.shape[-1])
        else:
            return action


class Pretrain_agent:
    def __init__(self,
                 student,
                 scenario,
                 strategy_name,
                 num_experts,
                 run_IDs,
                 n_eval_episodes,
                 render,
                 expert_objects,
                 batch_size=100, # 1 episode = 100 steps
                 epochs=10,
                 scheduler_gamma=0.7,
                 learning_rate=1,
                 no_cuda=False,
                 seed=1,
                 test_batch_size=100,
                 ewc_lambda=3000,
                 mem_size=100000
                 ):
        self.student = student
        self.scenario = scenario
        self.batch_size = batch_size
        self.epochs = epochs
        self.scheduler_gamma = scheduler_gamma
        self.learning_rate = learning_rate
        self.no_cuda = no_cuda
        self.seed = seed
        self.test_batch_size = test_batch_size
        self.num_experts = num_experts
        self.render = render
        self.expert_objects = expert_objects
        self.ewc_lambda = ewc_lambda
        self.mem_size = mem_size
        self.strategy_name = strategy_name
        self.videos_dir = os.path.join(EXPERIMENT_ABS_PATH, "render_vids")

        self.use_cuda = not self.no_cuda and th.cuda.is_available()
        self.device = th.device("cuda" if self.use_cuda else "cpu")
        self.kwargs = {"num_workers": 1, "pin_memory": True} if self.use_cuda else {}

        self.criterion = nn.MSELoss()
        self.strategy = None
        self.run_IDs = run_IDs
        self.n_eval_episodes = n_eval_episodes

        # Extract initial policy
        self.model = mymodel(self.device, self.student)

        self.bc_reward = []
        self.bc_rotation = []
        self.behavior_cloning()

    def behavior_cloning(self):
        # Define an Optimizer and a learning rate schedule.
        optimizer = optim.Adadelta(self.model.parameters(), lr=self.learning_rate)

        sched = LRSchedulerPlugin(
            StepLR(optimizer, step_size=1, gamma=self.scheduler_gamma)
        )
        # create strategy
        print(f"@@@@@@ strategy_name : {self.strategy_name}")
        if self.strategy_name == "JointTraining":
            self.strategy = JointTraining(
                self.model,
                optimizer,
                self.criterion,
                train_mb_size=self.batch_size,
                train_epochs=self.epochs,
                device=self.device,
                plugins=[sched],
            )
            self.strategy.train(self.scenario.train_stream)
            # Implant the trained policy network back into the RL student agent
            self.student.policy = self.strategy.model.model

            # evaluate policy
            print(f"@@@@@@ render: {self.render}")
            exp_reward, exp_rotation = [], []
            for idx in range(self.num_experts):
                if self.render:
                    os.makedirs(os.path.join(self.videos_dir, str(self.strategy_name)), exist_ok=True)
                    vid_filename = os.path.join(self.videos_dir, str(self.strategy_name), str(self.strategy_name) + "_" + str(self.expert_objects[idx]) + "_render_vid.mp4")
                    print("@@@@@@ ", vid_filename)

                mean_reward_student, std_reward_student, mean_z_rotation_student, std_z_rotation_student = evaluate_policy(
                    model=self.student, run_ID=self.run_IDs[idx], n_eval_episodes=self.n_eval_episodes,
                    deterministic=False, render=self.render, vid_filename=vid_filename if self.render else None)
                exp_reward.append(mean_reward_student)
                exp_rotation.append(mean_z_rotation_student)
                print(
                    f"Mean reward student on {self.expert_objects[idx]} env = {mean_reward_student:.2f} +/- {std_reward_student:.2f}")
                print(
                    f"z rotation student on {self.expert_objects[idx]} env = {mean_z_rotation_student:.2f} +/- {std_z_rotation_student:.2f}")

            self.bc_reward.append(exp_reward)
            self.bc_rotation.append(exp_rotation)
        else:
            if self.strategy_name == "Cumulative":
                self.strategy = Cumulative(
                    self.model,
                    optimizer,
                    self.criterion,
                    train_epochs=self.epochs,
                    device=self.device,
                    train_mb_size=self.batch_size,
                    plugins=[sched],
                )
            elif self.strategy_name == "GEM":
                    self.strategy = GEM(
                    self.model,
                    optimizer,
                    self.criterion,
                    20000, #patterns_per_exp, Patterns to store in the memory for each experience
                    0.5, #memory_strength, Offset to add to the projection direction
                    train_epochs=self.epochs,
                    device=self.device,
                    train_mb_size=self.batch_size,
                    plugins=[sched],
                )
            elif self.strategy_name == "AGEM":
                    self.strategy = AGEM(
                    self.model,
                    optimizer,
                    self.criterion,
                    20000, #patterns_per_exp, Patterns to store in the memory for each experience
                    100, #sample_size, Number of patterns to sample from memory when projecting gradient
                    train_epochs=self.epochs,
                    device=self.device,
                    train_mb_size=self.batch_size,
                    plugins=[sched],
                )
            elif self.strategy_name == "EWC":
                self.strategy = EWC(
                    self.model,
                    optimizer,
                    self.criterion,
                    ewc_lambda=self.ewc_lambda,
                    mode="separate",
                    train_epochs=self.epochs,
                    device=self.device,
                    train_mb_size=self.batch_size,
                    plugins=[sched],
                )
            elif self.strategy_name == "GDumb":
                self.strategy = GDumb(
                    self.model,
                    optimizer,
                    mem_size=self.mem_size,
                    criterion=self.criterion,
                    train_mb_size=self.batch_size,
                    train_epochs=self.epochs,
                    device=self.device,
                    plugins=[sched],
                )
            elif self.strategy_name == "MAS":
                self.strategy = MAS(
                    self.model,
                    optimizer,
                    criterion=self.criterion,
                    lambda_reg=1.0,
                    alpha=0.5,
                    train_mb_size=self.batch_size,
                    train_epochs=self.epochs,
                    device=self.device,
                    plugins=[sched],
                )
            elif self.strategy_name == "SynapticIntelligence":
                self.strategy = SynapticIntelligence(
                    self.model,
                    optimizer,
                    criterion=self.criterion,
                    si_lambda=0.01,
                    eps=1e-07,
                    train_mb_size=self.batch_size,
                    train_epochs=self.epochs,
                    device=self.device,
                    plugins=[sched],
                )
            elif self.strategy_name == "GSS_greedy":
                self.strategy = GSS_greedy(
                    self.model,
                    optimizer,
                    mem_size=self.mem_size,
                    mem_strength=1,
                    criterion=self.criterion,
                    train_mb_size=self.batch_size,
                    train_epochs=self.epochs,
                    device=self.device,
                    plugins=[sched],
                )
            elif self.strategy_name == "Replay":
                self.strategy = Replay(
                    self.model,
                    optimizer,
                    mem_size=self.mem_size,
                    criterion=self.criterion,
                    train_mb_size=self.batch_size,
                    train_epochs=self.epochs,
                    device=self.device,
                    plugins=[sched],
                )
            elif self.strategy_name == "ReplayRP":
                replay = ReplayPlugin(mem_size=self.mem_size, storage_policy=RewardPriorityExperienceBalancedBuffer(max_size=self.mem_size, adaptive_size=True))
                self.strategy = Naive(
                    self.model,
                    optimizer,
                    criterion=self.criterion,
                    train_mb_size=self.batch_size,
                    train_epochs=self.epochs,
                    device=self.device,
                    plugins=[sched, replay],
                )
            elif self.strategy_name == "ReplayED":
                self.strategy = ReplayED(
                    self.model,
                    optimizer,
                    buffer_size=self.mem_size,
                    criterion=self.criterion,
                    train_mb_size=self.batch_size,
                    train_epochs=self.epochs,
                    device=self.device,
                    plugins=[sched],
                )
            elif self.strategy_name == "EWC_Replay":
                replay = ReplayPlugin(mem_size=self.mem_size)
                ewc = EWCPlugin(ewc_lambda=self.ewc_lambda)
                self.strategy = SupervisedTemplate(
                    self.model,
                    optimizer,
                    criterion=self.criterion,
                    train_mb_size=self.batch_size,
                    train_epochs=self.epochs,
                    device=self.device,
                    plugins=[sched, replay, ewc]
                )
            elif self.strategy_name == "EWC_ReplayED":
                ewc = EWCPlugin(ewc_lambda=self.ewc_lambda)
                self.strategy = ReplayED(
                    self.model,
                    optimizer,
                    buffer_size=self.mem_size,
                    criterion=self.criterion,
                    train_mb_size=self.batch_size,
                    train_epochs=self.epochs,
                    device=self.device,
                    plugins=[sched, ewc],
                )
            elif self.strategy_name == "EWC_ReplayRP":
                ewc = EWCPlugin(ewc_lambda=self.ewc_lambda)
                replay = ReplayPlugin(mem_size=self.mem_size, storage_policy=RewardPriorityExperienceBalancedBuffer(max_size=self.mem_size, adaptive_size=True))
                self.strategy = Naive(
                    self.model,
                    optimizer,
                    criterion=self.criterion,
                    train_mb_size=self.batch_size,
                    train_epochs=self.epochs,
                    device=self.device,
                    plugins=[sched, ewc, replay],
                )
            else:
                self.strategy = Naive(
                    self.model,
                    optimizer,
                    self.criterion,
                    train_mb_size=self.batch_size,
                    train_epochs=self.epochs,
                    device=self.device,
                    plugins=[sched],
                )

            # train on the selected scenario with the chosen strategy
            print("Starting experiment...")
            for experience in self.scenario.train_stream:
                exp_reward, exp_rotation = [], []
                print("Start training on experience ", experience.current_experience)
                self.strategy.train(experience)
                print("End training on experience", experience.current_experience)

                # Implant the trained policy network back into the RL student agent
                self.student.policy = self.strategy.model.model

                # evaluate policy
                for idx in range(self.num_experts):
                    if self.render:
                        os.makedirs(os.path.join(self.videos_dir, str(self.strategy_name)), exist_ok=True)
                        vid_filename = os.path.join(self.videos_dir, str(self.strategy_name), "experience_" + str(experience.current_experience) + "_" + str(self.strategy_name) + "_" + str(self.expert_objects[idx]) + "_render_vid.mp4")
                        print("@@@@@@ ", vid_filename)
                    mean_reward_student, std_reward_student, mean_z_rotation_student, std_z_rotation_student = evaluate_policy(
                        model=self.student, run_ID=self.run_IDs[idx], n_eval_episodes=self.n_eval_episodes, deterministic=True,
                        render=self.render, vid_filename=vid_filename if self.render else None)
                    exp_reward.append(mean_reward_student)
                    exp_rotation.append(mean_z_rotation_student)
                    print(
                        f"Mean reward student on {self.expert_objects[idx]} env = {mean_reward_student:.2f} +/- {std_reward_student:.2f}")
                    print(
                        f"z rotation student on {self.expert_objects[idx]} env = {mean_z_rotation_student:.2f} +/- {std_z_rotation_student:.2f}")
                self.bc_reward.append(exp_reward)
                self.bc_rotation.append(exp_rotation)

def run(run_IDs, exp_abs_path=EXPERIMENT_ABS_PATH, seed=100, render=False, debug=False, n_eval_episodes=10, strategy_name="Naive", num_runs=5):
    print(f"seed = {seed}, n_eval_episodes = {n_eval_episodes}")
    set_random_seed(seed)
    num_experts = len(run_IDs)
    expert_policy = []

    # Load expert agents
    for idx in range(num_experts):
        print("@@@@@@ idx: ", idx, run_IDs[idx])
        expert_policy.append(Policy_rollout(exp_abs_path=exp_abs_path, run_ID=run_IDs[idx], render=render, debug=debug))

    # if num_experts > 1:
    #     # Evaluate the expert trained for multi-objects manipulation
    #     # baseline_runID = [run_IDs[0][0], "PPO_multi_objects", run_IDs[0][-1]]
    #     baseline_run_group_name = (list(run_IDs[0][1].split("_")))[0] + "_" + "_".join([(list(run_IDs[i][1].split("_")))[-1] for i in range(num_experts)])
    #     baseline_runID = [run_IDs[2][0], baseline_run_group_name, run_IDs[0][-1]]
    #     print(f"@@@@@@ baseline_runID: {baseline_runID}")
    #     try:
    #         baseline_run_dir, baseline_run_config = load_run_config_file(baseline_runID)
    #         for i in range(num_experts):
    #             if render:
    #                 os.makedirs(os.path.join(EXPERIMENT_ABS_PATH, "render_vids", "Oracle"), exist_ok=True)
    #                 vid_filename = os.path.join(EXPERIMENT_ABS_PATH, "render_vids",  "Oracle", "Oracle_Agent_on_" + str(expert_policy[i].run_config['object']) + "_render_vid.mp4")
    #                 print("@@@@@@ ", vid_filename)
    #             alg = construct_policy_model.ALGS[expert_policy[i].run_config["alg"]]
    #             mean_reward_expert, std_reward_expert, mean_z_rotation_expert, std_z_rotation_expert = evaluate_policy(
    #                 model=alg.load(os.path.join(baseline_run_dir, "models/best_model")), run_ID=run_IDs[i],
    #                 n_eval_episodes=n_eval_episodes, deterministic=True, render=render, vid_filename=vid_filename if render else None)
    #             print(
    #                 f"Mean reward {baseline_run_config['object']} expert on {expert_policy[i].run_config['object']} env = {mean_reward_expert:.2f} +/- {std_reward_expert:.2f}")
    #             print(
    #                 f"z rotation {baseline_run_config['object']} expert on {expert_policy[i].run_config['object']} env = {mean_z_rotation_expert:.2f} +/- {std_z_rotation_expert:.2f}")
    #     except:
    #         print(f"Exception : The expert trained for multi-objects manipulation does not exist : {baseline_runID}")

    # # Evaluate the expert trained for individual-object manipulation, deterministic=False
    # for idx in range(num_experts):
    #     for j in range(num_experts):
    #        if render:
    #            os.makedirs(os.path.join(EXPERIMENT_ABS_PATH, "render_vids", "Expert"), exist_ok=True)
    #            vid_filename = os.path.join(EXPERIMENT_ABS_PATH, "render_vids",  "Expert", str(expert_policy[idx].run_config['object']) + "_expert_on_" + str(expert_policy[j].run_config['object']) + "_env_render_vid.mp4")
    #            print("@@@@@@ ", vid_filename)
    #         alg = construct_policy_model.ALGS[expert_policy[idx].run_config["alg"]]
    #         mean_reward_expert, std_reward_expert, mean_z_rotation_expert, std_z_rotation_expert = evaluate_policy(
    #             model=alg.load(os.path.join(expert_policy[idx].run_dir, "models/best_model")), run_ID=run_IDs[j],
    #             n_eval_episodes=n_eval_episodes, deterministic=False, render=render, vid_filename=vid_filename if render else None)
    #         print(
    #             f"@@@@@@ stochastic: Mean reward {expert_policy[idx].run_config['object']} expert on {expert_policy[j].run_config['object']} env = {mean_reward_expert:.2f} +/- {std_reward_expert:.2f}")
    #         print(
    #             f"@@@@@@ stochastic: z rotation {expert_policy[idx].run_config['object']} expert on {expert_policy[j].run_config['object']} env = {mean_z_rotation_expert:.2f} +/- {std_z_rotation_expert:.2f}")

    # # Evaluate the expert trained for individual-object manipulation, deterministic=True
    # for idx in range(num_experts):
    #     for j in range(num_experts):
    #         if render:
    #             os.makedirs(os.path.join(EXPERIMENT_ABS_PATH, "render_vids", "Expert"), exist_ok=True)
    #             vid_filename = os.path.join(EXPERIMENT_ABS_PATH, "render_vids", "Expert", str(expert_policy[idx].run_config['object']) + "_expert_on_" + str(expert_policy[j].run_config['object']) + "_env_render_vid.mp4")
    #             print("@@@@@@ ", vid_filename)
    #         alg = construct_policy_model.ALGS[expert_policy[idx].run_config["alg"]]
    #         mean_reward_expert, std_reward_expert, mean_z_rotation_expert, std_z_rotation_expert = evaluate_policy(
    #             model=alg.load(os.path.join(expert_policy[idx].run_dir, "models/best_model")), run_ID=run_IDs[j],
    #             n_eval_episodes=n_eval_episodes, deterministic=True, render=render, vid_filename=vid_filename if render else None)
    #         print(
    #             f"@@@@@@ deterministic: Mean reward {expert_policy[idx].run_config['object']} expert on {expert_policy[j].run_config['object']} env = {mean_reward_expert:.2f} +/- {std_reward_expert:.2f}")
    #         print(
    #             f"@@@@@@ deterministic: z rotation {expert_policy[idx].run_config['object']} expert on {expert_policy[j].run_config['object']} env = {mean_z_rotation_expert:.2f} +/- {std_z_rotation_expert:.2f}")

    run_reward, run_rotation = [], []
    for run_idx in range(num_runs):
        # Construct a student agent for behavior cloning
        print("@@@@@@ run_idx : ", run_idx)
        student = construct_policy_model.construct_policy_model(expert_policy[0].run_config["alg"],
                                                                expert_policy[0].run_config["policy"], expert_policy[0].env,
                                                                verbose=1)

        # # Evaluate the initial student policy
        # for idx in range(num_experts):
        #     mean_reward_student, std_reward_student, mean_z_rotation_student, std_z_rotation_student = evaluate_policy(
        #         model=student, run_ID=run_IDs[idx], n_eval_episodes=n_eval_episodes, deterministic=True, render=render)
        #     print(
        #         f"Mean reward student on {expert_policy[idx].run_config['object']} env = {mean_reward_student:.2f} +/- {std_reward_student:.2f}")
        #     print(
        #         f"z rotation student on {expert_policy[idx].run_config['object']} env = {mean_z_rotation_student:.2f} +/- {std_z_rotation_student:.2f}")

        # Prepare the dataset for pretraining
        print(f"num_experts = {num_experts}")
        train_expert_dataset, test_expert_dataset = [], []
        # for i in range(num_experts):
        #     train_size = int(0.8 * len(expert_policy[i].expert_dataset))
        #     test_size = len(expert_policy[i].expert_dataset) - train_size
        #     train_expert, test_expert = random_split(
        #         expert_policy[i].expert_dataset, [train_size, test_size]
        #     )
        #     # train_expert, test_expert = expert_policy[i].expert_dataset[:train_size], expert_policy[i].expert_dataset[train_size:]

        #     train_expert_dataset.append(train_expert)
        #     test_expert_dataset.append(test_expert)
        # for i in range(num_experts):
        #     print(f"$$$$$$ shape: {expert_policy[i].run_config['object']}, {expert_policy[i].expert_dataset[:][2]}")
        #     print(f"$$$$$$ len: {len(expert_policy[i].expert_dataset[0][0][0])}, {len(expert_policy[i].expert_dataset[0][1][0])}")

        # create scenario
        scenario = dataset_benchmark(
            train_datasets=[expert_policy[i].expert_dataset for i in range(num_experts)],
            test_datasets=test_expert_dataset,
            dataset_type=expert_policy[0].expert_dataset.dataset_type
        )

        print(f"scenario: {expert_policy[0].expert_dataset.dataset_type}")

        # Pretrain the student agent
        expert_objects = [expert_policy[idx].run_config['object'] for idx in range(num_experts)]
        agent = Pretrain_agent(student, scenario, strategy_name, num_experts, run_IDs, n_eval_episodes, render, expert_objects)
        run_reward.append(agent.bc_reward)
        run_rotation.append(agent.bc_rotation)

        os.makedirs(EXPERT_ABS_PATH, exist_ok=True)
        now = datetime.now()
        dt = now.strftime("%d-%m-%y_%H-%M-%S")
        student_save_path = os.path.join(EXPERT_ABS_PATH, f"student_{'_'.join(expert_objects)}_{agent.strategy_name}_{run_idx}_{dt}")
        student.save(student_save_path)
        print(f"Saving agent at: {student_save_path}")

    run_reward, run_rotation = np.array(run_reward), np.array(run_rotation)
    print("@@@@@@ run_reward.shape", run_reward.shape)

    if agent.strategy_name=="JointTraining":
        for n in range(num_experts):
            print(f"$$$$$$ {agent.strategy_name} {expert_policy[n].run_config['object']} reward : {np.mean(run_reward[:,:,n]):.2f} +/- {np.std(run_reward[:,:,n]):.2f}")
            print(f"$$$$$$ {agent.strategy_name} {expert_policy[n].run_config['object']} z_rotation : {np.mean(run_rotation[:,:,n]):.2f} +/- {np.std(run_rotation[:,:,n]):.2f}")

    else:
        for m in range(num_experts):
            for n in range(num_experts):
                print(f"$$$$$$ {agent.strategy_name} {expert_policy[n].run_config['object']} experience_{m} reward : {np.mean(run_reward[:,m,n]):.2f} +/- {np.std(run_reward[:,m,n]):.2f}")
                print(f"$$$$$$ {agent.strategy_name} {expert_policy[n].run_config['object']} experience_{m} z_rotation : {np.mean(run_rotation[:,m,n]):.2f} +/- {np.std(run_rotation[:,m,n]):.2f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Policy rollout run")
    parser.add_argument(
        "-e1",
        "--exp_name_1",
        help="Experiment name 1",
        required=True,
        default=None,
    )
    parser.add_argument(
        "-g1",
        "--run_group_name_1",
        help="Run-group name 1",
        required=True,
        default=None,
    )
    parser.add_argument(
        "-r1",
        "--run_name_1",
        help="Run Name 1",
        required=True,
        default=None,
    )
    parser.add_argument(
        "-e2",
        "--exp_name_2",
        help="Experiment name 2",
        required=False,
        default=None,
    )
    parser.add_argument(
        "-g2",
        "--run_group_name_2",
        help="Run-group name 2",
        required=False,
        default=None,
    )
    parser.add_argument(
        "-r2",
        "--run_name_2",
        help="Run Name 2",
        required=False,
        default=None,
    )
    parser.add_argument(
        "-e3",
        "--exp_name_3",
        help="Experiment name 3",
        required=False,
        default=None,
    )
    parser.add_argument(
        "-g3",
        "--run_group_name_3",
        help="Run-group name 3",
        required=False,
        default=None,
    )
    parser.add_argument(
        "-r3",
        "--run_name_3",
        help="Run Name 3",
        required=False,
        default=None,
    )
    parser.add_argument(
        "-e4",
        "--exp_name_4",
        help="Experiment name 4",
        required=False,
        default=None,
    )
    parser.add_argument(
        "-g4",
        "--run_group_name_4",
        help="Run-group name 4",
        required=False,
        default=None,
    )
    parser.add_argument(
        "-r4",
        "--run_name_4",
        help="Run Name 4",
        required=False,
        default=None,
    )
    parser.add_argument(
        "-e5",
        "--exp_name_5",
        help="Experiment name 5",
        required=False,
        default=None,
    )
    parser.add_argument(
        "-g5",
        "--run_group_name_5",
        help="Run-group name 5",
        required=False,
        default=None,
    )
    parser.add_argument(
        "-r5",
        "--run_name_5",
        help="Run Name 5",
        required=False,
        default=None,
    )
    parser.add_argument(
        "--exp_abs_path",
        help="Experiment directory absolute path",
        required=False,
        default=EXPERIMENT_ABS_PATH,
    )

    parser.add_argument(
        "-v",
        "--render",
        help="Render the env of one of the threads",
        action="store_true",
    )

    parser.add_argument(
        "--seed",
        help="Random seed",
        required=False,
        default=100
    )

    parser.add_argument(
        "-n",
        "--n_eval_episodes",
        help="number of episodes for evaluation",
        required=False,
        default=10
    )

    parser.add_argument(
        "-s",
        "--shuffle",
        help="shuffle the expert dataset",
        action="store_true",
    )

    parser.add_argument(
        "-stg",
        "--strategies",
        help="continual learning strategies for training",
        required=True,
        default="Naive"
    )

    parser.add_argument(
        "-nr",
        "--num_runs",
        help="num of runs",
        required=False,
        default=5
    )

    arg = parser.parse_args()
    seed = int(arg.seed) if arg.seed is not None else None
    n_eval_episodes = int(arg.n_eval_episodes) if arg.n_eval_episodes is not None else None
    num_runs = int(arg.num_runs) if arg.num_runs is not None else None
    run_IDs = []
    run_ID_1 = [arg.exp_name_1, arg.run_group_name_1, arg.run_name_1]
    run_IDs.append(run_ID_1)
    if arg.exp_name_2 and arg.run_group_name_2 and arg.run_name_2:
        run_ID_2 = [arg.exp_name_2, arg.run_group_name_2, arg.run_name_2]
        run_IDs.append(run_ID_2)
    if arg.exp_name_3 and arg.run_group_name_3 and arg.run_name_3:
        run_ID_3 = [arg.exp_name_3, arg.run_group_name_3, arg.run_name_3]
        run_IDs.append(run_ID_3)
    if arg.exp_name_4 and arg.run_group_name_4 and arg.run_name_4:
        run_ID_4 = [arg.exp_name_4, arg.run_group_name_4, arg.run_name_4]
        run_IDs.append(run_ID_4)
    if arg.exp_name_5 and arg.run_group_name_5 and arg.run_name_5:
        run_ID_5 = [arg.exp_name_5, arg.run_group_name_5, arg.run_name_5]
        run_IDs.append(run_ID_5)

    run(run_IDs=run_IDs, exp_abs_path=arg.exp_abs_path, render=arg.render, seed=seed, n_eval_episodes=n_eval_episodes, strategy_name=arg.strategies, num_runs=num_runs)
