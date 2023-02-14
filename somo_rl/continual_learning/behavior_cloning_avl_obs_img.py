import os
import sys
import argparse
import numpy as np
import torch as th
import pandas as pd
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
from avalanche.benchmarks.utils import AvalancheTensorDataset, AvalancheDataset, AvalancheConcatDataset, as_regression_dataset
from avalanche.benchmarks.generators import dataset_benchmark
from avalanche.training.supervised import Naive, EWC, JointTraining, Replay
from avalanche.training.plugins import ReplayPlugin, EWCPlugin
from avalanche.training.plugins.lr_scheduling import LRSchedulerPlugin
from avalanche.training.templates.supervised import SupervisedTemplate
from avalanche.benchmarks.utils.dataset_utils import ConstantSequence

path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.insert(0, path)

from user_settings import EXPERIMENT_ABS_PATH, EXPERT_ABS_PATH
from somo_rl.utils import construct_policy_model
from somo_rl.utils.load_env import load_env
from somo_rl.utils.evaluate_policy import evaluate_policy
from somo_rl.utils.load_run_config_file import load_run_config_file


class Policy_rollout:
    def __init__(self, exp_abs_path, run_ID, no_cuda=False, render=False, debug=False):
        self.expert_dataset = None
        self.train_expert_dataset = None
        self.test_expert_dataset = None
        self.run_dir, self.run_config = load_run_config_file(run_ID=run_ID, exp_abs_path=exp_abs_path)
        self.env = load_env(self.run_config, debug=debug)
        self.no_cuda = no_cuda
        self.render = render
        self.use_cuda = not self.no_cuda and th.cuda.is_available()
        self.device = th.device("cuda" if self.use_cuda else "cpu")
        self.load_rollouts()

    def load_rollouts(self):
        saved_data_path = os.path.join(self.run_dir, f"expert_data_{self.run_config['object']}.npz")
        print(f"@@@@@@@@ {saved_data_path} : {os.path.isfile(saved_data_path)}")
        if os.path.isfile(saved_data_path):
            expert_observations, expert_actions = np.load(saved_data_path)["expert_observations"], np.load(saved_data_path)["expert_actions"]
            print(f"Using saved data from : {saved_data_path} {expert_observations.shape} {expert_actions.shape}")
        else:
            models_dir = os.path.join(self.run_dir, f"models")
            alg = construct_policy_model.ALGS[self.run_config["alg"]]
            rl_model = alg.load(os.path.join(models_dir, "best_model"))
            list_observations = []
            list_actions = []
            if os.path.isfile(os.path.join(models_dir, "1best_obs_model_gru")):
                print(f"@@@@@@ Preparing data from the rl_agent and obs_img_gru_model from: {models_dir}")
                obs_img_gru_model = th.load(os.path.join(models_dir, "best_obs_model_gru"), map_location=th.device('cpu'))
                obs_img_gru_model.eval()
                for _ in tqdm(range(1000)):
                    obs = th.load(os.path.join(models_dir, "obs_tensor"), map_location=th.device('cpu'))
                    hidden = None
                    for _ in range(self.run_config["max_episode_steps"]):
                        obs = th.as_tensor(obs)
                        list_observations.append(obs)
                        action, _ = rl_model.predict(obs, deterministic=False)
                        action = th.as_tensor(action)
                        input_data = th.cat((obs, action), -1)
                        obs, hidden = obs_img_gru_model(input_data.view(1, 1, input_data.shape[0]), hidden) # (seq, batch, hidden)
                        obs = th.squeeze(obs).detach().numpy()
                        list_actions.append(action)
            elif os.path.isfile(os.path.join(models_dir, "1best_obs_model_mlp")):
                print(f"@@@@@@ Preparing data from the rl_agent and obs_img_mlp_model from: {models_dir}")
                obs_img_mlp_model = th.load(os.path.join(models_dir, "best_obs_model_mlp"), map_location=th.device('cpu'))
                obs_img_mlp_model.eval()
                for _ in tqdm(range(1000)):
                    obs = th.load(os.path.join(models_dir, "obs_tensor"), map_location=th.device('cpu'))   
                    for _ in range(self.run_config["max_episode_steps"]):
                        obs = th.tensor(obs)
                        list_observations.append(obs)
                        action, _ = rl_model.predict(obs, deterministic=False)
                        action = th.tensor(action)
                        input_data = th.cat((obs, action), -1)
                        obs = obs_img_mlp_model(input_data)
                        list_actions.append(action)
            else:
                print(f"@@@@@@ Preparing data from the rl_agent and env from: {models_dir}")
                for epi in tqdm(range(1000)):
                    obs = self.env.reset(run_render=False)#self.render)
                    for _ in range(self.run_config["max_episode_steps"]):
                        list_observations.append(th.tensor(obs))
                        action, _ = rl_model.predict(obs, deterministic=False)
                        obs, _, _, info = self.env.step(action)
                        list_actions.append(th.tensor(action))
                    print(f"@@@@@@ Episode_{epi}, z_rotation: {info['z_rotation_step']}")
                self.env.close()
            
            expert_observations, expert_actions = (th.stack(list_observations)).detach().numpy(), (th.stack(list_actions)).detach().numpy()
            np.savez_compressed(
                os.path.join(self.run_dir, f"expert_data_{self.run_config['object']}"),
                expert_observations=expert_observations,
                expert_actions=expert_actions,
            )

        #print(expert_observations.shape, expert_actions.shape)
        self.expert_dataset = as_regression_dataset(AvalancheTensorDataset(th.tensor(expert_observations), th.tensor(expert_actions)))
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
        if isinstance(self.student, (A2C, PPO)):
            action, _, _ = self.model(input)
        else:
            # SAC/TD3:
            action = self.model(input)
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
                 batch_size=1,
                 epochs=3,
                 scheduler_gamma=0.7,
                 learning_rate=1,
                 no_cuda=False,
                 seed=1,
                 test_batch_size=2,
                 ewc_lambda=3000,
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
        self.strategy_name = strategy_name

        self.use_cuda = not self.no_cuda and th.cuda.is_available()
        self.device = th.device("cuda" if self.use_cuda else "cpu")
        self.kwargs = {"num_workers": 1, "pin_memory": True} if self.use_cuda else {}

        self.criterion = nn.MSELoss()
        self.strategy = None
        self.run_IDs = run_IDs
        self.n_eval_episodes = n_eval_episodes

        # Extract initial policy
        self.model = mymodel(self.device, self.student)
        self.behavior_cloning()

    def behavior_cloning(self):
        # Define an Optimizer and a learning rate schedule.
        optimizer = optim.Adadelta(self.model.parameters(), lr=self.learning_rate)

        sched = LRSchedulerPlugin(
            StepLR(optimizer, step_size=1, gamma=self.scheduler_gamma)
        )
        # create strategy
        print(f"strategy_name : {self.strategy_name}")
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
            for idx in range(self.num_experts):
                mean_reward_student, std_reward_student, mean_z_rotation_student, std_z_rotation_student = evaluate_policy(
                    model=self.student, run_ID=self.run_IDs[idx], n_eval_episodes=self.n_eval_episodes,
                    deterministic=False,
                    render=self.render)
                print(
                    f"Mean reward student on {self.expert_objects[idx]} env = {mean_reward_student:.2f} +/- {std_reward_student:.2f}")
                print(
                    f"z rotation student on {self.expert_objects[idx]} env = {mean_z_rotation_student:.2f} +/- {std_z_rotation_student:.2f}")

        else:
            if self.strategy_name == "EWC":
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
            elif self.strategy_name == "Replay":
                self.strategy = Replay(
                    self.model,
                    optimizer,
                    mem_size=100000,
                    criterion=self.criterion,
                    train_mb_size=self.batch_size,
                    train_epochs=self.epochs,
                    device=self.device,
                    plugins=[sched],
                )
            elif self.strategy_name == "ReplayED":
                self.strategy = ReplayED(
                    self.model,
                    optimizer,
                    buffer_size=100000,
                    criterion=self.criterion,
                    train_mb_size=self.batch_size,
                    train_epochs=self.epochs,
                    device=self.device,
                    plugins=[sched],
                )
            elif self.strategy_name == "EWC_Replay":
                replay = ReplayPlugin(mem_size=100000)
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
                    buffer_size=100000,
                    criterion=self.criterion,
                    train_mb_size=self.batch_size,
                    train_epochs=self.epochs,
                    device=self.device,
                    plugins=[sched, ewc],
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
                print("Start training on experience ", experience.current_experience)
                self.strategy.train(experience)
                print("End training on experience", experience.current_experience)

                # Implant the trained policy network back into the RL student agent
                self.student.policy = self.strategy.model.model

                # evaluate policy
                for idx in range(self.num_experts):
                    mean_reward_student, std_reward_student, mean_z_rotation_student, std_z_rotation_student = evaluate_policy(
                        model=self.student, run_ID=self.run_IDs[idx], n_eval_episodes=self.n_eval_episodes, deterministic=False,
                        render=self.render)
                    print(
                        f"Mean reward student on {self.expert_objects[idx]} env = {mean_reward_student:.2f} +/- {std_reward_student:.2f}")
                    print(
                        f"z rotation student on {self.expert_objects[idx]} env = {mean_z_rotation_student:.2f} +/- {std_z_rotation_student:.2f}")


def run(run_IDs, exp_abs_path=EXPERIMENT_ABS_PATH, seed=100, render=False, debug=False, n_eval_episodes=10, strategy_name="Naive"):
    print(f"seed = {seed}, n_eval_episodes = {n_eval_episodes}")
    set_random_seed(seed)
    num_experts = len(run_IDs)
    expert_policy = []

    # Load expert agents
    for idx in range(num_experts):
        expert_policy.append(Policy_rollout(exp_abs_path=exp_abs_path, run_ID=run_IDs[idx], render=render, debug=debug))

    # if num_experts > 1:
    #     # Evaluate the expert trained for multi-objects manipulation
    #     # baseline_runID = [run_IDs[0][0], "PPO_multi_objects", run_IDs[0][-1]]
    #     baseline_run_group_name = (list(run_IDs[0][1].split("_")))[0] + "_" + "_".join([(list(run_IDs[i][1].split("_")))[-1] for i in range(num_experts)])
    #     baseline_runID = [run_IDs[0][0], baseline_run_group_name, run_IDs[0][-1]]
    #     try:
    #         baseline_run_dir, baseline_run_config = load_run_config_file(baseline_runID)
    #         for i in range(num_experts):
    #             alg = construct_policy_model.ALGS[expert_policy[i].run_config["alg"]]
    #             mean_reward_expert, std_reward_expert, mean_z_rotation_expert, std_z_rotation_expert = evaluate_policy(
    #                 model=alg.load(os.path.join(baseline_run_dir, "models/best_model")), run_ID=run_IDs[i],
    #                 n_eval_episodes=n_eval_episodes, deterministic=False, render=render)
    #             print(
    #                 f"Mean reward {baseline_run_config['object']} expert on {expert_policy[i].run_config['object']} env = {mean_reward_expert:.2f} +/- {std_reward_expert:.2f}")
    #             print(
    #                 f"z rotation {baseline_run_config['object']} expert on {expert_policy[i].run_config['object']} env = {mean_z_rotation_expert:.2f} +/- {std_z_rotation_expert:.2f}")
    #     except:
    #         print(f"Exception : The expert trained for multi-objects manipulation does not exist : {baseline_runID}")

    # # Evaluate the expert trained for individual-object manipulation
    # for idx in range(num_experts):
    #     for j in range(num_experts):
    #         alg = construct_policy_model.ALGS[expert_policy[idx].run_config["alg"]]
    #         mean_reward_expert, std_reward_expert, mean_z_rotation_expert, std_z_rotation_expert = evaluate_policy(
    #             model=alg.load(os.path.join(expert_policy[idx].run_dir, "models/best_model")), run_ID=run_IDs[j],
    #             n_eval_episodes=n_eval_episodes, deterministic=False, render=render)
    #         print(
    #             f"Mean reward {expert_policy[idx].run_config['object']} expert on {expert_policy[j].run_config['object']} env = {mean_reward_expert:.2f} +/- {std_reward_expert:.2f}")
    #         print(
    #             f"z rotation {expert_policy[idx].run_config['object']} expert on {expert_policy[j].run_config['object']} env = {mean_z_rotation_expert:.2f} +/- {std_z_rotation_expert:.2f}")

    # Construct a student agent for behavior cloning
    student = construct_policy_model.construct_policy_model(expert_policy[0].run_config["alg"],
                                                            expert_policy[0].run_config["policy"], expert_policy[0].env,
                                                            verbose=1)

    # # Evaluate the initial student policy
    # for idx in range(num_experts):
    #     mean_reward_student, std_reward_student, mean_z_rotation_student, std_z_rotation_student = evaluate_policy(
    #         model=student, run_ID=run_IDs[idx], n_eval_episodes=n_eval_episodes, deterministic=False, render=render)
    #     print(
    #         f"Mean reward student on {expert_policy[idx].run_config['object']} env = {mean_reward_student:.2f} +/- {std_reward_student:.2f}")
    #     print(
    #         f"z rotation student on {expert_policy[idx].run_config['object']} env = {mean_z_rotation_student:.2f} +/- {std_z_rotation_student:.2f}")

    # Prepare the dataset for pretraining
    print(f"num_experts = {num_experts}")
    train_expert_dataset, test_expert_dataset = [], []
    for i in range(num_experts):
        train_size = int(0.8 * len(expert_policy[i].expert_dataset))
        test_size = len(expert_policy[i].expert_dataset) - train_size
        train_expert, test_expert = random_split(
            expert_policy[i].expert_dataset, [train_size, test_size]
        )
        train_expert_dataset.append(train_expert)
        test_expert_dataset.append(test_expert)

    # create scenario
    scenario = dataset_benchmark(
        train_datasets=[expert_policy[i].expert_dataset for i in range(num_experts)],
        test_datasets=test_expert_dataset,
        dataset_type=expert_policy[0].expert_dataset.dataset_type
    )

    print(f"scenario: {expert_policy[0].expert_dataset.dataset_type}")

    # Pretrain the student agent
    expert_objects = [expert_policy[idx].run_config['object'] for idx in range(num_experts)]
    Pretrain_agent(student, scenario, strategy_name, num_experts, run_IDs, n_eval_episodes, render, expert_objects)
    os.makedirs(EXPERT_ABS_PATH, exist_ok=True)
    student_save_path = os.path.join(EXPERT_ABS_PATH, f"student_{'_'.join(expert_objects)}")
    student.save(student_save_path)
    print(f"Saving agent at: {student_save_path}")


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

    arg = parser.parse_args()
    seed = int(arg.seed) if arg.seed is not None else None
    n_eval_episodes = int(arg.n_eval_episodes) if arg.n_eval_episodes is not None else None
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

    run(run_IDs=run_IDs, exp_abs_path=arg.exp_abs_path, render=arg.render, seed=seed, n_eval_episodes=n_eval_episodes, strategy_name=arg.strategies)
