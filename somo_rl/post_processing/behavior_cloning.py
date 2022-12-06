import os
import sys
import argparse
import numpy as np
import torch as th
import pandas as pd
from torch.optim.lr_scheduler import StepLR
from torch.utils.data.dataset import Dataset, random_split
from torch import nn
from torch import optim
from stable_baselines3 import PPO, A2C
from stable_baselines3.common.utils import set_random_seed

path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.insert(0, path)

from user_settings import EXPERIMENT_ABS_PATH, EXPERT_ABS_PATH
from somo_rl.utils import construct_policy_model
from somo_rl.utils.load_env import load_env
from somo_rl.utils.evaluate_policy import evaluate_policy
from somo_rl.utils.load_run_config_file import load_run_config_file


def unison_shuffled_copies(expert_dataset):
    assert len(expert_dataset.observations) == len(expert_dataset.actions)
    p = np.random.permutation(len(expert_dataset.observations))

    return ExpertDataSet(expert_dataset.observations[p], expert_dataset.actions[p])

class ExpertDataSet(Dataset):
    def __init__(self, expert_observations, expert_actions):
        self.observations = expert_observations
        self.actions = expert_actions

    def __getitem__(self, index):
        return self.observations[index], self.actions[index]

    def __len__(self):
        return len(self.observations)


class Policy_rollout:
    def __init__(self, exp_abs_path, run_ID, render=False, debug=False):
        self.expert_dataset = None
        self.train_expert_dataset = None
        self.test_expert_dataset = None
        self.run_dir, self.run_config = load_run_config_file(run_ID=run_ID, exp_abs_path=exp_abs_path)
        self.env = load_env(self.run_config, render=render, debug=debug)
        self.load_rollouts()

    def load_rollouts(self):
        saved_data_path = os.path.join(self.run_dir, f"expert_data_{self.run_config['object']}.npz")
        if os.path.isfile(saved_data_path):
            expert_observations, expert_actions = np.load(saved_data_path)["expert_observations"], np.load(saved_data_path)["expert_actions"]
            print(f"Using saved data from : {saved_data_path}")
        else:
            print(f"Preparing data from: {os.path.join(self.run_dir, 'results/processed_data/')}")
            actions_df_list = []
            observations_df_list = []
            for run in os.scandir(os.path.join(self.run_dir, "results/processed_data/")):
                if os.path.isdir(run):
                    actions_df_list.append(pd.read_pickle(os.path.join(run.path, 'actions.pkl')).iloc[:, :8])
                    observations_df_list.append(pd.read_pickle(os.path.join(run.path, 'observations.pkl')))

            actions_df = pd.concat(actions_df_list)
            observations_df = pd.concat(observations_df_list)

            expert_observations, expert_actions = observations_df.to_numpy(dtype=np.double), actions_df.to_numpy(dtype=np.double)

            np.savez_compressed(
                os.path.join(self.run_dir, f"expert_data_{self.run_config['object']}"),
                expert_observations=expert_observations,
                expert_actions=expert_actions,
            )

        self.expert_dataset = ExpertDataSet(expert_observations, expert_actions)
        print(f"size of expert_dataset_{self.run_config['object']}: {len(self.expert_dataset)}")


class Pretrain_agent:
    def __init__(self,
                 student,
                 train_expert_dataset,
                 test_expert_dataset,
                 batch_size=64,
                 epochs=1,
                 scheduler_gamma=0.7,
                 learning_rate=1.0,
                 log_interval=100,
                 no_cuda=True,
                 seed=1,
                 test_batch_size=64,
                 ):
        self.student = student
        self.train_expert_dataset = train_expert_dataset
        self.test_expert_dataset = test_expert_dataset
        self.batch_size = batch_size
        self.epochs = epochs
        self.scheduler_gamma = scheduler_gamma
        self.learning_rate = learning_rate
        self.log_interval = log_interval
        self.no_cuda = no_cuda
        self.seed = seed
        self.test_batch_size = test_batch_size

        self.use_cuda = not self.no_cuda and th.cuda.is_available()
        self.device = th.device("cuda" if self.use_cuda else "cpu")
        self.kwargs = {"num_workers": 1, "pin_memory": True} if self.use_cuda else {}

        self.criterion = nn.MSELoss()

        # Extract initial policy
        self.model = student.policy.to(self.device)
        self.behavior_cloning()

    def train(self, epoch, train_loader, optimizer):
        self.model.train()

        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(self.device), target.to(self.device)
            optimizer.zero_grad()

            # A2C/PPO policy outputs actions, values, log_prob
            # SAC/TD3 policy outputs actions only
            if isinstance(self.student, (A2C, PPO)):
                action, _, _ = self.model(data)
            else:
                # SAC/TD3:
                action = self.model(data)
            action_prediction = action.double()

            loss = self.criterion(action_prediction, target)
            loss.backward()
            optimizer.step()
            if batch_idx % self.log_interval == 0:
                print(
                    "Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}".format(
                        epoch,
                        batch_idx * len(data),
                        len(train_loader.dataset),
                        100.0 * batch_idx / len(train_loader),
                        loss.item(),
                    )
                )

    def test(self, test_loader):
        self.model.eval()
        test_loss = 0
        with th.no_grad():
            for data, target in test_loader:
                data, target = data.to(self.device), target.to(self.device)

                # A2C/PPO policy outputs actions, values, log_prob
                # SAC/TD3 policy outputs actions only
                if isinstance(self.student, (A2C, PPO)):
                    action, _, _ = self.model(data)
                else:
                    # SAC/TD3:
                    action = self.model(data)
                action_prediction = action.double()

                test_loss = self.criterion(action_prediction, target)
        test_loss /= len(test_loader.dataset)
        print(f"Test set: Average loss: {test_loss:.6f}")

    def behavior_cloning(self):
        # Define an Optimizer and a learning rate schedule.
        optimizer = optim.Adadelta(self.model.parameters(), lr=self.learning_rate)
        scheduler = StepLR(optimizer, step_size=1, gamma=self.scheduler_gamma)

        # use PyTorch `DataLoader` to load previously created `ExpertDataset` for training and testing
        train_loader = th.utils.data.DataLoader(
            dataset=self.train_expert_dataset, batch_size=self.batch_size, shuffle=False, **self.kwargs
        )
        test_loader = th.utils.data.DataLoader(
            dataset=self.test_expert_dataset, batch_size=self.test_batch_size, shuffle=False, **self.kwargs,
        )

        # Now we are finally ready to train the policy model.
        for epoch in range(1, self.epochs + 1):
            self.train(epoch, train_loader, optimizer)
            self.test(test_loader)
            scheduler.step()

        # Implant the trained policy network back into the RL student agent
        self.student.policy = self.model


def run(run_IDs, exp_abs_path=EXPERIMENT_ABS_PATH, seed=100, render=False, debug=False, n_eval_episodes=10, shuffle=True):
    print(f"seed = {seed}, n_eval_episodes = {n_eval_episodes}")
    set_random_seed(seed)
    num_experts = len(run_IDs)
    expert_policy = []

    # Load expert agents
    for idx in range(num_experts):
        expert_policy.append(Policy_rollout(exp_abs_path=exp_abs_path, run_ID=run_IDs[idx], render=render, debug=debug))

    if num_experts > 1:
        # Evaluate the expert trained for multi-objects manipulation
        # baseline_runID = [run_IDs[0][0], "PPO_multi_objects", run_IDs[0][-1]]
        baseline_run_group_name = (list(run_IDs[0][1].split("_")))[0] + "_" + "_".join([(list(run_IDs[i][1].split("_")))[-1] for i in range(num_experts)])
        baseline_runID = [run_IDs[0][0], baseline_run_group_name, run_IDs[0][-1]]
        try:
            baseline_run_dir, baseline_run_config = load_run_config_file(baseline_runID)
            for i in range(num_experts):
                alg = construct_policy_model.ALGS[expert_policy[i].run_config["alg"]]
                mean_reward_expert, std_reward_expert, mean_z_rotation_expert, std_z_rotation_expert = evaluate_policy(
                    model=alg.load(os.path.join(baseline_run_dir, "models/best_model")), run_ID=run_IDs[i],
                    n_eval_episodes=n_eval_episodes, deterministic=False, render=render)
                print(
                    f"Mean reward {baseline_run_config['object']} expert on {expert_policy[i].run_config['object']} env = {mean_reward_expert:.2f} +/- {std_reward_expert:.2f}")
                print(
                    f"z rotation {baseline_run_config['object']} expert on {expert_policy[i].run_config['object']} env = {mean_z_rotation_expert:.2f} +/- {std_z_rotation_expert:.2f}")
        except:
            print(f"Exception : The expert trained for multi-objects manipulation does not exist : {baseline_runID}")

    # Evaluate the expert trained for individual-object manipulation
    for idx in range(num_experts):
        for j in range(num_experts):
            alg = construct_policy_model.ALGS[expert_policy[idx].run_config["alg"]]
            mean_reward_expert, std_reward_expert, mean_z_rotation_expert, std_z_rotation_expert = evaluate_policy(
                model=alg.load(os.path.join(expert_policy[idx].run_dir, "models/best_model")), run_ID=run_IDs[j],
                n_eval_episodes=n_eval_episodes, deterministic=False, render=render)
            print(
                f"Mean reward {expert_policy[idx].run_config['object']} expert on {expert_policy[j].run_config['object']} env = {mean_reward_expert:.2f} +/- {std_reward_expert:.2f}")
            print(
                f"z rotation {expert_policy[idx].run_config['object']} expert on {expert_policy[j].run_config['object']} env = {mean_z_rotation_expert:.2f} +/- {std_z_rotation_expert:.2f}")

    # Construct a student agent for behavior cloning
    student = construct_policy_model.construct_policy_model(expert_policy[0].run_config["alg"],
                                                            expert_policy[0].run_config["policy"], expert_policy[0].env,
                                                            verbose=1)

    # Evaluate the initial student policy
    for idx in range(num_experts):
        mean_reward_student, std_reward_student, mean_z_rotation_student, std_z_rotation_student = evaluate_policy(
            model=student, run_ID=run_IDs[idx], n_eval_episodes=n_eval_episodes, deterministic=False, render=render)
        print(
            f"Mean reward student on {expert_policy[idx].run_config['object']} env = {mean_reward_student:.2f} +/- {std_reward_student:.2f}")
        print(
            f"z rotation student on {expert_policy[idx].run_config['object']} env = {mean_z_rotation_student:.2f} +/- {std_z_rotation_student:.2f}")

    if shuffle:
        # Prepare the dataset for pretraining
        print(f"num_experts = {num_experts}")
        if num_experts > 1:
            os.makedirs(EXPERT_ABS_PATH, exist_ok=True)
            saved_pretraining_data_path = os.path.join(EXPERT_ABS_PATH,
                                                       f"pretraining_data_{'_'.join([expert_policy[idx].run_config['object'] for idx in range(num_experts)])}.npz")
            if os.path.isfile(saved_pretraining_data_path):
                expert_observations_dataset, expert_actions_dataset = np.load(saved_pretraining_data_path)[
                                                                          "expert_observations"], \
                                                                      np.load(saved_pretraining_data_path)["expert_actions"]
                print(f"Using saved data from : {saved_pretraining_data_path}")
            else:
                print(f"Preparing dataset for pretraining from each expert dataset")
                expert_actions_dataset = np.concatenate(
                    [expert_policy[i].expert_dataset.actions for i in range(num_experts)], axis=0, dtype=np.double)
                expert_observations_dataset = np.concatenate(
                    [expert_policy[i].expert_dataset.observations for i in range(num_experts)], axis=0, dtype=np.double)

                np.savez_compressed(
                    os.path.join(EXPERT_ABS_PATH,
                                 f"pretraining_data_{'_'.join([expert_policy[idx].run_config['object'] for idx in range(num_experts)])}"),
                    expert_observations=expert_observations_dataset,
                    expert_actions=expert_actions_dataset,
                )
            expert_dataset = ExpertDataSet(expert_observations_dataset, expert_actions_dataset)
        else:
            expert_dataset = expert_policy[0].expert_dataset

        # Shuffle the pretraining dataset
        print("Shuffling the pretraining dataset")
        expert_dataset = unison_shuffled_copies(expert_dataset)

        train_size = int(0.8 * len(expert_dataset))
        test_size = len(expert_dataset) - train_size

        train_expert_dataset, test_expert_dataset = random_split(
            expert_dataset, [train_size, test_size]
        )
        # train_expert_dataset = test_expert_dataset = expert_dataset

        print(f"train dataset for pretraining: {len(train_expert_dataset)}")
        print(f"test dataset for pretraining: {len(test_expert_dataset)}")

        # Pretrain the student agent
        pretrain_agent = Pretrain_agent(student, train_expert_dataset, test_expert_dataset)
        student_save_path = os.path.join(EXPERT_ABS_PATH,
                                         f"student_{'_'.join([expert_policy[idx].run_config['object'] for idx in range(num_experts)])}")
        student.save(student_save_path)
        print(f"Saving agent at: {student_save_path}")

        # Evaluate the trained student policy
        for idx in range(num_experts):
            mean_reward_student, std_reward_student, mean_z_rotation_student, std_z_rotation_student = evaluate_policy(
                model=pretrain_agent.student, run_ID=run_IDs[idx], n_eval_episodes=n_eval_episodes, deterministic=False,
                render=render)
            print(
                f"Mean reward student on {expert_policy[idx].run_config['object']} env = {mean_reward_student:.2f} +/- {std_reward_student:.2f}")
            print(
                f"z rotation student on {expert_policy[idx].run_config['object']} env = {mean_z_rotation_student:.2f} +/- {std_z_rotation_student:.2f}")
    else:
        print(f"num_experts = {num_experts}")
        for idx in range(num_experts):
            os.makedirs(EXPERT_ABS_PATH, exist_ok=True)
            expert_dataset = expert_policy[idx].expert_dataset

            train_size = int(0.8 * len(expert_dataset))
            test_size = len(expert_dataset) - train_size

            train_expert_dataset, test_expert_dataset = random_split(
                expert_dataset, [train_size, test_size]
            )
            # train_expert_dataset = test_expert_dataset = expert_dataset

            print(
                f"{expert_policy[idx].run_config['object']} train dataset for pretraining: {len(train_expert_dataset)}")
            print(f"{expert_policy[idx].run_config['object']} test dataset for pretraining: {len(test_expert_dataset)}")

            # Pretrain the student agent
            pretrain_agent = Pretrain_agent(student, train_expert_dataset, test_expert_dataset)
            student_save_path = os.path.join(EXPERT_ABS_PATH,
                                             f"student_{'_'.join([expert_policy[i].run_config['object'] for i in range(num_experts)])}_{idx}")
            student.save(student_save_path)
            print(f"Saving student agent {idx} at: {student_save_path}")

            # Evaluate the trained student policy
            for i in range(num_experts):
                mean_reward_student, std_reward_student, mean_z_rotation_student, std_z_rotation_student = evaluate_policy(
                    model=pretrain_agent.student, run_ID=run_IDs[i], n_eval_episodes=n_eval_episodes,
                    deterministic=False,
                    render=render)
                print(
                    f"Mean reward student on {expert_policy[i].run_config['object']} env = {mean_reward_student:.2f} +/- {std_reward_student:.2f}")
                print(
                    f"z rotation student on {expert_policy[i].run_config['object']} env = {mean_z_rotation_student:.2f} +/- {std_z_rotation_student:.2f}")


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

    run(run_IDs=run_IDs, exp_abs_path=arg.exp_abs_path, render=arg.render, seed=seed, n_eval_episodes=n_eval_episodes, shuffle=arg.shuffle)
