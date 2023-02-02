import os
import gym
import sys
import torch as th
from torch import nn
from torch import optim
import warnings
import numpy as np

from typing import Any, Dict, Optional, Union
from stable_baselines3.common.results_plotter import load_results, ts2xy
from stable_baselines3.common.callbacks import BaseCallback, EventCallback
from stable_baselines3.common.vec_env import DummyVecEnv, VecEnv, sync_envs_normalization

path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.insert(0, path)

from somo_rl.utils.evaluate_policy import evaluate_policy


class Obs_Img_NN(nn.Module):
    def __init__(self):
        super(Obs_Img_NN, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(268+64, 268),
            nn.Tanh(),
            nn.Linear(268, 268),
            nn.Tanh(),
            nn.Linear(268, 268),
        )

    def forward(self, x):
        output = self.model(x)
        return output


class Multi_Obj_EvalCallback(EventCallback):
    """
    Callback for evaluating an agent.

    .. warning::

      When using multiple environments, each call to  ``env.step()``
      will effectively correspond to ``n_envs`` steps.
      To account for that, you can use ``eval_freq = max(eval_freq // n_envs, 1)``

    :param run_ID: The list of [exp_name, run_group_name, run_name], the environment used for initialization
    :param eval_env: The environment used for initialization
    :param callback_on_new_best: Callback to trigger
        when there is a new best model according to the ``mean_reward``
    :param callback_after_eval: Callback to trigger after every evaluation
    :param n_eval_episodes: The number of episodes to test the agent
    :param eval_freq: Evaluate the agent every ``eval_freq`` call of the callback.
    :param log_path: Path to a folder where the evaluations (``evaluations.npz``)
        will be saved. It will be updated at each evaluation.
    :param best_model_save_path: Path to a folder where the best model
        according to performance on the eval env will be saved.
    :param deterministic: Whether the evaluation should
        use a stochastic or deterministic actions.
    :param render: Whether to render or not the environment during evaluation
    :param verbose: Verbosity level: 0 for no output, 1 for indicating information about evaluation results
    """

    def __init__(
        self,
        run_ID: Optional[list] = None,
        eval_env: Union[gym.Env, VecEnv] = None,
        callback_on_new_best: Optional[BaseCallback] = None,
        callback_after_eval: Optional[BaseCallback] = None,
        n_eval_episodes: int = 5,
        eval_freq: int = 10000,
        log_path: Optional[str] = None,
        best_model_save_path: Optional[str] = None,
        deterministic: bool = True,
        render: bool = False,
        verbose: int = 1,
    ):
        super().__init__(callback_after_eval, verbose=verbose)

        self.callback_on_new_best = callback_on_new_best
        if self.callback_on_new_best is not None:
            # Give access to the parent
            self.callback_on_new_best.parent = self

        self.run_ID = run_ID
        self.n_eval_episodes = n_eval_episodes
        self.eval_freq = eval_freq
        self.best_mean_reward = -np.inf
        self.last_mean_reward = -np.inf
        self.deterministic = deterministic
        self.render = render

        # Convert to VecEnv for consistency
        if not isinstance(eval_env, VecEnv):
            eval_env = DummyVecEnv([lambda: eval_env])

        self.eval_env = eval_env
        self.best_model_save_path = best_model_save_path
        # Logs will be written in ``evaluations.npz``
        if log_path is not None:
            log_path = os.path.join(log_path, "evaluations")
        self.log_path = log_path
        self.evaluations_results = []
        self.evaluations_z_rotation = []
        self.evaluations_timesteps = []

    def _init_callback(self) -> None:
        # Does not work in some corner cases, where the wrapper is not the same
        if not isinstance(self.training_env, type(self.eval_env)):
            warnings.warn("Training and eval env are not of the same type" f"{self.training_env} != {self.eval_env}")

        # Create folders if needed
        if self.best_model_save_path is not None:
            os.makedirs(self.best_model_save_path, exist_ok=True)
        if self.log_path is not None:
            os.makedirs(os.path.dirname(self.log_path), exist_ok=True)

        # Init callback called on new best model
        if self.callback_on_new_best is not None:
            self.callback_on_new_best.init_callback(self.model)

    def _on_step(self) -> bool:

        continue_training = True

        if self.eval_freq > 0 and self.n_calls % self.eval_freq == 0:

            # Sync training and eval env if there is VecNormalize
            if self.model.get_vec_normalize_env() is not None:
                try:
                    sync_envs_normalization(self.training_env, self.eval_env)
                except AttributeError as e:
                    raise AssertionError(
                        "Training and eval env are not wrapped the same way, "
                        "see https://stable-baselines3.readthedocs.io/en/master/guide/callbacks.html#evalcallback "
                        "and warning above."
                    ) from e

            mean_reward, std_reward, mean_z_rotation, std_z_rotation = evaluate_policy(
                model=self.model,
                run_ID=self.run_ID,
                env=self.eval_env,
                n_eval_episodes=self.n_eval_episodes,
                deterministic=self.deterministic,
                render=self.render,
            )

            if self.log_path is not None:
                self.evaluations_timesteps.append(self.num_timesteps)
                self.evaluations_results.append(mean_reward)
                self.evaluations_z_rotation.append(mean_z_rotation)

                kwargs = {}

                np.savez(
                    self.log_path,
                    timesteps=self.evaluations_timesteps,
                    results=self.evaluations_results,
                    z_rotation=self.evaluations_z_rotation,
                    **kwargs,
                )

            self.last_mean_reward = mean_reward

            if self.verbose >= 1:
                print(f"Eval num_timesteps={self.num_timesteps}, " f"episode_reward={mean_reward:.2f} +/- {std_reward:.2f}", f"episode_z_rotation={mean_z_rotation:.2f} +/- {std_z_rotation:.2f}")
            # Add to current Logger
            self.logger.record("eval/mean_reward", float(mean_reward))
            self.logger.record("eval/mean_z_rotation", float(mean_z_rotation))

            # Dump log so the evaluation results are printed with the correct timestep
            self.logger.record("time/total_timesteps", self.num_timesteps, exclude="tensorboard")
            self.logger.dump(self.num_timesteps)

            if mean_reward > self.best_mean_reward:
                if self.verbose >= 1:
                    print("New best mean reward!")
                if self.best_model_save_path is not None:
                    self.model.save(os.path.join(self.best_model_save_path, "best_model"))
                self.best_mean_reward = mean_reward
                # Trigger callback on new best model, if needed
                if self.callback_on_new_best is not None:
                    continue_training = self.callback_on_new_best.on_step()

            # Trigger callback after every evaluation, if needed
            if self.callback is not None:
                continue_training = continue_training and self._on_event()

        return continue_training

    def update_child_locals(self, locals_: Dict[str, Any]) -> None:
        """
        Update the references to the local variables.

        :param locals_: the local variables during rollout collection
        """
        if self.callback:
            self.callback.update_locals(locals_)


class SaveOnBestTrainingRewardCallback(BaseCallback):
    """
    Callback for saving a model (the check is done every ``check_freq`` steps)
    based on the training reward (in practice, we recommend using ``EvalCallback``).

    :param check_freq:
    :param monitoring_dir: Path to the folder where the monitoring data wes saved.
    :param models_dir: Path to the folder where the model will be saved.
    :param verbose: Verbosity level: 0 for no output, 1 for info messages, 2 for debug messages
    """
    def __init__(self, check_freq: int, monitoring_dir: str, models_dir: str, verbose: int = 0):
        super(SaveOnBestTrainingRewardCallback, self).__init__(verbose)
        self.check_freq = check_freq
        self.monitoring_dir = monitoring_dir
        self.models_dir = os.path.join(models_dir, "best_model")
        self.best_mean_reward = -np.inf

    def _on_step(self) -> bool:
        if self.n_calls % self.check_freq == 0:

            # Retrieve training reward
            x, y = ts2xy(load_results(self.monitoring_dir), "timesteps")
            if len(x) > 0:
                y = y.reshape(-1, self.locals['env'].num_envs).sum(1)
                # Mean training reward over the last 100 episodes
                mean_reward = np.mean(y[-100:])
                self.logger.record("mean_reward", mean_reward)
                if self.verbose >= 1:
                    print(f"Num timesteps: {self.num_timesteps}")
                    print(f"Best mean reward: {self.best_mean_reward:.2f} - Last mean reward per episode: {mean_reward:.2f}")

                # New best model, you could save the agent here
                if mean_reward > self.best_mean_reward:
                    self.best_mean_reward = mean_reward
                    # Example for saving best model
                    if self.verbose >= 1:
                        print(f"Saving new best model to {self.models_dir}")
                    self.model.save(self.models_dir)

        return True


class TensorboardCallback(BaseCallback):
    """
    Custom callback for plotting z rotation value in tensorboard.
    """

    def __init__(self, eval_freq: int, verbose=0):
        self.eval_freq = eval_freq
        super(TensorboardCallback, self).__init__(verbose)

    def _on_step(self) -> bool:
        # Log z rotation value
        if self.n_calls % self.eval_freq == 0:
            info = self.locals["infos"][0]
            if "z_rotation_step" in info:
                z_rotation = info["z_rotation_step"]
            else:
                z_rotation = info["z_rotation"]
            self.logger.record("z_rotation", z_rotation)
        return True


class Observation_imagination_Callback(BaseCallback):
    """
    Train a model that takes the (obs, action) as input and output the new obs.
    """

    def __init__(self, models_dir: str, save_freq: int, device: Union[th.device, str] = "mps", verbose=0):
        super(Observation_imagination_Callback, self).__init__(verbose)
        self.models_dir = os.path.join(models_dir, "obs_model")
        self.save_freq = save_freq
        self.obs_tensor_path = os.path.join(models_dir, "obs_tensor")
        self.criterion = nn.MSELoss()
        self.learning_rate = 0.0001
        self.epochs = 10
        self.device = device
        self.obs_img_model = Obs_Img_NN().to(self.device)
        self.optimizer = optim.Adadelta(self.obs_img_model.parameters(), lr=self.learning_rate)

    def train(self, train_loader):

        for batch_idx, (data, target) in enumerate(train_loader):
            data, obs_target = data.to(self.device), target.to(self.device)
            self.optimizer.zero_grad()

            observation = self.model(data)
            observation_prediction = observation.double()

            loss = self.criterion(observation_prediction, obs_target)
            loss.backward()
            self.optimizer.step()
        print(f"Loss: {loss.item():.6f}")

    def _on_step(self) -> bool:
        print(f"@@@@@@ n_steps: {self.locals['n_steps']}, obs_tensor: {self.locals['obs_tensor'].shape}, new_obs: {self.locals['new_obs'].shape}, actions: {self.locals['actions'].shape}")
        num_envs = self.locals['actions'].shape[0]
        if self.locals['n_steps'] == 0:
            print(f"@@@@@@ SAVE obs_tensor to {self.obs_tensor_path}")
            th.save(self.locals['obs_tensor'].tolist()[0], self.obs_tensor_path)

        input_data = th.cat((self.locals['obs_tensor'], th.from_numpy(self.locals['actions'])), -1)
        target_data = th.from_numpy(self.locals['new_obs'])
        print(f"@@@@@@ input shape: {input_data.shape}, target shape {target_data.shape}")

        dataset = th.utils.data.TensorDataset(input_data, target_data)
        train_loader = th.utils.data.DataLoader(dataset=dataset, batch_size=num_envs, shuffle=False)

        for epoch in range(1, self.epochs + 1):
            self.train(train_loader)
            print(f"Train Epoch: {epoch}")
 
        if self.n_calls % self.save_freq == 0:
            self.obs_img_model.save(self.obs_img_model)

        return True
