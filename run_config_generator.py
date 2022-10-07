import os
import yaml
import argparse
from pathlib import Path
from somo_rl.utils import parse_config
from user_settings import EXPERIMENT_ABS_PATH

def run_config_generator(path, seed_start, seed_end):
	"""
	parameters:
	1. path: path to the run
	2. seed_start: starting value of the range of random seed
	3. seed_end: ending value of the range of random seed
	"""
	checkpoint_cb = {
		"save_freq": 2000
		}
	eval_cb = {
		"eval_freq": 2000,
	        "n_eval_episodes": 1
		}
	observation_flags = {
	        "box_pos": None,
	        "box_or": None,
	        "box_velocity": None,
	        "positions": 4,
	        "velocities": 4,
	        "tip_pos": None,
	        "angles": 4,
	        "curvatures": 5,
	        "applied_input_torques": None
	        }
	reward_flags = {
	        "z_rotation_step": 1000,
	        "x_rotation": -1,
	        "y_rotation": -1,
	        "position": -100
	        }
	policy_kwargs = {
	        "learning_rate": 0.003,
	        "buffer_size": 500000,
	        "learning_starts": 1000,
	        "batch_size": 25
	        }

	for seed in range(seed_start, seed_end+1):
		dir=os.path.join(path, f"seed_{seed}")
		Path(dir).mkdir(parents=True, exist_ok=True)
		run_config_dict = {
			"object": "cube",
			"action_time": 0.01,
			"alg": "SAC",
			"bullet_time_step": 0.0002,
			"checkpoint_cb": checkpoint_cb,
			"env_id": "InHandManipulationInverted-v0",
			"eval_cb": eval_cb,
			"failure_penalty_multiplier": 0,
			"max_episode_steps": 100,
			"max_torque_rate": 130,
			"num_threads": 1,
			"observation_flags": observation_flags,
			"policy": "SACMlpPolicy",
			"reward_flags": reward_flags,
			"seed": seed,
			"torque_multiplier": 65,
			"training_timesteps": 500000,
			"policy_kwargs": policy_kwargs,
			"invert_hand": True,
			"tensorboard_log": os.path.join(dir, "tensorboard_log")
			}
		run_config = os.path.join(dir, 'run_config.yaml')
		with open(run_config, 'w+', encoding='utf8') as outfile:
			yaml.dump(run_config_dict, outfile, default_flow_style=False)
			if not parse_config.validate_config(run_config):
				raise (Exception, "ERROR: Invalid run config")

if __name__ == "__main__":
	parser = argparse.ArgumentParser(description="Generating run config yaml")
	parser.add_argument(
		"-e",
		"--exp_name",
		help="Experiment name",
		required=True,
		default=None,
	)
	parser.add_argument(
		"-g",
		"--run_group_name",
		help="Run-group name",
		required=True,
		default=None,
	)
	parser.add_argument(
		"--exp_abs_path",
		help="Experiment directory absolute path",
		required=False,
		default=EXPERIMENT_ABS_PATH,
	)
	parser.add_argument(
		"-ss",
		"--seed_start",
		help="Starting value of the range of random seed",
		required=True,
		default=0
	)
	parser.add_argument(
		"-se",
		"--seed_end",
		help="Ending value of the range of random seed",
		required=True,
		default=19
	)
	arg = parser.parse_args()
	path = os.path.join(arg.exp_abs_path, arg.exp_name, arg.run_group_name)
	run_config_generator(path=path, seed_start=int(arg.seed_start), seed_end=int(arg.seed_end))
