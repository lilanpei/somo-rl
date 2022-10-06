import os
import yaml
from pathlib import Path

def run_config_generator(path):
	"""
	parameters:
	1.path: path to the run
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

	for seed in range(20):
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
		with open(os.path.join(dir, 'run_config.yaml'), 'w+', encoding='utf8') as outfile:
			yaml.dump(run_config_dict, outfile, default_flow_style=False)

if __name__ == "__main__":
	path = "/home/lanpei/thesis_project/somo-rl/test_exp/InHandManipulationInverted-v0/SAC_test"
	run_config_generator(path)
