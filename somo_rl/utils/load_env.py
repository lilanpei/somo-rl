import os
import gym
import sys

path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.insert(0, path)

from somo_rl.utils.import_environment import import_env


def load_env(run_config, render=False, debug=False):

    import_env(run_config["env_id"])

    env = gym.make(
        run_config["env_id"],
        run_config=run_config,
        run_ID=f"{run_config['env_id']}",
        render=render,
        debug=debug,
    )

    env.seed(run_config['seed'])

    return env