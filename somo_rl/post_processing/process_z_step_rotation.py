import os
import sys
import argparse
from pathlib import Path
import glob

import matplotlib.pyplot as plt
import pickle

path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.insert(0, path)

from user_settings import EXPERIMENT_ABS_PATH


class Process_z_rotation_data:
    def __init__(self, exp_abs_path, run_ID):
        self.run_ID = run_ID
        self.run_dir = Path(exp_abs_path)
        for subdivision in self.run_ID:
            self.run_dir = self.run_dir / subdivision
        self.run_results_dir = self.run_dir / "results" / "processed_data"

def process_z_step_rotation_data(
        exp_abs_path,
        exp_name,
        run_group_name,
        run_name,
        show=True,
        save_figs=True
):
    run_ID = [exp_name, run_group_name, run_name]
    z_step_rotation_data = Process_z_rotation_data(exp_abs_path=exp_abs_path, run_ID=run_ID)
    for i, res in enumerate(glob.glob(os.path.join(z_step_rotation_data.run_results_dir, "**", "info.pkl"), recursive=True)):
        print((list(res.split(os.sep)))[-2])
        with open(res, 'rb') as f:
            plt.plot(pickle.load(f)['z_rotation_step'])
            plt.xlabel = "steps"
            plt.ylabel = "z_rotation_step"
            plt.title = (list(res.split(os.sep)))[-2]
            plt.tight_layout()

            if save_figs:
                print("save to ", res[:-8])
                plt.savefig(os.path.join(res[:-8], "z_step_rotation.png"))
                plt.savefig(os.path.join(res[:-8], "z_step_rotation.eps"), format="eps")

            if show:
                plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="postprocess run argument parser")
    parser.add_argument(
        "-e",
        "--exp_name",
        help="Experiment name",
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
        "-g",
        "--run_group_name",
        help="Run-group name",
        required=True,
        default=None,
    )
    parser.add_argument(
        "-r",
        "--run_name",
        help="Run Name",
        required=True,
        default=None,
    )

    parser.add_argument(
        "--save",
        help="save plots",
        action="store_true",
    )

    arg = parser.parse_args()

    process_z_step_rotation_data(
        arg.exp_abs_path,
        arg.exp_name,
        arg.run_group_name,
        arg.run_name,
        show=True,
        save_figs=arg.save
    )