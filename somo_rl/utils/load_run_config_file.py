import os
import sys
from pathlib import Path

path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.insert(0, path)

from somo_rl.utils import parse_config
from user_settings import EXPERIMENT_ABS_PATH


def load_run_config_file(run_ID, exp_abs_path=EXPERIMENT_ABS_PATH):

    run_dir = Path(exp_abs_path)
    for subdivision in run_ID:
        run_dir = run_dir / subdivision
    run_config_file = run_dir / "run_config.yaml"
    run_config = parse_config.validate_config(run_config_file)

    if not run_config:
        raise (Exception, "ERROR: Invalid run config")

    return run_dir, run_config