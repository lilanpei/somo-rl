import os, sys, shutil
import pytest

path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, path)

from pathlib import Path
from somo_rl.train_policy import run as run_training


def safe_delete(folder):
    """
    deletes everything in folder
    :param folder:
    :return:
    """
    for filename in os.listdir(folder):
        file_path = os.path.join(folder, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print("Failed to delete %s. Reason: %s" % (file_path, e))


def somogym_training_tester(env_name, render=False, debug=False):
    test_experiments_path = os.path.join(os.path.dirname(__file__), f"tmp/")
    # TODO: get this from somogym import
    run_config_path = Path(os.path.dirname(__file__)) / f"../tests/assets/test_configs/{env_name}.yaml"
    
    run_training(
        test_experiments_path,
        "training_test_runs",
        "trainig_test_run_group",
        f"{env_name}-test",
        render=render,
        debug=debug,
        overwrite=True,
        note="TRAINING TEST!",
        run_config_input=run_config_path,
    )

    safe_delete(test_experiments_path)

# ANTIPODAL GRIPPER
def test_AntipodalGripper_training():
    somogym_training_tester("AntipodalGripper")
@pytest.mark.gui
def test_AntipodalGripper_training_gui():
    somogym_training_tester("AntipodalGripper", render=True, debug=True)

# IN-HAND MANIPULATION
def test_InHandManipulation_training():
    somogym_training_tester("InHandManipulation")
@pytest.mark.gui
def test_InHandManipulation_training_gui():
    somogym_training_tester("InHandManipulation", render=True, debug=True)

# IN-HAND MANIPULATION INVERTED
def test_InHandManipulationInverted_training():
    somogym_training_tester("InHandManipulationInverted")
@pytest.mark.gui
def test_InHandManipulationInverted_training_gui():
    somogym_training_tester("InHandManipulationInverted", render=True, debug=True)

# PEN SPINNER
def test_PenSpinner_training():
    somogym_training_tester("PenSpinner")
@pytest.mark.gui
def test_PenSpinner_training_gui():
    somogym_training_tester("PenSpinner", render=True, debug=True)

# # PEN SPINNER FAR
# def test_PenSpinnerFar_training():
#     somogym_training_tester("PenSpinnerFar")
# @pytest.mark.gui
# def test_PenSpinnerFar_training_gui():
#     somogym_training_tester("PenSpinnerFar", render=True, debug=True)

# PLANAR BLOCK PUSHING
def test_PlanarBlockPushing_training():
    somogym_training_tester("PlanarBlockPushing")
@pytest.mark.gui
def test_PlanarBlockPushing_training_gui():
    somogym_training_tester("PlanarBlockPushing", render=True, debug=True)

# PLANAR REACHING
def test_PlanarReaching_training():
    somogym_training_tester("PlanarReaching")
@pytest.mark.gui
def test_PlanarReaching_training_gui():
    somogym_training_tester("PlanarReaching", render=True, debug=True)

# SNAKE LOCOMOTION CONTINUOUS (CONTINUOUS LOCOSNAKE DOES NOT WORK YET)
# def test_SnakeLocomotionContinuous_training():
#     somogym_training_tester("SnakeLocomotionContinuous")
# @pytest.mark.gui
# def test_SnakeLocomotionContinuous_training_gui():
#     somogym_training_tester("SnakeLocomotionContinuous", render=True, debug=True)

# SNAKE LOCOMOTION DISCRETE
def test_SnakeLocomotionDiscrete_training():
    somogym_training_tester("SnakeLocomotionDiscrete")
@pytest.mark.gui
def test_SnakeLocomotionDiscrete_training_gui():
    somogym_training_tester("SnakeLocomotionDiscrete", render=True, debug=True)

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "gui":
        test_AntipodalGripper_training_gui()
        test_InHandManipulation_training_gui()
        test_InHandManipulationInverted_training_gui()
        test_PenSpinner_training_gui()
        # test_PenSpinnerFar_training_gui()
        test_PlanarBlockPushing_training_gui()
        test_PlanarReaching_training_gui()
        # test_SnakeLocomotionContinuous_training_gui()
        test_SnakeLocomotionDiscrete_training_gui()
    else:
        test_AntipodalGripper_training()
        test_InHandManipulation_training()
        test_InHandManipulationInverted_training()
        test_PenSpinner_training()
        # test_PenSpinnerFar_training()
        test_PlanarBlockPushing_training()
        test_PlanarReaching_training()
        # test_SnakeLocomotionContinuous_training()
        test_SnakeLocomotionDiscrete_training()
