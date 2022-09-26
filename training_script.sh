#!/bin/bash

for seed in {0..19}
do
    python somo_rl/train_policy.py -e "InHandManipulationInverted-v0" -g "SAC" -r "random_seed_$seed" --note "z_rotation_step: 1000, Episode: 5000, learning_rate: 0.0003, buffer_size: 500000, learning_starts: 50, batch_size: 25" -o
done