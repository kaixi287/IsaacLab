#!/bin/bash

# Command template
command="./docker/container.sh job --task Isaac-PosTracking-Flat-Anymal-D-v0 --headless --logger wandb --run_name mlp_rand_mid --log_project_name position_tracking3 --log_run_name mlp_rand_mid"

sleep 11000
# Loop through seeds 1 to 5
for seed in {2..5}; do
    # Execute the command with the current seed
    $command --seed $seed
    
    # Wait for 10 minutes (600 seconds) before running the next command
    sleep 1800
done

# Command template
rnn_command="./docker/container.sh job --task Isaac-PosTracking-Flat-Anymal-D-v0 --headless --logger wandb --run_name rnn_rand_mid --log_project_name position_tracking3 --log_run_name rnn_rand_mid --actor_critic_class_name ActorCriticRecurrent"

# Loop through seeds 1 to 5
for seed in {2..5}; do
    # Execute the command with the current seed
    $command --seed $seed
    
    # Wait for 10 minutes (600 seconds) before running the next command
    sleep 1800
done