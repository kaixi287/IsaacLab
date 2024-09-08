#!/bin/bash

# Command template
command="./docker/cluster/cluster_interface.sh job base --task Isaac-PosTracking-Flat-Anymal-D-v0 --headless --logger wandb --run_name mlp_aug_time --log_project_name position_tracking5 --log_run_name mlp_aug"

sleep 11000
# Loop through seeds 1 to 5
for seed in {2..5}; do
    # sleep 3600
    # Execute the command with the current seed
    $command --seed $seed
    
    # Wait for 10 minutes (600 seconds) before running the next command
    sleep 400
done

# Command template
# rnn_command="./docker/cluster/cluster_interface.sh job base --task Isaac-PosTracking-Flat-Anymal-D-v0 --headless --logger wandb --run_name rnn_aug_time --log_project_name position_tracking5 --log_run_name rnn_aug --actor_critic_class_name ActorCriticRecurrent"

# # Loop through seeds 1 to 5
# for seed in {1..5}; do
#     # sleep 600
#     # Execute the command with the current seed
#     $rnn_command --seed $seed
    
#     # Wait for 10 minutes (600 seconds) before running the next command
#     sleep 600
# done
