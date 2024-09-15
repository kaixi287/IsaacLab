#!/bin/bash

# # Command template
# command="./docker/cluster/cluster_interface.sh job base --task Isaac-PosTracking-Flat-Anymal-D-v0 --headless --logger wandb --run_name mlp_aug_time --log_project_name position_tracking5 --log_run_name mlp_aug"

# sleep 11000
# # Loop through seeds 1 to 5
# for seed in {2..5}; do
#     # sleep 3600
#     # Execute the command with the current seed
#     $command --seed $seed
    
#     # Wait for 10 minutes (600 seconds) before running the next command
#     sleep 400
# done

# Command template
rnn_command="./docker/cluster/cluster_interface.sh job base --task Isaac-PosTracking-Flat-Anymal-D-v0 --headless --logger wandb --run_name rnn_aug_mid --log_project_name position_tracking3 --log_run_name rnn_aug --actor_critic_class ActorCriticRecurrent"

# Loop through seeds 1 to 5
for seed in {1..2}; do
    # sleep 600
    # Execute the command with the current seed
    $rnn_command --seed $seed
    
    # Wait for 10 minutes (600 seconds) before running the next command
    sleep 300
done

# Command template
rnn_command="./docker/cluster/cluster_interface.sh job base --task Isaac-PosTracking-Flat-Anymal-D-v0 --headless --logger wandb --run_name rnn_aug_mid --log_project_name position_tracking3 --log_run_name rnn_aug --actor_critic_class ActorCriticRecurrent --lr_step_size 500 --lr_decay 0.5"

# Loop through seeds 1 to 5
for seed in {1..2}; do
    # sleep 600
    # Execute the command with the current seed
    $rnn_command --seed $seed
    
    # Wait for 10 minutes (600 seconds) before running the next command
    sleep 300
done
