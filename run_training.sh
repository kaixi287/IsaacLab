#!/bin/bash

# Command template
# mlp_command="./docker/cluster/cluster_interface.sh job base --task Isaac-PosTracking-Flat-G1-v0 --headless --logger wandb --log_project_name g1_pos_tracking_payload_final --log_run_name mlp_aug --actor_critic_class ActorCritic"
mlp_command="./docker/cluster/cluster_interface.sh job base --task Isaac-PosTracking-Rough-Anymal-D-v0 --headless --logger wandb --log_project_name hardware_experiment --log_run_name mlp_aug_random_05_add_stand_command_1 --actor_critic_class ActorCritic"

# Loop through seeds 1 to 5
for seed in {7..7}; do
    # sleep 600
    # Execute the command with the current seed
    $mlp_command --seed $seed --run_name mlp_aug_random_05_add_stand_command_1_$seed

    # Wait for 10 minutes (600 seconds) before running the next command
    sleep 30
done

# Command template
# rnn_command="./docker/cluster/cluster_interface.sh job base --task Isaac-PosTracking-Flat-G1-v0 --headless --logger wandb --log_project_name g1_pos_tracking_payload_final --log_run_name rnn_aug --actor_critic_class ActorCriticRecurrent"
rnn_command="./docker/cluster/cluster_interface.sh job base --task Isaac-PosTracking-Rough-Anymal-D-v0 --headless --logger wandb --log_project_name hardware_experiment --log_run_name rnn_aug_random_05_add_stand_command_1 --actor_critic_class ActorCriticRecurrent"

# Loop through seeds 1 to 5
for seed in {7..7}; do
    # sleep 100
    # Execute the command with the current seed
    $rnn_command --seed $seed --run_name rnn_aug_random_05_add_stand_command_1_$seed

    # Wait for 10 minutes (600 seconds) before running the next command
    sleep 30
done
