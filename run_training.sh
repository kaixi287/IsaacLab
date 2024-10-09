#!/bin/bash

# Command template
# mlp_command="./docker/cluster/cluster_interface.sh job base --task Isaac-PosTracking-Flat-Anymal-D-v0 --headless --logger wandb --run_name mlp_lf_payload --log_project_name pos_tracking_with_payload --log_run_name mlp_lf --actor_critic_class ActorCritic"
mlp_command="./docker/cluster/cluster_interface.sh job base --task Isaac-Velocity-Flat-G1-v0 --headless --logger wandb --run_name mlp_aug_hip --log_project_name g1 --log_run_name mlp_aug --actor_critic_class ActorCritic"

# Loop through seeds 1 to 5
for seed in {1..3}; do
    # sleep 600
    # Execute the command with the current seed
    $mlp_command --seed $seed

    # Wait for 10 minutes (600 seconds) before running the next command
    sleep 100
done

# Command template
# rnn_command="./docker/cluster/cluster_interface.sh job base --task Isaac-PosTracking-Flat-Anymal-D-v0 --headless --logger wandb --run_name rnn_lf_payload --log_project_name pos_tracking_with_payload --log_run_name rnn_lf --actor_critic_class ActorCriticRecurrent"
rnn_command="./docker/cluster/cluster_interface.sh job base --task Isaac-Velocity-Flat-G1-v0 --headless --logger wandb --run_name rnn_aug_hip --log_project_name g1 --log_run_name rnn_aug --actor_critic_class ActorCriticRecurrent"

# Loop through seeds 1 to 5
for seed in {1..3}; do
    # sleep 600
    # Execute the command with the current seed
    $rnn_command --seed $seed

    # Wait for 10 minutes (600 seconds) before running the next command
    sleep 100
done
