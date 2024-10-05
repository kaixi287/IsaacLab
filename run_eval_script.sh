#!/bin/bash

load_mlp_runs=(
    "2024-10-03_20-58-54_mlp_rand_payload"
    "2024-10-03_21-27-24_mlp_aug_payload"
    "2024-10-03_21-32-22_mlp_aug_payload"
    "2024-10-03_21-37-52_mlp_aug_payload"
    "2024-10-03_22-30-12_mlp_lf_payload"
    "2024-10-03_22-35-42_mlp_lf_payload"
    "2024-10-03_22-41-23_mlp_lf_payload"
)

# Array of load_run arguments
load_rnn_runs=(
    "2024-10-03_21-04-13_rnn_rand_payload"
    "2024-10-03_21-43-25_rnn_aug_payload"
    "2024-10-03_21-48-44_rnn_aug_payload"
    "2024-10-03_21-54-05_rnn_aug_payload"
    "2024-10-03_22-46-32_rnn_lf_payload"
    "2024-10-03_22-52-15_rnn_lf_payload"
    "2024-10-03_22-57-18_rnn_lf_payload"
)


# Loop through each load_run argument
for run in "${load_mlp_runs[@]}"; do
    # Determine the log_run_name based on the load_run argument
    if [[ "$run" == *"mlp_rand"* ]]; then
        log_run_name="mlp_rand"
    elif [[ "$run" == *"mlp_aug"* ]]; then
        log_run_name="mlp_aug"
    elif [[ "$run" == *"mlp_lf"* ]]; then
        log_run_name="mlp_lf"
    else
        log_run_name="None"
    fi

    # Execute the command with the current load_run argument and appropriate log_run_name
    ./docker/cluster/cluster_interface.sh job base --task Isaac-PosTracking-Flat-Anymal-D-Play-v0 --headless --enable_cameras --load_run "$run" --num_envs 1000 --seed 10 --logger wandb --log_project_name pos_tracking_with_payload_eval_ood --log_run_name "$log_run_name" --actor_critic_class_name ActorCritic

    # Wait for 10 minutes (600 seconds) before running the next command
    sleep 200
done

# Loop through each load_run argument
for run in "${load_rnn_runs[@]}"; do
    # Determine the log_run_name based on the load_run argument
    if [[ "$run" == *"rnn_rand"* ]]; then
        log_run_name="rnn_rand"
    elif [[ "$run" == *"rnn_aug"* ]]; then
        log_run_name="rnn_aug"
    elif [[ "$run" == *"rnn_lf"* ]]; then
        log_run_name="rnn_lf"
    else
        log_run_name="None"
    fi

    # Execute the command with the current load_run argument and appropriate log_run_name
    ./docker/cluster/cluster_interface.sh job base --task Isaac-PosTracking-Flat-Anymal-D-Play-v0 --headless --enable_cameras --load_run "$run" --num_envs 1000 --seed 10 --logger wandb --log_project_name pos_tracking_with_payload_eval_ood --log_run_name "$log_run_name" --actor_critic_class_name ActorCriticRecurrent

    # Wait for 10 minutes (600 seconds) before running the next command
    sleep 200
done
