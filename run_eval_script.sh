#!/bin/bash

# Array of load_run arguments
load_mlp_runs=(
    "2025-01-07_12-39-14_mlp_aug_joint_0"
    "2025-01-07_12-39-47_mlp_aug_joint_1"
    "2025-01-07_12-39-47_mlp_aug_joint_2"
    "2025-01-07_12-40-50_mlp_aug_joint_3"
    "2025-01-07_12-40-50_mlp_aug_joint_5"
    "2025-01-07_12-41-15_mlp_aug_joint_4"
    "2025-01-07_12-47-42_mlp_lf_joint_0"
    "2025-01-07_12-47-42_mlp_lf_joint_1"
    "2025-01-07_12-48-44_mlp_lf_joint_2"
    "2025-01-07_12-49-29_mlp_lf_joint_3"
    "2025-01-07_12-50-05_mlp_lf_joint_4"
    "2025-01-07_12-52-45_mlp_lf_joint_5"
    "2025-01-07_13-04-14_mlp_rand_joint_0"
    "2025-01-07_13-04-58_mlp_rand_joint_1"
    "2025-01-07_13-05-13_mlp_rand_joint_3"
    "2025-01-07_13-05-14_mlp_rand_joint_2"
    "2025-01-07_13-05-14_mlp_rand_joint_4"
    "2025-01-07_13-05-14_mlp_rand_joint_5"
)

load_rnn_runs=(
    "2025-01-07_12-52-20_rnn_lf_joint_0"
    "2025-01-07_12-52-45_rnn_lf_joint_1"
    "2025-01-07_12-53-40_rnn_lf_joint_2"
    "2025-01-07_12-53-40_rnn_lf_joint_3"
    "2025-01-07_12-53-40_rnn_lf_joint_4"
    "2025-01-07_12-56-31_rnn_lf_joint_5"
    "2025-01-07_13-07-46_rnn_rand_joint_0"
    "2025-01-07_13-07-46_rnn_rand_joint_1"
    "2025-01-07_13-09-40_rnn_rand_joint_2"
    "2025-01-07_13-10-10_rnn_rand_joint_3"
    "2025-01-07_13-10-45_rnn_rand_joint_4"
    "2025-01-07_13-11-28_rnn_rand_joint_5"
    "2025-01-07_12-41-38_rnn_aug_joint_0"
    "2025-01-07_12-42-31_rnn_aug_joint_1"
    "2025-01-07_12-43-17_rnn_aug_joint_2"
    "2025-01-07_12-43-48_rnn_aug_joint_3"
    "2025-01-07_12-43-48_rnn_aug_joint_4"
    "2025-01-07_12-44-56_rnn_aug_joint_5"
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
    ./docker/cluster/cluster_interface.sh job base --task Isaac-PosTracking-Flat-Anymal-D-Play-v0 --headless --enable_cameras --load_run "$run" --num_envs 1000 --seed 24 --logger wandb --log_project_name aynmal_pos_tracking_payload_id_eval_concat3 --log_run_name "$log_run_name"_ood --actor_critic_class_name ActorCritic

    # Wait for 10 minutes (600 seconds) before running the next command
    sleep 10
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
    ./docker/cluster/cluster_interface.sh job base --task Isaac-PosTracking-Flat-Anymal-D-Play-v0 --headless --enable_cameras --load_run "$run" --num_envs 1000 --seed 24 --logger wandb --log_project_name aynmal_pos_tracking_payload_id_eval_concat3 --log_run_name "$log_run_name"_ood --actor_critic_class_name ActorCriticRecurrent

    # Wait for 10 minutes (600 seconds) before running the next command
    sleep 10
done
