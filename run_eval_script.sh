#!/bin/bash

# Array of load_run arguments
load_mlp_runs=(
    # "2024-11-25_00-49-47_mlp_aug_joint_15"
    # "2024-11-25_00-51-50_mlp_aug_joint_16"
    # "2024-11-25_00-54-18_mlp_aug_joint_17"
    # "2024-11-25_00-56-17_mlp_aug_joint_18"
    # "2024-11-25_00-58-54_mlp_aug_joint_19"
    # "2024-11-25_08-07-40_mlp_lf_joint_15"
    # "2024-11-25_08-09-36_mlp_lf_joint_16"
    # "2024-11-25_08-11-34_mlp_lf_joint_17"
    # "2024-11-25_08-12-55_mlp_lf_joint_18"
    # "2024-11-25_08-15-00_mlp_lf_joint_19"
    # "2024-11-25_08-55-25_mlp_rand_joint_15"
    # "2024-11-25_08-57-29_mlp_rand_joint_16"
    # "2024-11-25_08-58-54_mlp_rand_joint_17"
    # "2024-11-25_09-01-57_mlp_rand_joint_18"
    # "2024-11-25_09-03-06_mlp_rand_joint_19"
)

load_rnn_runs=(
    # "2024-11-25_01-00-02_rnn_aug_joint_15"
    "2024-11-25_01-03-36_rnn_aug_joint_16"
    "2024-11-25_01-07-15_rnn_aug_joint_17"
    "2024-11-25_01-11-06_rnn_aug_joint_18"
    "2024-11-25_01-14-47_rnn_aug_joint_19"
    "2024-11-25_08-17-40_rnn_lf_joint_15"
    "2024-11-25_08-21-23_rnn_lf_joint_16"
    "2024-11-25_08-24-32_rnn_lf_joint_17"
    "2024-11-25_08-28-55_rnn_lf_joint_18"
    "2024-11-25_08-32-00_rnn_lf_joint_19"
    "2024-11-25_09-06-33_rnn_rand_joint_15"
    "2024-11-25_09-10-09_rnn_rand_joint_16"
    "2024-11-25_09-13-11_rnn_rand_joint_17"
    "2024-11-25_09-17-17_rnn_rand_joint_18"
    "2024-11-25_09-20-37_rnn_rand_joint_19"
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
    ./docker/cluster/cluster_interface.sh job base --task Isaac-PosTracking-Flat-Anymal-D-Play-v0 --headless --enable_cameras --load_run "$run" --num_envs 1000 --seed 7 --logger wandb --log_project_name anymal_pos_tracking_joint_failure_id_eval_final --log_run_name "$log_run_name" --actor_critic_class_name ActorCritic

    # Wait for 10 minutes (600 seconds) before running the next command
    sleep 20
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
    ./docker/cluster/cluster_interface.sh job base --task Isaac-PosTracking-Flat-Anymal-D-Play-v0 --headless --enable_cameras --load_run "$run" --num_envs 1000 --seed 7 --logger wandb --log_project_name anymal_pos_tracking_joint_failure_id_eval_final --log_run_name "$log_run_name" --actor_critic_class_name ActorCriticRecurrent

    # Wait for 10 minutes (600 seconds) before running the next command
    sleep 20
done
