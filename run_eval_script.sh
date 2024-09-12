#!/bin/bash

load_mlp_runs=(
    "2024-09-01_15-11-17_mlp_rand_mid"
    "2024-09-04_20-12-15_mlp_rand_mid"
    "2024-09-04_20-24-12_mlp_rand_mid"
    "2024-09-04_20-33-02_mlp_rand_mid"
    "2024-09-04_20-43-59_mlp_rand_mid"
    "2024-09-02_00-16-27_mlp_aug_mid"
    "2024-09-02_00-57-46_mlp_aug_mid"
    "2024-09-02_01-41-21_mlp_aug_mid"
    "2024-09-02_02-22-13_mlp_aug_mid"
    "2024-09-02_15-19-58_mlp_aug_mid"
    "2024-09-05_22-36-11_mlp_lf_mid"
    "2024-09-05_22-42-50_mlp_lf_mid"
    "2024-09-05_22-49-51_mlp_lf_mid"
    "2024-09-05_22-57-09_mlp_lf_mid"
    "2024-09-05_23-04-09_mlp_lf_mid"
)

# Array of load_run arguments
load_rnn_runs=(
    "2024-09-01_15-05-28_rnn_rand_mid"
    "2024-09-04_21-00-09_rnn_rand_mid"
    "2024-09-04_21-04-44_rnn_rand_mid"
    "2024-09-04_21-16-45_rnn_rand_mid"
    "2024-09-04_21-26-30_rnn_rand_mid"
    "2024-09-05_00-00-51_rnn_lf_mid"
    "2024-09-05_00-09-24_rnn_lf_mid"
    "2024-09-05_00-18-26_rnn_lf_mid"
    "2024-09-05_00-28-02_rnn_lf_mid"
    "2024-09-04_23-51-50_rnn_lf_mid"
    "2024-09-06_11-02-12_rnn_aug_mid"
    "2024-09-06_11-07-09_rnn_aug_mid"
    "2024-09-06_11-12-52_rnn_aug_mid"
    "2024-09-06_11-20-50_rnn_aug_mid"
    "2024-09-06_11-34-18_rnn_aug_mid"
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
    ./docker/cluster/cluster_interface.sh job base --task Isaac-PosTracking-Flat-Anymal-D-Play-v0 --headless --enable_cameras --load_run "$run" --num_envs 1000 --seed 10 --logger wandb --log_project_name position_tracking_eval3_10 --log_run_name "$log_run_name" --actor_critic_class_name ActorCritic
    
    # Wait for 10 minutes (600 seconds) before running the next command
    sleep 350
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
    ./docker/cluster/cluster_interface.sh job base --task Isaac-PosTracking-Flat-Anymal-D-Play-v0 --headless --enable_cameras --load_run "$run" --num_envs 1000 --seed 10 --logger wandb --log_project_name position_tracking_eval3_10 --log_run_name "$log_run_name" --actor_critic_class_name ActorCriticRecurrent
    
    # Wait for 10 minutes (600 seconds) before running the next command
    sleep 350
done
