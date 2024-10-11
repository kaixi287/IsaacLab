#!/bin/bash

# Array of load_run arguments
load_mlp_runs=(
    "2024-10-09_13-24-47_mlp_rand_hip"
    "2024-10-09_20-57-34_mlp_rand_hip"
    "2024-10-06_21-15-41_mlp_rand_hip"
    "2024-10-10_12-10-05_mlp_aug_hip"
    "2024-10-10_13-52-04_mlp_aug_hip"
    "2024-10-10_12-16-31_mlp_aug_hip"
    "2024-10-10_14-00-55_mlp_aug_hip"
    "2024-10-08_17-00-37_mlp_lf_hip"
    "2024-10-09_18-59-38_mlp_lf_hip"
    "2024-10-09_16-34-48_mlp_lf_hip"
    "2024-10-09_16-36-29_mlp_lf_hip"
)

# Array of load_run arguments
load_rnn_runs=(
    "2024-10-06_21-16-24_rnn_rand_hip"
    "2024-10-07_23-28-07_rnn_rand_hip"
    "2024-10-08_00-42-26_rnn_rand_hip"
    "2024-10-10_12-11-41_rnn_aug_hip"
    "2024-10-10_12-19-05_rnn_aug_hip"
    "2024-10-10_14-02-45_rnn_aug_hip"
    "2024-10-10_14-05-17_rnn_aug_hip"
    "2024-10-09_19-00-50_rnn_lf_hip"
    "2024-10-09_19-02-39_rnn_lf_hip"
    "2024-10-09_19-04-40_rnn_lf_hip"
    "2024-10-08_17-02-45_rnn_lf_hip"
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
    ./docker/cluster/cluster_interface.sh job base --task Isaac-Velocity-Flat-G1-Play-v0 --headless --enable_cameras --load_run "$run" --num_envs 1000 --seed 7 --logger wandb --log_project_name g1_id_eval --log_run_name "$log_run_name" --actor_critic_class_name ActorCritic

    # Wait for 10 minutes (600 seconds) before running the next command
    sleep 250
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
    ./docker/cluster/cluster_interface.sh job base --task Isaac-Velocity-Flat-G1-Play-v0 --headless --enable_cameras --load_run "$run" --num_envs 1000 --seed 7 --logger wandb --log_project_name g1_id_eval --log_run_name "$log_run_name" --actor_critic_class_name ActorCriticRecurrent

    # Wait for 10 minutes (600 seconds) before running the next command
    sleep 250
done
