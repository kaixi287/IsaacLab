#!/bin/bash

load_mlp_runs=(
    "2024-09-02_18-21-51_mlp_rand"
    "2024-09-02_18-31-44_mlp_rand"
    "2024-09-02_18-48-11_mlp_rand"
    "2024-09-02_18-52-14_mlp_rand"
    "2024-09-02_19-02-50_mlp_rand"
    "2024-09-03_01-02-06_mlp_aug"
    "2024-09-03_01-59-25_mlp_aug"
    "2024-09-03_03-01-45_mlp_aug"
    "2024-09-03_04-00-49_mlp_aug"
    "2024-09-03_05-01-00_mlp_aug"
    "2024-09-03_16-48-11_mlp_lf"
    "2024-09-03_17-17-27_mlp_lf"
    "2024-09-03_17-25-00_mlp_lf"
    "2024-09-03_17-40-50_mlp_lf"
    "2024-09-03_17-52-26_mlp_lf"
)

# Array of load_run arguments
load_rnn_runs=(
    "2024-09-02_19-16-01_rnn_rand"
    "2024-09-02_19-24-08_rnn_rand"
    "2024-09-02_19-44-50_rnn_rand"
    "2024-09-02_19-59-02_rnn_rand"
    "2024-09-02_20-11-39_rnn_rand"
    "2024-09-02_21-20-18_rnn_aug"
    "2024-09-03_16-09-59_rnn_aug"
    "2024-09-03_16-18-36_rnn_aug"
    "2024-09-03_16-29-30_rnn_aug"
    "2024-09-03_16-40-03_rnn_aug"
    "2024-09-03_18-01-24_rnn_lf"
    "2024-09-03_19-04-28_rnn_lf"
    "2024-09-03_19-16-14_rnn_lf"
    "2024-09-03_19-26-48_rnn_lf"
    "2024-09-03_19-38-07_rnn_lf"
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
    ./docker/container.sh job --task Isaac-PosTracking-Flat-Anymal-D-Play-v0 --headless --enable_cameras --load_run "$run" --num_envs 1000 --seed 7 --logger wandb --log_project_name position_tracking_eval_4 --log_run_name "$log_run_name" --actor_critic_class_name ActorCritic
    
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
    ./docker/container.sh job --task Isaac-PosTracking-Flat-Anymal-D-Play-v0 --headless --enable_cameras --load_run "$run" --num_envs 1000 --seed 7 --logger wandb --log_project_name position_tracking_eval_4 --log_run_name "$log_run_name" --actor_critic_class_name ActorCriticRecurrent
    
    # Wait for 10 minutes (600 seconds) before running the next command
    sleep 350
done
