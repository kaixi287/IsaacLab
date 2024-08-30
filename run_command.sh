#!/bin/bash

# Array of load_run arguments
load_mlp_runs=(
    # "2024-08-25_19-17-58_mlp_lf_1"
    # "2024-08-25_19-39-33_mlp_lf_2"
    # "2024-08-25_19-46-47_mlp_lf_3"
    # "2024-08-25_20-38-28_mlp_lf_5"
    # "2024-08-26_14-58-53_mlp_lf_6"
    # "2024-08-26_15-53-27_mlp_lf_7"
    # "2024-08-26_17-00-18_mlp_lf_8"
    # "2024-08-26_23-41-19_mlp_lf_9"
    # "2024-08-24_12-20-36_mlp_rand_1"
    # "2024-08-25_11-01-02_mlp_rand_2"
    # "2024-08-25_11-12-18_mlp_rand_3"
    # "2024-08-25_11-27-06_mlp_rand_5"
    # "2024-08-26_13-25-59_mlp_rand_6"
    # "2024-08-26_16-03-05_mlp_rand_7"
    # "2024-08-26_16-20-01_mlp_rand_8"
    # "2024-08-26_23-14-13_mlp_rand_9"
    # "2024-08-25_13-55-47_mlp_aug_1"
    # "2024-08-25_14-56-26_mlp_aug_2"
    "2024-08-25_15-12-32_mlp_aug_3"
    "2024-08-25_16-10-01_mlp_aug_5"
    "2024-08-26_13-34-31_mlp_aug_6"
    "2024-08-26_19-40-21_mlp_aug_7"
    "2024-08-26_19-44-17_mlp_aug_8"
    "2024-08-26_23-19-53_mlp_aug_9"
)

# Array of load_run arguments
load_rnn_runs=(
    "2024-08-24_15-11-15_rnn_rand_1"
    "2024-08-24_17-07-19_rnn_rand_2"
    "2024-08-24_18-05-53_rnn_rand_3"
    "2024-08-25_01-02-51_rnn_rand_5"
    "2024-08-26_13-18-17_rnn_rand_6"
    "2024-08-26_15-47-46_rnn_rand_7"
    "2024-08-26_15-52-59_rnn_rand_8"
    "2024-08-26_23-07-14_rnn_rand_9"
    # "2024-08-25_16-16-46_rnn_lf_1"
    # "2024-08-25_16-27-47_rnn_lf_2"
    # "2024-08-25_18-07-25_rnn_lf_3"
    # "2024-08-25_19-10-33_rnn_lf_5"
    # "2024-08-26_14-45-58_rnn_lf_6"
    # "2024-08-26_16-37-27_rnn_lf_7"
    # "2024-08-26_16-43-13_rnn_lf_8"
    # "2024-08-26_23-37-14_rnn_lf_9"
    "2024-08-26_23-33-14_rnn_aug_9"
    "2024-08-25_11-40-32_rnn_aug_1"
    "2024-08-25_11-53-01_rnn_aug_2"
    "2024-08-25_12-02-04_rnn_aug_3"
    "2024-08-25_13-48-35_rnn_aug_5"
    "2024-08-26_13-52-51_rnn_aug_6"
    "2024-08-26_18-54-48_rnn_aug_7"
    "2024-08-26_19-35-46_rnn_aug_8"
    "2024-08-26_23-33-14_rnn_aug_9"
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
    ./docker/container.sh job --task Isaac-PosTracking-Flat-Anymal-D-Play-v0 --headless --enable_cameras --load_run "$run" --num_envs 1000 --seed 7 --logger wandb --log_project_name position_tracking_oodd_eval2 --log_run_name "$log_run_name" --actor_critic_class_name ActorCritic
    
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
    ./docker/container.sh job --task Isaac-PosTracking-Flat-Anymal-D-Play-v0 --headless --enable_cameras --load_run "$run" --num_envs 1000 --seed 7 --logger wandb --log_project_name position_tracking_oodd_eval2 --log_run_name "$log_run_name" --actor_critic_class_name ActorCriticRecurrent
    
    # Wait for 10 minutes (600 seconds) before running the next command
    sleep 250
done
