#!/bin/bash

# Array of load_run arguments
load_mlp_runs=(
    "2024-12-03_23-04-21_mlp_rand_joint_77"
    "2024-12-04_11-52-18_mlp_aug_joint_77"
    "2024-12-03_22-47-27_mlp_lf_joint_77"
)

load_rnn_runs=(
    "2024-12-04_11-49-01_rnn_rand_joint_77"
    "2024-12-04_11-50-45_rnn_lf_joint_77"
    "2024-12-04_11-51-26_rnn_aug_joint_77"
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
    ./docker/cluster/cluster_interface.sh job base --task Isaac-Velocity-Flat-Anymal-D-Play-v0 --headless --enable_cameras --load_run "$run" --num_envs 1000 --seed 7 --logger wandb --log_project_name anymal_vel_tracking_joint_failure_id_eval_final --log_run_name "$log_run_name" --actor_critic_class_name ActorCritic

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
    ./docker/cluster/cluster_interface.sh job base --task Isaac-Velocity-Flat-Anymal-D-Play-v0 --headless --enable_cameras --load_run "$run" --num_envs 1000 --seed 7 --logger wandb --log_project_name anymal_vel_tracking_joint_failure_id_eval_final --log_run_name "$log_run_name" --actor_critic_class_name ActorCriticRecurrent

    # Wait for 10 minutes (600 seconds) before running the next command
    sleep 10
done
