#!/bin/bash
load_mlp_runs=(
    "2024-10-03_20-58-54_mlp_rand_payload"
    "2024-10-03_21-27-24_mlp_aug_payload"
    "2024-10-03_21-32-22_mlp_aug_payload"
    "2024-10-03_21-37-52_mlp_aug_payload"
)

# Array of load_run arguments
load_rnn_runs=(
    "2024-10-03_21-04-13_rnn_rand_payload"
    "2024-10-03_21-43-25_rnn_aug_payload"
    "2024-10-03_21-48-44_rnn_aug_payload"
    "2024-10-03_21-54-05_rnn_aug_payload"
)

# Loop through each load_run argument
for run in "${load_mlp_runs[@]}"; do

    # Execute the command with the current load_run argument and appropriate log_run_name
    ./docker/cluster/cluster_interface.sh job base --task Isaac-PosTracking-Flat-Anymal-D-Play-v0 --headless --enable_cameras --load_run "$run" --num_envs 8 --seed 100 --video --video_length 2000 --actor_critic_class_name ActorCritic

    # Wait for 10 minutes (600 seconds) before running the next command
    sleep 200
done

# Loop through each load_run argument
for run in "${load_rnn_runs[@]}"; do

    # Execute the command with the current load_run argument and appropriate log_run_name
    ./docker/cluster/cluster_interface.sh job base --task Isaac-PosTracking-Flat-Anymal-D-Play-v0 --headless --enable_cameras --load_run "$run" --num_envs 8 --seed 100 --video --video_length 2000 --actor_critic_class_name ActorCriticRecurrent

    # Wait for 10 minutes (600 seconds) before running the next command
    sleep 200
done
