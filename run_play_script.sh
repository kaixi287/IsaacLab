#!/bin/bash
load_mlp_runs=(
    "2024-10-05_23-23-33_mlp_aug_payload"
    "2024-10-05_23-25-36_mlp_aug_payload"
    "2024-10-05_23-30-19_mlp_aug_payload"
    "2024-10-06_04-51-16_mlp_rand_payload"
    "2024-10-06_04-53-03_mlp_rand_payload"
    "2024-10-06_04-55-07_mlp_rand_payload"
)

# Array of load_run arguments
load_rnn_runs=(
    "2024-10-05_23-30-49_rnn_aug_payload"
    "2024-10-06_01-16-19_rnn_aug_payload"
    "2024-10-06_01-35-01_rnn_aug_payload"
    "2024-10-06_04-57-15_rnn_rand_payload"
    "2024-10-06_04-59-19_rnn_rand_payload"
    "2024-10-06_06-51-34_rnn_rand_payload"
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
