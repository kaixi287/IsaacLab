#!/bin/bash
load_mlp_runs=(
    "2024-08-24_12-20-36_mlp_rand_1"
    "2024-08-25_11-01-02_mlp_rand_2"
    "2024-08-25_11-12-18_mlp_rand_3"
    "2024-08-25_11-27-06_mlp_rand_5"
    "2024-08-25_13-55-47_mlp_aug_1"
    "2024-08-25_14-56-26_mlp_aug_2"
    "2024-08-25_15-12-32_mlp_aug_3"
    "2024-08-25_16-10-01_mlp_aug_5"
)


# Array of load_run arguments
load_rnn_runs=(
    "2024-08-26_13-18-17_rnn_rand_6"
    "2024-08-26_15-47-46_rnn_rand_7"
    "2024-08-26_15-52-59_rnn_rand_8"
    "2024-08-26_23-07-14_rnn_rand_9"
    "2024-08-26_13-52-51_rnn_aug_6"
    "2024-08-26_18-54-48_rnn_aug_7"
    "2024-08-26_19-35-46_rnn_aug_8"
    "2024-08-26_23-33-14_rnn_aug_9"
)

# Loop through each load_run argument
for run in "${load_mlp_runs[@]}"; do

    # Execute the command with the current load_run argument and appropriate log_run_name
    ./docker/container.sh job --task Isaac-PosTracking-Flat-Anymal-D-Play-v0 --headless --enable_cameras --load_run "$run" --num_envs 8 --seed 100 --video --video_length 2000 --actor_critic_class_name ActorCritic
    
    # Wait for 10 minutes (600 seconds) before running the next command
    sleep 400
done

# Loop through each load_run argument
for run in "${load_rnn_runs[@]}"; do

    # Execute the command with the current load_run argument and appropriate log_run_name
    ./docker/container.sh job --task Isaac-PosTracking-Flat-Anymal-D-Play-v0 --headless --enable_cameras --load_run "$run" --num_envs 8 --seed 100 --video --video_length 2000 --actor_critic_class_name ActorCriticRecurrent
    
    # Wait for 10 minutes (600 seconds) before running the next command
    sleep 400
done
