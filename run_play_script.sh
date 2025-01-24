#!/bin/bash
load_mlp_runs=(
    "2025-01-23_10-09-18_mlp_lf_20"
    "2025-01-23_10-11-00_mlp_lf_21"
)

load_rnn_runs=(
    "2025-01-23_10-09-18_rnn_lf_20"
    "2025-01-23_10-09-19_rnn_lf_21"
)

# rnn_aug_runs=(
#     "2025-01-07_12-41-38_rnn_aug_joint_0"
#     "2025-01-07_12-42-31_rnn_aug_joint_1"
#     "2025-01-07_12-43-17_rnn_aug_joint_2"
#     "2025-01-07_12-43-48_rnn_aug_joint_3"
#     "2025-01-07_12-43-48_rnn_aug_joint_4"
#     "2025-01-07_12-44-56_rnn_aug_joint_5"
# )

# Loop through each load_run argument
for run in "${load_mlp_runs[@]}"; do

    # Execute the command with the current load_run argument and appropriate log_run_name
    ./docker/cluster/cluster_interface.sh job base --task Isaac-PosTracking-Flat-G1-Play-v0 --headless --enable_cameras --load_run "$run" --num_envs 8 --seed 100 --video --video_length 2000 --actor_critic_class_name ActorCritic

    # Wait for 10 minutes (600 seconds) before running the next command
    sleep 10
done

# Loop through each load_run argument
for run in "${load_rnn_runs[@]}"; do

    # Execute the command with the current load_run argument and appropriate log_run_name
    ./docker/cluster/cluster_interface.sh job base --task Isaac-PosTracking-Flat-G1-Play-v0 --headless --enable_cameras --load_run "$run" --num_envs 8 --seed 100 --video --video_length 2000 --actor_critic_class_name ActorCriticRecurrent

    # Wait for 10 minutes (600 seconds) before running the next command
    sleep 10
done

# for run in "${rnn_aug_runs[@]}"; do

#     # Execute the command with the current load_run argument and appropriate log_run_name
#     ./docker/cluster/cluster_interface.sh job base --task Isaac-PosTracking-Flat-Anymal-D-Play-v0 --headless --enable_cameras --load_run "$run" --num_envs 1 --seed 100 --video --video_length 2000 --actor_critic_class_name ActorCriticRecurrent --checkpoint model_4000.pt

#     # Wait for 10 minutes (600 seconds) before running the next command
#     sleep 10
# done
