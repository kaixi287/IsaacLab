#!/bin/bash
load_mlp_runs=(
    # "2024-11-10_22-26-56_mlp_aug"
    # "2024-11-10_22-28-27_mlp_rand"
    # "2024-11-10_23-00-53_mlp_rand"
    # "2024-11-10_23-06-22_mlp_rand"
    # "2024-11-10_23-11-27_mlp_rand"
    # "2024-11-10_23-17-08_mlp_rand"
    # "2024-11-10_23-33-35_mlp_aug"
    # "2024-11-10_23-37-20_mlp_aug"
    # "2024-11-10_23-40-58_mlp_aug"
    # "2024-11-10_23-45-44_mlp_aug"
    # "2024-11-11_00-41-23_mlp_lf"
    # "2024-11-11_00-44-33_mlp_lf"
    # "2024-11-11_00-48-58_mlp_lf"
    # "2024-11-11_00-52-37_mlp_lf"
    # "2024-11-11_11-32-39_mlp_lf"
    # "2024-11-11_11-33-09_mlp_aug"
    "2024-11-12_21-02-43_mlp_aug_increase_hip_dev_entropy"
    "2024-11-12_21-03-48_mlp_lf_increase_hip_dev_entropy"
    "2024-11-12_21-05-14_mlp_rand_increase_hip_dev_entropy"
    "2024-11-13_00-23-10_mlp_lf_increase_hip_dev"
    "2024-11-13_00-24-09_mlp_aug_increase_hip_dev"
    "2024-11-13_00-26-00_mlp_rand_increase_hip_dev"
)

load_rnn_runs=(
    # "2024-11-10_23-00-06_rnn_rand"
    # "2024-11-10_23-22-23_rnn_rand"
    # "2024-11-10_23-28-01_rnn_rand"
    # "2024-11-10_23-30-46_rnn_rand"
    # "2024-11-10_23-33-09_rnn_rand"
    # "2024-11-10_23-49-02_rnn_aug"
    # "2024-11-10_23-53-20_rnn_aug"
    # "2024-11-11_00-00-49_rnn_aug"
    # "2024-11-11_01-00-48_rnn_lf"
    # "2024-11-11_01-03-23_rnn_lf"
    # "2024-11-11_01-07-41_rnn_lf"
    # "2024-11-11_11-31-07_rnn_lf"
    # "2024-11-11_00-56-49_rnn_lf"
    "2024-11-12_21-02-27_rnn_aug_increase_hip_dev_entropy"
    "2024-11-12_21-03-48_rnn_lf_increase_hip_dev_entropy"
    "2024-11-12_21-05-14_rnn_rand_increase_hip_dev_entropy"
    "2024-11-13_00-23-10_rnn_lf_increase_hip_dev"
    "2024-11-13_00-24-45_rnn_aug_increase_hip_dev"
    "2024-11-13_00-25-02_rnn_rand_increase_hip_dev"
)

# Loop through each load_run argument
for run in "${load_mlp_runs[@]}"; do

    # Execute the command with the current load_run argument and appropriate log_run_name
    ./docker/cluster/cluster_interface.sh job base --task Isaac-PosTracking-Flat-G1-Play-v0 --headless --enable_cameras --load_run "$run" --num_envs 8 --seed 100 --video --video_length 2000 --actor_critic_class_name ActorCritic

    # Wait for 10 minutes (600 seconds) before running the next command
    sleep 60
done

# Loop through each load_run argument
for run in "${load_rnn_runs[@]}"; do

    # Execute the command with the current load_run argument and appropriate log_run_name
    ./docker/cluster/cluster_interface.sh job base --task Isaac-PosTracking-Flat-G1-Play-v0 --headless --enable_cameras --load_run "$run" --num_envs 8 --seed 100 --video --video_length 2000 --actor_critic_class_name ActorCriticRecurrent

    # Wait for 10 minutes (600 seconds) before running the next command
    sleep 60
done
