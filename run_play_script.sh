#!/bin/bash
load_mlp_runs=(
    # "2024-09-01_15-11-17_mlp_rand_mid"
    "2024-09-04_20-12-15_mlp_rand_mid"
    "2024-09-04_20-24-12_mlp_rand_mid"
    "2024-09-04_20-33-02_mlp_rand_mid"
    "2024-09-04_20-43-59_mlp_rand_mid"
    "2024-09-03_20-41-35_mlp_rand_time"
    "2024-09-04_17-07-54_mlp_rand_time"
    "2024-09-04_17-14-50_mlp_rand_time"
    "2024-09-04_17-22-07_mlp_rand_time"
    "2024-09-04_17-29-51_mlp_rand_time"
)

# Array of load_run arguments
load_rnn_runs=(
    # "2024-09-04_15-51-56_rnn_aug_time"
    # "2024-09-05_16-20-52_rnn_aug_time"
    # "2024-09-05_16-24-54_rnn_aug_time"
    # "2024-09-06_11-41-30_rnn_aug_time"
    # "2024-09-06_14-25-17_rnn_aug_time"
    "2024-09-06_11-02-12_rnn_aug_mid"
    "2024-09-06_11-07-09_rnn_aug_mid"
    "2024-09-06_11-12-52_rnn_aug_mid"
    "2024-09-06_11-20-50_rnn_aug_mid"
    "2024-09-06_11-34-18_rnn_aug_mid"
)

# Loop through each load_run argument
# for run in "${load_mlp_runs[@]}"; do

#     # Execute the command with the current load_run argument and appropriate log_run_name
#     ./docker/cluster/cluster_interface.sh job base --task Isaac-PosTracking-Flat-Anymal-D-Play-v0 --headless --enable_cameras --load_run "$run" --num_envs 8 --seed 100 --video --video_length 2000 --actor_critic_class_name ActorCritic
    
#     # Wait for 10 minutes (600 seconds) before running the next command
#     sleep 400
# done

# sleep 1800
# Loop through each load_run argument
for run in "${load_rnn_runs[@]}"; do

    # Execute the command with the current load_run argument and appropriate log_run_name
    ./docker/cluster/cluster_interface.sh job base --task Isaac-PosTracking-Flat-Anymal-D-Play-v0 --headless --enable_cameras --load_run "$run" --num_envs 8 --seed 100 --video --video_length 2000 --actor_critic_class_name ActorCriticRecurrent --checkpoint model_1200.pt
    
    # Wait for 10 minutes (600 seconds) before running the next command
    sleep 350
done
