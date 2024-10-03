#!/bin/bash
load_mlp_runs=(
    "2024-09-10_16-42-55_mlp_rand_no_mid"
    "2024-09-10_22-28-20_mlp_rand_no_mid"
    "2024-09-10_17-03-53_mlp_rand_no_mid"
)

# Array of load_run arguments
load_rnn_runs=(
    "2024-09-10_17-14-10_rnn_rand_no_mid"
    "2024-09-10_17-24-24_rnn_rand_no_mid"
    "2024-09-10_17-35-11_rnn_rand_no_mid"
)

# Loop through each load_run argument
for run in "${load_mlp_runs[@]}"; do

    # Execute the command with the current load_run argument and appropriate log_run_name
    ./docker/cluster/cluster_interface.sh job base --task Isaac-PosTracking-Flat-Anymal-D-Play-v0 --headless --enable_cameras --load_run "$run" --num_envs 8 --seed 100 --video --video_length 2000 --actor_critic_class_name ActorCritic
    
    # Wait for 10 minutes (600 seconds) before running the next command
    sleep 350
done

# sleep 1800
# Loop through each load_run argument
for run in "${load_rnn_runs[@]}"; do

    # Execute the command with the current load_run argument and appropriate log_run_name
    ./docker/cluster/cluster_interface.sh job base --task Isaac-PosTracking-Flat-Anymal-D-Play-v0 --headless --enable_cameras --load_run "$run" --num_envs 8 --seed 100 --video --video_length 2000 --actor_critic_class_name ActorCriticRecurrent --checkpoint model_1200.pt
    
    # Wait for 10 minutes (600 seconds) before running the next command
    sleep 350
done
