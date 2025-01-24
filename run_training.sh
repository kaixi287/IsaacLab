#!/bin/bash

# Command template
# mlp_command="./docker/cluster/cluster_interface.sh job base --task Isaac-PosTracking-Flat-G1-v0 --headless --logger wandb --log_project_name g1_pos_tracking_payload_final --log_run_name mlp_aug --actor_critic_class ActorCritic"
mlp_command="./docker/cluster/cluster_interface.sh job base --task Isaac-PosTracking-Flat-G1-v0 --headless --logger wandb --log_project_name g1_pos_tracking_payload_concat --log_run_name mlp_rand_add_joint_obs --actor_critic_class ActorCritic"

# Loop through seeds 1 to 5
for seed in {20..22}; do
    # sleep 600
    # Execute the command with the current seed
    $mlp_command --seed $seed --run_name mlp_rand_$seed

    # Wait for 10 minutes (600 seconds) before running the next command
    sleep 10
done

# Command template
# rnn_command="./docker/cluster/cluster_interface.sh job base --task Isaac-PosTracking-Flat-G1-v0 --headless --logger wandb --log_project_name g1_pos_tracking_payload_final --log_run_name rnn_aug --actor_critic_class ActorCriticRecurrent"
rnn_command="./docker/cluster/cluster_interface.sh job base --task Isaac-PosTracking-Flat-G1-v0 --headless --logger wandb --log_project_name g1_pos_tracking_payload_concat --log_run_name rnn_rand_add_joint_obs --actor_critic_class ActorCriticRecurrent"

# Loop through seeds 1 to 5
for seed in {20..22}; do
    # sleep 100
    # Execute the command with the current seed
    $rnn_command --seed $seed --run_name rnn_rand_$seed

    # Wait for 10 minutes (600 seconds) before running the next command
    sleep 10
done

# mlp_runs=(
#     "2025-01-07_13-04-14_mlp_rand_joint_0"
#     "2025-01-07_13-04-58_mlp_rand_joint_1"
#     "2025-01-07_13-05-14_mlp_rand_joint_2"
#     "2025-01-07_13-05-13_mlp_rand_joint_3"
#     "2025-01-07_13-05-14_mlp_rand_joint_4"
#     "2025-01-07_13-05-14_mlp_rand_joint_5"
# )

# rnn_runs=(
#     "2025-01-07_13-07-46_rnn_rand_joint_0"
#     "2025-01-07_13-07-46_rnn_rand_joint_1"
#     "2025-01-07_13-09-40_rnn_rand_joint_2"
#     "2025-01-07_13-10-10_rnn_rand_joint_3"
#     "2025-01-07_13-10-45_rnn_rand_joint_4"
#     "2025-01-07_13-11-28_rnn_rand_joint_5"
# )

# for run in "${mlp_runs[@]}"; do
#     # Extract the seed from the run name (last two digits after 'joint_')
#     seed=$(echo "$run" | grep -oP 'joint_\K\d+')

#     # Execute the command with the extracted seed
#     ./docker/cluster/cluster_interface.sh job base \
#         --task Isaac-PosTracking-Flat-Anymal-D-v0 \
#         --headless \
#         --logger wandb \
#         --log_project_name anynmal_pos_tracking_joint_concat_orig_feetair \
#         --log_run_name mlp_rand \
#         --actor_critic_class ActorCritic \
#         --seed "$seed" \
#         --run_name "mlp_rand_joint_$seed" \
#         --resume True \
#         --load_run "$run"

#     sleep 30
# done

# for run in "${rnn_runs[@]}"; do
#     # Extract the seed from the run name (last two digits after 'joint_')
#     seed=$(echo "$run" | grep -oP 'joint_\K\d+')

#     # Execute the command with the extracted seed
#     ./docker/cluster/cluster_interface.sh job base \
#         --task Isaac-PosTracking-Flat-Anymal-D-v0 \
#         --headless \
#         --logger wandb \
#         --log_project_name anynmal_pos_tracking_joint_concat_orig_feetair \
#         --log_run_name rnn_rand \
#         --actor_critic_class ActorCriticRecurrent \
#         --seed "$seed" \
#         --run_name "rnn_rand_joint_$seed" \
#         --resume True \
#         --load_run "$run"

#     sleep 30
# done
