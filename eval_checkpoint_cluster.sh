#!/bin/bash

# Set the base command parameters
TASK="Isaac-Velocity-Flat-G1-v0"
HEADLESS="--headless"
ENABLE_CAMERAS="--enable_cameras"
NUM_ENVS=1000
SEED=15
MAX_ITERATIONS=500
ACTOR_CRITIC_CLASS="ActorCriticRecurrent"
RESUME="--resume True"
LOG_ROOT_PATH="../experiments/g1_vel/rnn_aug_logs"
LOGGER="--logger wandb"
LOG_PROJECT_NAME="g1_vel_payload_rnn_aug_ood3"
LOG_RUN_NAME="rnn_aug_ood"

# Path to the script
SCRIPT_PATH="source/standalone/workflows/rsl_rl/train.py"

LOAD_RUN_FOLDERS=(
    "2024-10-15_23-21-57_rnn_aug_payload"
    "2024-10-15_23-27-44_rnn_aug_payload"
    "2024-10-15_23-32-08_rnn_aug_payload"
    "2024-10-15_23-37-53_rnn_aug_payload"
    "2024-10-15_23-43-12_rnn_aug_payload"
    # Add more folders as needed
)

# Path to the script
SCRIPT_PATH="source/standalone/workflows/rsl_rl/train.py"

# Loop through each LOAD_RUN folder
for LOAD_RUN in "${LOAD_RUN_FOLDERS[@]}"; do
    echo "Processing LOAD_RUN: $LOAD_RUN"

    # Loop through all checkpoints in the current LOAD_RUN folder
    for CHECKPOINT_PATH in "$LOG_ROOT_PATH/$LOAD_RUN"/model_*.pt; do
        CHECKPOINT=$(basename "$CHECKPOINT_PATH")
        echo "Executing with checkpoint: $CHECKPOINT"

        # ./isaaclab.sh -p "$SCRIPT_PATH" --task "$TASK" $HEADLESS $ENABLE_CAMERAS --num_envs "$NUM_ENVS" --seed "$SEED" --max_iterations "$MAX_ITERATIONS" --actor_critic_class_name "$ACTOR_CRITIC_CLASS" $RESUME --load_run "$LOAD_RUN" --log_root_path "$LOG_ROOT_PATH" $LOGGER --log_project_name "$LOG_PROJECT_NAME" --log_run_name "$LOG_RUN_NAME" --checkpoint "$CHECKPOINT" 2>/dev/null
        ./docker/cluster/cluster_interface.sh job base --task "$TASK" $HEADLESS $ENABLE_CAMERAS --num_envs "$NUM_ENVS" --seed "$SEED" --max_iterations "$MAX_ITERATIONS" --actor_critic_class_name "$ACTOR_CRITIC_CLASS" $RESUME --load_run "$LOAD_RUN" $LOGGER --log_project_name "$LOG_PROJECT_NAME" --log_run_name "$LOG_RUN_NAME" --checkpoint "$CHECKPOINT" --disable_update True 2>/dev/null

        sleep 200

        # # Cleanup step
        # sync  # Flush file system buffers
        # sleep 2  # Wait for 2 seconds to ensure all resources are released
    done
done
