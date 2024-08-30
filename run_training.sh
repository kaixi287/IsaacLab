#!/bin/bash

# Command template
command="./docker/container.sh job --task Isaac-PosTracking-Flat-Anymal-D-v0 --headless --logger wandb --run_name mlp_rand --log_project_name position_tracking3 --log_run_name mlp_rand"

# Loop through seeds 1 to 5
for seed in {1..5}; do
    # Execute the command with the current seed
    $command --seed $seed
    
    # Wait for 10 minutes (600 seconds) before running the next command
    sleep 400
done