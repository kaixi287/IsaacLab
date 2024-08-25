# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to play a checkpoint if an RL agent from RSL-RL."""

"""Launch Isaac Sim Simulator first."""

import argparse

from omni.isaac.lab.app import AppLauncher

# local imports
import cli_args  # isort: skip

# add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with RSL-RL.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during evaluation.")
parser.add_argument("--video_length", type=int, default=1000, help="Length of the recorded video (in steps).")
parser.add_argument("--video_interval", type=int, default=50, help="Interval between video recordings (in steps).")
parser.add_argument("--cpu", action="store_true", default=False, help="Use CPU pipeline.")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument("--log_root_path", type=str, default=None, help="Relative path of log root directory.")
parser.add_argument("--test_symmetry", action="store_true", default=False, help="Whether to test symmetry augmentation.")

# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import os
import torch
import sys
import numpy as np

from rsl_rl.runners import OnPolicyRunner

import omni.isaac.lab_tasks  # noqa: F401
from omni.isaac.lab_tasks.utils import get_checkpoint_path, parse_env_cfg
from omni.isaac.lab_tasks.utils.wrappers.rsl_rl import (
    RslRlOnPolicyRunnerCfg,
    RslRlVecEnvWrapper,
    export_policy_as_jit,
    export_policy_as_onnx,
)
if args_cli.test_symmetry:
    from omni.isaac.lab_tasks.manager_based.pose_tracking.config.anymal_d.symmetry import get_symmetric_states


def main():
    """Play with RSL-RL agent."""
    # parse configuration
    env_cfg = parse_env_cfg(
        args_cli.task, use_gpu=not args_cli.cpu, num_envs=args_cli.num_envs, use_fabric=not args_cli.disable_fabric
    )
    agent_cfg: RslRlOnPolicyRunnerCfg = cli_args.parse_rsl_rl_cfg(args_cli.task, args_cli)

    if args_cli.log_root_path is not None:
        log_root_path = args_cli.log_root_path
    else:
        # specify directory for logging experiments
        log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    # log_file = os.path.join(log_root_path, "evaluation.log")
    # sys.stdout = open(log_file, 'a')
    # sys.stderr = open(log_file, 'a')
    print(f"[INFO] Loading experiment from directory: {log_root_path}")
    resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)
    print(f"[INFO]: Loading model checkpoint from: {resume_path}")
    test_symmetry = args_cli.test_symmetry

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)
    # wrap for video recording
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_root_path, agent_cfg.load_run, "play_videos"),
            "step_trigger": lambda step: step % args_cli.video_interval == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during evaluation.")
        env = gym.wrappers.RecordVideo(env, **video_kwargs)
    # wrap around environment for rsl-rl
    env = RslRlVecEnvWrapper(env)

    # load previously trained model
    ppo_runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    ppo_runner.load(resume_path)
    print(f"[INFO]: Loading model checkpoint from: {resume_path}")

    # obtain the trained policy for inference
    policy = ppo_runner.get_inference_policy(device=env.unwrapped.device)

    # export policy to onnx/jit
    export_model_dir = os.path.join(os.path.dirname(resume_path), "exported")
    export_policy_as_jit(
        ppo_runner.alg.actor_critic, ppo_runner.obs_normalizer, path=export_model_dir, filename="policy.pt"
    )
    export_policy_as_onnx(
        ppo_runner.alg.actor_critic, normalizer=ppo_runner.obs_normalizer, path=export_model_dir, filename="policy.onnx"
    )

    # reset environment
    obs, _ = env.get_observations()

    # metrics
    num_envs = env.num_envs
    first_episode_rewards = np.zeros(num_envs)
    episode_completed = np.zeros(num_envs, dtype=bool)  # Track whether the first episode is completed for each env
    current_rewards = np.zeros(num_envs)
    current_lengths = np.zeros(num_envs)

    if test_symmetry:
        batch_size = num_envs // 4
        symmetry_loss = np.zeros(num_envs)
        episode_symmetry_loss = []

        # Get symmetric states for the initial observations and actions
        obs, _ = get_symmetric_states(obs=obs[:batch_size], env=env, is_critic=False)
        assert obs.shape[0] == num_envs
        
    step = 0
    num_episodes_recorded = 0

    # simulate environment
    while simulation_app.is_running():
        # run everything in inference mode
        with torch.inference_mode():
            actions = policy(obs)
            obs, rewards, dones, _ = env.step(actions)

            current_rewards += rewards.cpu().numpy()
            current_lengths += 1
            
            if test_symmetry:
                # Get symmetric observations and actions
                obs, sym_actions = get_symmetric_states(obs=obs[:batch_size], actions=actions[:batch_size], env=env)
                # compute the loss between predicted actions and symmetric actions
                mse_loss = torch.nn.MSELoss()
                loss = mse_loss(actions, sym_actions)
                symmetry_loss += loss.detach().cpu().numpy()

            if np.any(dones.cpu().numpy()):
                done_indices = np.where(dones.cpu().numpy())[0]
                for idx in done_indices:
                    if not episode_completed[idx]:
                        # Record the reward of the first episode
                        first_episode_rewards[idx] = current_rewards[idx]
                        episode_completed[idx] = True
                        num_episodes_recorded += 1

                    # Reset current rewards and lengths for the next episode
                    current_rewards[idx] = 0
                    current_lengths[idx] = 0
                    
                    if test_symmetry:
                        episode_symmetry_loss.append(symmetry_loss[idx])
                        symmetry_loss[idx] = 0

                # Print the progress
                string = f"Num Episodes Recorded: {num_episodes_recorded}/{num_envs}, Mean First Episode Reward (So Far): {np.mean(first_episode_rewards[episode_completed])}"
                if test_symmetry:
                    string += f", Mean Symmetry Loss (So Far over {len(episode_symmetry_loss)} episodes): {np.mean(episode_symmetry_loss)}"
                print(string)

            # Record video at specified intervals
            if args_cli.video and step % args_cli.video_interval == 0:
                env.reset()
                for _ in range(args_cli.video_length):
                    with torch.inference_mode():
                        actions = policy(obs)
                        obs, _, _, _ = env.step(actions)
            
            step += 1

    # Print final metrics for the first episodes
    print(f"Final Mean First Episode Reward: {np.mean(first_episode_rewards)}")
    print(f"Final Sum First Episode Reward: {np.sum(first_episode_rewards)}")
    print(f"Number of Recorded First Episodes: {num_episodes_recorded}")
    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()