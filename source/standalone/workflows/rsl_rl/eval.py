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
import statistics
from collections import deque

from rsl_rl.runners import OnPolicyRunner
from rsl_rl.utils.wandb_utils import WandbSummaryWriter

import omni.isaac.lab_tasks  # noqa: F401
from omni.isaac.lab_tasks.utils import get_checkpoint_path, parse_env_cfg
from omni.isaac.lab_tasks.utils.wrappers.rsl_rl import (
    RslRlOnPolicyRunnerCfg,
    RslRlVecEnvWrapper,
    export_policy_as_jit,
    export_policy_as_onnx,
)

def log(writer: WandbSummaryWriter, locs: dict, width: int = 80, pad: int = 35):
    # -- Episode info
    ep_string = ""
    if locs["ep_infos"]:
        for key in locs["ep_infos"][0]:
            infotensor = torch.tensor([], device=torch.device('cpu'))
            for ep_info in locs["ep_infos"]:
                # handle scalar and zero dimensional tensor infos
                if key not in ep_info:
                    continue
                if not isinstance(ep_info[key], torch.Tensor):
                    ep_info[key] = torch.Tensor([ep_info[key]], device=infotensor.device)
                if len(ep_info[key].shape) == 0:
                    ep_info[key] = ep_info[key].unsqueeze(0)
                infotensor = torch.cat((infotensor, ep_info[key].to(infotensor.device)))
            value = torch.mean(infotensor)
            # log to logger and terminal
            if "/" in key:
                writer.add_scalar(key, value, locs["num_episodes"])
                ep_string += f"""{f'{key}:':>{pad}} {value:.4f}\n"""
            else:
                writer.add_scalar("Episode/" + key, value, locs["num_episodes"])
                ep_string += f"""{f'Mean episode {key}:':>{pad}} {value:.4f}\n"""
    
    # -- Training
    if len(locs["rewbuffer"]) > 0:
        # everything else
        writer.add_scalar("Eval/mean_reward", statistics.mean(locs["rewbuffer"]), locs["num_episodes"])
        writer.add_scalar("Eval/mean_episode_length", statistics.mean(locs["lenbuffer"]), locs["num_episodes"])


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

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)
    eval_log_dir = os.path.join(log_root_path, agent_cfg.load_run, "evaluation")
    # wrap for video recording
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(eval_log_dir, "videos"),
            "step_trigger": lambda step: step % args_cli.video_interval == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during evaluation.")
        env = gym.wrappers.RecordVideo(env, **video_kwargs)
    # wrap around environment for rsl-rl
    env = RslRlVecEnvWrapper(env)

    if args_cli.logger == "wandb":
        log_dir = eval_log_dir
        writer = WandbSummaryWriter(log_dir=log_dir, flush_secs=10, cfg=agent_cfg.to_dict())
    else:
        log_dir = None

    # load previously trained model
    ppo_runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=log_dir, device=agent_cfg.device)
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
    # Book keeping
    step = 0
    num_episodes = 0
    ep_infos = []
    rewbuffer = deque(maxlen=1000)
    lenbuffer = deque(maxlen=1000)
    cur_reward_sum = torch.zeros(env.num_envs, dtype=torch.float, device=obs.device)
    cur_episode_length = torch.zeros(env.num_envs, dtype=torch.float, device=obs.device)

    # simulate environment
    while simulation_app.is_running():
        # run everything in inference mode
        with torch.inference_mode():
            actions = policy(obs)
            obs, rewards, dones, infos = env.step(actions)

            if log_dir is not None:
                cur_reward_sum += rewards
                cur_episode_length += 1
                new_ids = (dones > 0).nonzero(as_tuple=False)
                rewbuffer.extend(cur_reward_sum[new_ids][:, 0].cpu().numpy().tolist())
                lenbuffer.extend(cur_episode_length[new_ids][:, 0].cpu().numpy().tolist())
                cur_reward_sum[new_ids] = 0
                cur_episode_length[new_ids] = 0

                if len(new_ids) > 0:
                    num_episodes += len(new_ids)
                    # Book keeping
                    if "episode" in infos:
                        ep_infos.append(infos["episode"])
                    elif "log" in infos:
                        ep_infos.append(infos["log"])

                    # Only log infos if there are newly completed episodes
                    locs = {
                        "ep_infos": ep_infos,
                        "rewbuffer": rewbuffer,
                        "lenbuffer": lenbuffer,
                        "num_episodes": num_episodes  # or another variable that tracks the current iteration
                    }
                    log(writer, locs)

                if num_episodes == 1000:
                    print(f"Step: {step}, Mean Reward: {statistics.mean(rewbuffer):.4f}, Mean Episode Length: {statistics.mean(lenbuffer):.4f}")
                    break

            # Record video at specified intervals
            if args_cli.video and step % args_cli.video_interval == 0:
                env.reset()
                for _ in range(args_cli.video_length):
                    with torch.inference_mode():
                        actions = policy(obs)
                        obs, _, _, _ = env.step(actions)
            
            step += 1

    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()