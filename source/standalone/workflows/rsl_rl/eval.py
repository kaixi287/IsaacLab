# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to play a checkpoint if an RL agent from RSL-RL."""

"""Launch Isaac Sim Simulator first."""

import argparse
import glob

from omni.isaac.lab.app import AppLauncher

# local imports
import cli_args  # isort: skip

# add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with RSL-RL.")
parser.add_argument("--cpu", action="store_true", default=False, help="Use CPU pipeline.")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument("--log_root_path", type=str, default=None, help="Relative path of log root directory.")

# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

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
            # value = torch.mean(infotensor)
            value = torch.sum(infotensor)/locs["num_episodes"]
            # log to logger and terminal
            if "/" in key:
                writer.add_scalar(key, value, locs["i"])
                ep_string += f"""{f'{key}:':>{pad}} {value:.4f}\n"""
            else:
                writer.add_scalar("Episode/" + key, value, locs["i"])
                ep_string += f"""{f'Mean episode {key}:':>{pad}} {value:.4f}\n"""
    
    # -- Training
    if len(locs["rewbuffer"]) > 0:
        # everything else
        writer.add_scalar("Eval/mean_reward", statistics.mean(locs["rewbuffer"]), locs["i"])
        writer.add_scalar("Eval/mean_episode_length", statistics.mean(locs["lenbuffer"]), locs["i"])
        writer.add_scalar("Eval/success_rate", locs["success_count"]/locs["num_episodes"], locs["i"])
        writer.add_scalar("Eval/tracking_failure_rate", locs["tracking_failure_count"]/locs["num_episodes"], locs["i"])
        writer.add_scalar("Eval/early_termination_rate", locs["early_termination_count"]/locs["num_episodes"], locs["i"])


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
    print(f"[INFO] Loading experiment from directory: {log_root_path}")
    resume_paths = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode=None)
    eval_log_dir = os.path.join(log_root_path, agent_cfg.load_run, "evaluation")
    
    # wrap around environment for rsl-rl
    env = RslRlVecEnvWrapper(env)

    if args_cli.logger == "wandb":
        log_dir = eval_log_dir
        writer = WandbSummaryWriter(log_dir=log_dir, flush_secs=10, cfg=agent_cfg.to_dict())
    else:
        log_dir = None
    
    for resume_path in resume_paths:
        checkpoint_name = os.path.basename(resume_path)
        print(f"[INFO]: Loading model checkpoint from: {resume_path}")
        
        iter = int(checkpoint_name.split('_')[1].split('.')[0])
        
        with torch.inference_mode():
            # set seed of the environment
            env.seed(agent_cfg.seed)
            env.reset()

        # load the checkpoint
        ppo_runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=log_dir, device=agent_cfg.device)
        ppo_runner.load(resume_path)

        # obtain the trained policy for inference
        policy = ppo_runner.get_inference_policy(device=env.unwrapped.device)

        # reset environment
        obs, _ = env.get_observations()

        # metrics
        success_count = 0
        tracking_failure_count = 0
        early_termination_count = 0
        num_episodes = 0
        ep_infos = []
        rewbuffer = deque(maxlen=1000)
        lenbuffer = deque(maxlen=1000)
        cur_reward_sum = torch.zeros(env.num_envs, dtype=torch.float, device=obs.device)
        cur_episode_length = torch.zeros(env.num_envs, dtype=torch.float, device=obs.device)

        # simulate environment
        while num_episodes < 1000:
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

                    num_episodes += new_ids.numel()
                
                    if new_ids.numel() > 0:
                        success_count += infos["success_count"]
                        tracking_failure_count = infos["tracking_failure_count"]
                        early_termination_count = infos["early_termination_count"]
                        # Book keeping
                        if "episode" in infos:
                            ep_infos.append(infos["episode"])
                        elif "log" in infos:
                            ep_infos.append(infos["log"])
        
        if iter==0:
            iter = 1
                                    
        # Log infos
        locs = {
            "ep_infos": ep_infos,
            "rewbuffer": rewbuffer,
            "lenbuffer": lenbuffer,
            "success_count": success_count,
            "tracking_failure_count": tracking_failure_count,
            "early_termination_count": early_termination_count,
            "num_episodes": num_episodes,
            "i": iter
        }
        log(writer, locs)
        print(f"Checkpoint: {checkpoint_name}, Collected episodes: {num_episodes}, Mean Reward: {statistics.mean(rewbuffer):.4f}, Mean Episode Length: {statistics.mean(lenbuffer):.4f}, Success Rate: {success_count / num_episodes:.4f}")

    # close the simulator
    env.close()
    
    if log_dir is not None:
        writer.stop()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()