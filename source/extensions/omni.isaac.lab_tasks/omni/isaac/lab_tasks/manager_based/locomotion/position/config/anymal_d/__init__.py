# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import gymnasium as gym

from . import agents, flat_env_cfg

##
# Register Gym environments.
##

gym.register(
    id="Isaac-PosTracking-Flat-Anymal-D-v0",
    entry_point="omni.isaac.lab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": flat_env_cfg.AnymalDPosTrackingFlatEnvCfg,
        "rsl_rl_cfg_entry_point": agents.rsl_rl_ppo_cfg.AnymalDPosTrackingEnvPPORunnerCfg,
    },
)

gym.register(
    id="Isaac-PosTracking-Flat-Anymal-D-Play-v0",
    entry_point="omni.isaac.lab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": flat_env_cfg.AnymalDPosTrackingFlatEnvCfg_PLAY,
        "rsl_rl_cfg_entry_point": agents.rsl_rl_ppo_cfg.AnymalDPosTrackingEnvPPORunnerCfg,
    },
)
