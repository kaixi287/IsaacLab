# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import gymnasium as gym

from . import agents, pos_tracking_env_cfg

##
# Register Gym environments.
##

gym.register(
    id="Isaac-PosTracking-Flat-Anymal-D-v0",
    entry_point="omni.isaac.lab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": pos_tracking_env_cfg.PosTrackingEnvCfg,
        "rsl_rl_cfg_entry_point": agents.rsl_rl_cfg.PosTrackingEnvPPORunnerCfg,
    },
)

gym.register(
    id="Isaac-PosTracking-Flat-Anymal-D-Play-v0",
    entry_point="omni.isaac.lab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": pos_tracking_env_cfg.PosTrackingEnvCfg_PLAY,
        "rsl_rl_cfg_entry_point": agents.rsl_rl_cfg.PosTrackingEnvPPORunnerCfg,
    },
)
