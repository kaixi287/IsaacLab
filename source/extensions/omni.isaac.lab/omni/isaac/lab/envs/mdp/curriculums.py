# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Common functions that can be used to create curriculum for the learning environment.

The functions can be passed to the :class:`omni.isaac.lab.managers.CurriculumTermCfg` object to enable
the curriculum introduced by the function.
"""

from __future__ import annotations

import torch
from collections.abc import Sequence
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from omni.isaac.lab.envs import ManagerBasedRLEnv


def modify_reward_weight(env: ManagerBasedRLEnv, env_ids: Sequence[int], term_name: str, weight: float, num_steps: int):
    """Curriculum that modifies a reward weight a given number of steps.

    Args:
        env: The learning environment.
        env_ids: Not used since all environments are affected.
        term_name: The name of the reward term.
        weight: The weight of the reward term.
        num_steps: The number of steps after which the change should be applied.
    """
    if env.common_step_counter > num_steps:
        # obtain term settings
        term_cfg = env.reward_manager.get_term_cfg(term_name)
        # update term settings
        term_cfg.weight = weight
        env.reward_manager.set_term_cfg(term_name, term_cfg)

def modify_reward_weight_on_threshold(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int],
    term_name: str,
    weight: float,
    ref_term_name: str,
    threshold: float
):
    """Curriculum that modifies a reward weight when the reward term value exceeds a threshold.

    Args:
        env: The learning environment.
        term_name: The name of the reward term.
        weight: The new weight to apply to the reward term if the threshold is exceeded.
        ref_term_name: THe name of the reference reward term whose value is monitored to trigger the weight modification.
        threshold: The value threshold that triggers the weight change.
    """
    # Obtain the current value of the reward term
    current_term_value = torch.mean(env.reward_manager._episode_sums[ref_term_name])

    # Check if the term value exceeds the threshold
    if current_term_value > threshold:
        # Obtain current term settings
        term_cfg = env.reward_manager.get_term_cfg(term_name)
        # Update the weight of the reward term
        term_cfg.weight = weight
        # Apply the new settings
        env.reward_manager.set_term_cfg(term_name, term_cfg)
