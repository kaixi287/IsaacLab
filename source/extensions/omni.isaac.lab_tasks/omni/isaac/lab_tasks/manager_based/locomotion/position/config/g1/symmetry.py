# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# TODO implement symmetry for G1
import torch

from rsl_rl.env import VecEnv

# G1 joints and their corresponding indices:
#  0:  left_hip_pitch_joint
#  1:  right_hip_pitch_joint
#  2:  torso_joint
#  3:  left_hip_roll_joint
#  4:  right_hip_roll_joint
#  5:  left_shoulder_pitch_joint
#  6:  right_shoulder_pitch_joint
#  7:  left_hip_yaw_joint
#  8:  right_hip_yaw_joint
#  9:  left_shoulder_roll_joint
# 10:  right_shoulder_roll_joint
# 11:  left_knee_joint
# 12:  right_knee_joint
# 13:  left_shoulder_yaw_joint
# 14:  right_shoulder_yaw_joint
# 15:  left_ankle_pitch_joint
# 16:  right_ankle_pitch_joint
# 17:  left_elbow_pitch_joint
# 18:  right_elbow_pitch_joint
# 19:  left_ankle_roll_joint
# 20:  right_ankle_roll_joint
# 21:  left_elbow_roll_joint
# 22:  right_elbow_roll_joint
# 23:  left_five_joint
# 24:  left_three_joint
# 25:  left_zero_joint
# 26:  right_five_joint
# 27:  right_three_joint
# 28:  right_zero_joint
# 29:  left_six_joint
# 30:  left_four_joint
# 31:  left_one_joint
# 32:  right_six_joint
# 33:  right_four_joint
# 34:  right_one_joint
# 35:  left_two_joint
# 36:  right_two_joint


def _switch_arms_legs_lr(dof):
    dof_switched = torch.zeros_like(dof, device=dof.device)

    # Left leg <-> Right leg (hip pitch, hip roll, hip yaw, knee, ankle pitch, ankle roll)
    dof_switched[..., 0] = dof[..., 1]  # left_hip_pitch_joint <-> right_hip_pitch_joint
    dof_switched[..., 1] = dof[..., 0]

    dof_switched[..., 3] = dof[..., 4] * -1  # left_hip_roll_joint <-> right_hip_roll_joint (roll -> reverse sign)
    dof_switched[..., 4] = dof[..., 3] * -1

    dof_switched[..., 7] = dof[..., 8] * -1  # left_hip_yaw_joint <-> right_hip_yaw_joint (yaw -> reverse sign)
    dof_switched[..., 8] = dof[..., 7] * -1

    dof_switched[..., 11] = dof[..., 12]  # left_knee_joint <-> right_knee_joint
    dof_switched[..., 12] = dof[..., 11]

    dof_switched[..., 15] = dof[..., 16]  # left_ankle_pitch_joint <-> right_ankle_pitch_joint
    dof_switched[..., 16] = dof[..., 15]

    dof_switched[..., 19] = dof[..., 20] * -1  # left_ankle_roll_joint <-> right_ankle_roll_joint (roll -> reverse sign)
    dof_switched[..., 20] = dof[..., 19] * -1

    # Left arm <-> Right arm (shoulder pitch, shoulder roll, shoulder yaw, elbow pitch, elbow roll)
    dof_switched[..., 5] = dof[..., 6]  # left_shoulder_pitch_joint <-> right_shoulder_pitch_joint
    dof_switched[..., 6] = dof[..., 5]

    dof_switched[..., 9] = (
        dof[..., 10] * -1
    )  # left_shoulder_roll_joint <-> right_shoulder_roll_joint (roll -> reverse sign)
    dof_switched[..., 10] = dof[..., 9] * -1

    dof_switched[..., 13] = (
        dof[..., 14] * -1
    )  # left_shoulder_yaw_joint <-> right_shoulder_yaw_joint (yaw -> reverse sign)
    dof_switched[..., 14] = dof[..., 13] * -1

    dof_switched[..., 17] = dof[..., 18]  # left_elbow_pitch_joint <-> right_elbow_pitch_joint
    dof_switched[..., 18] = dof[..., 17]

    dof_switched[..., 21] = dof[..., 22] * -1  # left_elbow_roll_joint <-> right_elbow_roll_joint (roll -> reverse sign)
    dof_switched[..., 22] = dof[..., 21] * -1

    # Left hand <-> Right hand (fingers: zero, one, two, three, four, five, six)
    dof_switched[..., 23] = dof[..., 26]  # left_five_joint <-> right_five_joint
    dof_switched[..., 26] = dof[..., 23]

    dof_switched[..., 24] = dof[..., 27]  # left_three_joint <-> right_three_joint
    dof_switched[..., 27] = dof[..., 24]

    dof_switched[..., 25] = dof[..., 28]  # left_zero_joint <-> right_zero_joint
    dof_switched[..., 28] = dof[..., 25]

    dof_switched[..., 29] = dof[..., 32]  # left_six_joint <-> right_six_joint
    dof_switched[..., 32] = dof[..., 29]

    dof_switched[..., 30] = dof[..., 33]  # left_four_joint <-> right_four_joint
    dof_switched[..., 33] = dof[..., 30]

    dof_switched[..., 31] = dof[..., 34]  # left_one_joint <-> right_one_joint
    dof_switched[..., 34] = dof[..., 31]

    dof_switched[..., 35] = dof[..., 36]  # left_two_joint <-> right_two_joint
    dof_switched[..., 36] = dof[..., 35]

    return dof_switched


def _transform_obs_left_right(obs, has_height_scan=False):
    obs = obs.clone()
    # Flip lin vel y [1], ang vel x,z [3, 5], gravity y [7]
    obs[..., [1, 3, 5, 7]] *= -1
    # Flip velocity commands pos y [10]
    obs[..., 10] *= -1
    # Flip velocity commands ang yaw(z) [11]
    obs[..., 11] *= -1
    # dof pos
    obs[..., 12:49] = _switch_arms_legs_lr(obs[..., 12:49])
    # dof vel
    obs[..., 49:86] = _switch_arms_legs_lr(obs[..., 49:86])
    # last actions
    obs[..., 86:123] = _switch_arms_legs_lr(obs[..., 86:123])
    # TODO: correct height_scan flipping
    if has_height_scan:
        obs[..., 123:] = obs[..., 123:].view(*obs.shape[:-1], 21, 11).flip(dims=[-1]).view(*obs.shape[:-1], 21 * 11)
    return obs


def _transform_actions_left_right(actions):
    actions = actions.clone()
    actions[:] = _switch_arms_legs_lr(actions[:])
    return actions


@torch.no_grad()
def get_symmetric_states(
    obs: torch.Tensor | None = None,
    actions: torch.Tensor | None = None,
    env: VecEnv | None = None,
    is_critic: bool = False,
) -> tuple[torch.Tensor | None, torch.Tensor | None]:
    obs_aug, actions_aug = None, None

    if obs is not None:
        # Check if height_scan is enabled
        has_height_scan = False
        if env is not None:
            if is_critic and hasattr(env.cfg.observations, "critic"):
                has_height_scan = (
                    hasattr(env.cfg.observations.critic, "height_scan")
                    and env.cfg.observations.critic.height_scan is not None
                )
            else:
                has_height_scan = (
                    hasattr(env.cfg.observations.policy, "height_scan")
                    and env.cfg.observations.policy.height_scan is not None
                )

        num_envs = obs.shape[-2]
        obs_aug = torch.zeros(*obs.shape[:-2], num_envs * 2, obs.shape[-1], device=obs.device)
        obs_aug[..., :num_envs, :] = obs
        obs_aug[..., num_envs:, :] = _transform_obs_left_right(obs, has_height_scan)

    if actions is not None:
        num_envs = actions.shape[-2]
        actions_aug = torch.zeros(*actions.shape[:-2], num_envs * 2, actions.shape[-1], device=actions.device)
        actions_aug[..., :num_envs, :] = actions
        actions_aug[..., num_envs:, :] = _transform_actions_left_right(actions)

    return obs_aug, actions_aug
