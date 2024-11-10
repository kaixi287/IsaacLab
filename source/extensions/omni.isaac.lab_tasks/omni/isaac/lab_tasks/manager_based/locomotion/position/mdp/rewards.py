# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

import omni.isaac.lab.utils.math as math_utils
from omni.isaac.lab.assets import Articulation, RigidObject
from omni.isaac.lab.managers import SceneEntityCfg
from omni.isaac.lab.managers.manager_base import ManagerTermBase  # noqa: F401
from omni.isaac.lab.managers.manager_term_cfg import RewardTermCfg  # noqa: F401
from omni.isaac.lab.sensors import ContactSensor

if TYPE_CHECKING:
    from omni.isaac.lab.envs import ManagerBasedRLEnv
    from omni.isaac.lab.managers.command_manager import CommandTerm


def _command_duration_mask(env: ManagerBasedRLEnv, duration: float, command_name: str):
    command: CommandTerm = env.command_manager.get_term(command_name)
    mask = command.time_left <= duration
    return mask / duration


def _initial_command_duration_mask(env: ManagerBasedRLEnv, duration: float, command_name: str):
    command: CommandTerm = env.command_manager.get_term(command_name)
    mask = command.time_elapsed <= duration
    return mask / duration


def tracking_pos(
    env: ManagerBasedRLEnv, duration: float, command_name: str, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
):
    command = env.command_manager.get_term(command_name)

    asset: RigidObject = env.scene[asset_cfg.name]
    distance = torch.norm(command.pos_command_w[:, :2] - asset.data.root_pos_w[:, :2], dim=1)

    return (1.0 / (1.0 + torch.square(distance))) * _command_duration_mask(env, duration, command_name)


def tracking_pos2(
    env: ManagerBasedRLEnv, duration: float, command_name: str, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Reward for reaching the target location."""
    command = env.command_manager.get_term(command_name)

    asset: RigidObject = env.scene[asset_cfg.name]
    distance = torch.norm(command.pos_command_w[:, :2] - asset.data.root_pos_w[:, :2], dim=1)
    return (1 - 0.5 * distance) * _command_duration_mask(env, duration, command_name)


def tracking_heading(
    env: ManagerBasedRLEnv,
    duration: float,
    command_name: str,
    max_pos_distance: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
):
    command = env.command_manager.get_term(command_name)
    asset: RigidObject = env.scene[asset_cfg.name]

    distance = torch.abs(math_utils.wrap_to_pi(command.heading_command_w - asset.data.heading_w))
    position_distance = torch.norm(command.pos_command_w[:, :2] - asset.data.root_pos_w[:, :2], dim=1)
    return (
        (1.0 / (1.0 + torch.square(distance)))
        * (position_distance < max_pos_distance)
        * _command_duration_mask(env, duration, command_name)
    )


def tracking_heading2(
    env: ManagerBasedRLEnv,
    duration: float,
    command_name: str,
    max_pos_distance: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    command = env.command_manager.get_term(command_name)
    asset: RigidObject = env.scene[asset_cfg.name]

    distance = torch.abs(math_utils.wrap_to_pi(command.heading_command_w - asset.data.heading_w))
    position_distance = torch.norm(command.pos_command_w[:, :2] - asset.data.root_pos_w[:, :2], dim=1)
    return (
        (1 - 0.5 * distance)
        * (position_distance < max_pos_distance)
        * _command_duration_mask(env, duration, command_name)
    )


"""
Root penalties.
"""


def lin_vel_z_l2(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize z-axis base linear velocity using L2-kernel."""
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    return torch.square(asset.data.root_lin_vel_b[:, 2])


def ang_vel_xy_l2(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize xy-axis base angular velocity using L2-kernel."""
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    return torch.sum(torch.square(asset.data.root_ang_vel_b[:, :2]), dim=1)


def flat_orientation_l2(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize non-flat base orientation using L2-kernel.

    This is computed by penalizing the xy-components of the projected gravity vector.
    """
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    return torch.sum(torch.square(asset.data.projected_gravity_b[:, :2]), dim=1)


def base_acc(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names="base")):
    """Penalize high base accelerations.

    The function computes the acceleration penalty based on the difference in the root state
    velocities and angular velocities over the time step `env.dt`.
    """
    asset: RigidObject = env.scene[asset_cfg.name]
    # Compute linear acceleration
    linear_acc_sum = torch.sum(torch.square(asset.data.body_acc_w[:, asset_cfg.body_ids, :3]), dim=-1).squeeze(1)
    # Compute angular acceleration
    angular_acc_sum = 0.02 * torch.sum(torch.square(asset.data.body_acc_w[:, asset_cfg.body_ids, 3:]), dim=-1).squeeze(
        1
    )

    return linear_acc_sum + angular_acc_sum


"""
Joint penalties.
"""


def joint_torques_l2(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize joint torques applied on the articulation using L2-kernel.

    NOTE: Only the joints configured in :attr:`asset_cfg.joint_ids` will have their joint torques contribute to the L2 norm.
    """
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    return torch.sum(torch.square(asset.data.applied_torque[:, asset_cfg.joint_ids]), dim=1)


def joint_acc_l2(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize joint accelerations on the articulation using L2-kernel.

    NOTE: Only the joints configured in :attr:`asset_cfg.joint_ids` will have their joint accelerations contribute to the L2 norm.
    """
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    return torch.sum(torch.square(asset.data.joint_acc[:, asset_cfg.joint_ids]), dim=1)


def joint_vel_l2(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize joint velocities on the articulation using L2 squared kernel.

    NOTE: Only the joints configured in :attr:`asset_cfg.joint_ids` will have their joint velocities contribute to the term.
    """
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    return torch.sum(torch.square(asset.data.joint_vel[:, asset_cfg.joint_ids]), dim=1)


def joint_pos_limits(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize joint positions if they cross the soft limits.

    This is computed as a sum of the absolute value of the difference between the joint position and the soft limits.
    """
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    # compute out of limits constraints
    out_of_limits = -(
        asset.data.joint_pos[:, asset_cfg.joint_ids] - asset.data.soft_joint_pos_limits[:, asset_cfg.joint_ids, 0]
    ).clip(max=0.0)
    out_of_limits += (
        asset.data.joint_pos[:, asset_cfg.joint_ids] - asset.data.soft_joint_pos_limits[:, asset_cfg.joint_ids, 1]
    ).clip(min=0.0)
    return torch.sum(out_of_limits, dim=1)


"""
Feet penalties.
"""


def feet_acc(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names=[".*_FOOT"])
) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    return torch.sum(torch.norm(asset.data.body_acc_w[:, asset_cfg.body_ids], dim=-1), dim=1)


def feet_balance(
    env: ManagerBasedRLEnv,
    duration: float,
    command_name: str,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names=[".*_FOOT"]),
) -> torch.Tensor:
    """
    Penalize foot positions that are unbalanced with respect to the base center if the robot should stand still.
    """
    asset: Articulation = env.scene[asset_cfg.name]

    root_position = asset.data.root_pos_w
    foot_positions = asset.data.body_pos_w[:, asset_cfg.body_ids]  # Extract only the positions (first 3 values)

    # Calculate distances of each foot from the base center
    distances_from_center = torch.norm(foot_positions - root_position.unsqueeze(1), dim=-1)

    # Calculate the imbalance as the variance in these distances (higher variance indicates more imbalance)
    imbalance = torch.var(distances_from_center, dim=1)

    command = env.command_manager.get_term(command_name)
    should_stand = torch.norm(command.pos_command_w[:, :2] - asset.data.root_pos_w[:, :2], dim=1) <= 0.1
    # if command.cfg.include_heading:
    #     should_stand &= torch.abs(command.heading_command_w - asset.data.heading_w) < 0.5
    return imbalance * should_stand * _command_duration_mask(env, duration, command_name)


def feet_slide(env, sensor_cfg: SceneEntityCfg, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    # Penalize feet sliding
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    contacts = contact_sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids, :].norm(dim=-1).max(dim=1)[0] > 1.0
    asset = env.scene[asset_cfg.name]
    body_vel = asset.data.body_lin_vel_w[:, asset_cfg.body_ids, :2]
    reward = torch.sum(body_vel.norm(dim=-1) * contacts, dim=1)
    return reward


def feet_air_time_positive_biped(env, command_name: str, threshold: float, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """Reward long steps taken by the feet for bipeds.

    This function rewards the agent for taking steps up to a specified threshold and also keep one foot at
    a time in the air.

    If the commands are small (i.e. the agent is not supposed to take a step), then the reward is zero.
    """
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    # compute the reward
    air_time = contact_sensor.data.current_air_time[:, sensor_cfg.body_ids]
    contact_time = contact_sensor.data.current_contact_time[:, sensor_cfg.body_ids]
    in_contact = contact_time > 0.0
    in_mode_time = torch.where(in_contact, contact_time, air_time)
    single_stance = torch.sum(in_contact.int(), dim=1) == 1
    reward = torch.min(torch.where(single_stance.unsqueeze(-1), in_mode_time, 0.0), dim=1)[0]
    reward = torch.clamp(reward, max=threshold)
    # no reward for zero command
    reward *= torch.norm(env.command_manager.get_command(command_name)[:, :2], dim=1) > 0.1
    return reward


"""
Action penalties.
"""


def applied_torque_limits(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize applied torques if they cross the limits.

    This is computed as a sum of the absolute value of the difference between the applied torques and the limits.

    .. caution::
        Currently, this only works for explicit actuators since we manually compute the applied torques.
        For implicit actuators, we currently cannot retrieve the applied torques from the physics engine.
    """
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    # compute out of limits constraints
    # TODO: We need to fix this to support implicit joints.
    out_of_limits = torch.abs(
        asset.data.applied_torque[:, asset_cfg.joint_ids] - asset.data.computed_torque[:, asset_cfg.joint_ids]
    )
    return torch.sum(out_of_limits, dim=1)


def action_rate_l2(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Penalize the rate of change of the actions using L2-kernel."""
    return torch.sum(torch.square(env.action_manager.action - env.action_manager.prev_action), dim=1)


def action_rate_huber(env: ManagerBasedRLEnv, delta: float = 0.5) -> torch.Tensor:
    """
    Penalize the rate of change of the actions using Huber loss.

    Args:
    - env: The environment object containing action data.
    - delta: The Huber loss threshold. Default is 1.0.

    Returns:
    - Huber loss penalty for the action rate.
    """
    # Calculate the difference between the current and previous actions
    action_diff = env.action_manager.action - env.action_manager.prev_action
    abs_action_diff = torch.abs(action_diff)

    # Apply Huber loss conditionally based on delta threshold
    quadratic_term = 0.5 * torch.square(action_diff)  # Quadratic penalty for small changes
    linear_term = delta * (abs_action_diff - 0.3 * delta)  # Linear penalty for larger changes

    # Use torch.where to apply quadratic penalty where |diff| <= delta, else linear penalty
    huber_loss = torch.where(abs_action_diff <= delta, quadratic_term, linear_term)

    # Sum over all dimensions to get the total action rate penalty
    return torch.sum(huber_loss, dim=1)


def action_l2(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Penalize the actions using L2-kernel."""
    return torch.sum(torch.square(env.action_manager.action), dim=1)


"""
Contact sensor.
"""


def undesired_contacts(env: ManagerBasedRLEnv, threshold: float, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """Penalize undesired contacts as the number of violations that are above a threshold."""
    # extract the used quantities (to enable type-hinting)
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    # check if contact force is above threshold
    net_contact_forces = contact_sensor.data.net_forces_w_history
    is_contact = torch.max(torch.norm(net_contact_forces[:, :, sensor_cfg.body_ids], dim=-1), dim=1)[0] > threshold
    # sum over contacts for each environment
    return torch.sum(is_contact, dim=1)


def contact_forces(env: ManagerBasedRLEnv, threshold: float, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """Penalize contact forces as the amount of violations of the net contact force."""
    # extract the used quantities (to enable type-hinting)
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    net_contact_forces = contact_sensor.data.net_forces_w_history
    # compute the violation
    violation = torch.max(torch.norm(net_contact_forces[:, :, sensor_cfg.body_ids], dim=-1), dim=1)[0] - threshold
    # compute the penalty
    return torch.sum(violation.clip(min=0.0), dim=1)


def contact_forces_sq(env: ManagerBasedRLEnv, max_contact_force: float, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """Penalize high contact forces with a squared penalty for violations."""
    # Extract the used quantities (to enable type-hinting)
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    net_contact_forces = contact_sensor.data.net_forces_w_history
    # Compute the norm of the contact forces
    contact_forces = torch.norm(net_contact_forces[:, :, sensor_cfg.body_ids], dim=-1)
    # Clip the contact forces to a maximum value of 2 * max_contact_force
    clipped_contact_forces = contact_forces.clip(max=2.0 * max_contact_force)
    # Compute the squared penalty for forces exceeding the max_contact_force threshold
    violation = torch.square((clipped_contact_forces - max_contact_force).clip(min=0.0))
    # Sum the penalties across the selected bodies and timesteps
    return torch.sum(violation, dim=1)


def collision(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """Penalize collisions on selected bodies using the history of contact forces."""
    # Extract the used quantities (to enable type-hinting)
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    net_contact_forces = contact_sensor.data.net_forces_w_history
    # Compute the collision force
    collision_force = torch.max(torch.norm(net_contact_forces[:, :, sensor_cfg.body_ids], dim=-1), dim=1)[0]
    # Apply conditional clipping
    collision_force = torch.where(collision_force > 1.0, collision_force.clip(min=200.0), collision_force)

    return torch.sum(collision_force, dim=(1)) / 200.0


# -- exploration reward
def move_in_direction(
    env: ManagerBasedRLEnv,
    command_name: str,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    epsilon: float = 1e-8,
) -> torch.Tensor:
    """Encourage exploration towards the target."""
    command = env.command_manager.get_command(command_name)
    des_pos_b = command[:, :2]
    vel_target = des_pos_b / (torch.norm(des_pos_b, dim=1).unsqueeze(1) + epsilon)

    asset: RigidObject = env.scene[asset_cfg.name]
    x_dot_b = asset.data.root_lin_vel_b[:, :2]
    vel = x_dot_b / (torch.norm(x_dot_b, dim=1).unsqueeze(1) + epsilon)

    return vel[:, 0] * vel_target[:, 0] + vel[:, 1] * vel_target[:, 1]


# -- stalling penalty
def dont_wait(
    env: ManagerBasedRLEnv, min_vel: str, command_name: str, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Penalize stalling."""
    command = env.command_manager.get_term(command_name)

    asset: RigidObject = env.scene[asset_cfg.name]
    far_away = torch.norm(command.pos_command_w[:, :2] - asset.data.root_pos_w[:, :2], dim=1) > 0.25
    waiting = torch.norm(asset.data.root_lin_vel_w[:, :2], dim=1) < min_vel

    return far_away * waiting


def stand_still_pose(
    env: ManagerBasedRLEnv,
    duration: float,
    command_name: str,
    distance_threshold: float = 0.1,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
):
    command = env.command_manager.get_term(command_name)

    asset: Articulation = env.scene[asset_cfg.name]
    should_stand = torch.norm(command.pos_command_w[:, :2] - asset.data.root_pos_w[:, :2], dim=1) <= distance_threshold
    # if command.cfg.include_heading:
    #     should_stand &= torch.abs(command.heading_command_w - asset.data.heading_w) < 0.5
    return (
        torch.sum(torch.square(asset.data.joint_pos - asset.data.default_joint_pos), dim=1)
        * should_stand
        * _command_duration_mask(env, duration, command_name)
    )


def stand_still_pose_reward(
    env: ManagerBasedRLEnv,
    duration: float,
    command_name: str,
    distance_threshold: float = 0.1,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
):
    command = env.command_manager.get_term(command_name)

    asset: Articulation = env.scene[asset_cfg.name]
    should_stand = torch.norm(command.pos_command_w[:, :2] - asset.data.root_pos_w[:, :2], dim=1) <= distance_threshold

    # Calculate deviation from default joint positions
    joint_deviation = torch.sum(torch.square(asset.data.joint_pos - asset.data.default_joint_pos), dim=1)

    # Invert deviation to create a reward, rewarding closer joint positions
    reward = torch.exp(-joint_deviation)  # The closer to default, the higher the reward
    # reward = 1 / (1 + joint_deviation)  # The closer to default, the higher the reward

    return reward * should_stand * _command_duration_mask(env, duration, command_name)


def stand_still_vel(
    env: ManagerBasedRLEnv,
    duration: float,
    command_name: str,
    distance_threshold: float = 0.1,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
):
    command = env.command_manager.get_term(command_name)

    asset: Articulation = env.scene[asset_cfg.name]
    should_stand = torch.norm(command.pos_command_w[:, :2] - asset.data.root_pos_w[:, :2], dim=1) <= distance_threshold
    # if command.cfg.include_heading:
    #     should_stand &= torch.abs(command.heading_command_w - asset.data.heading_w) < 0.5
    return (
        torch.sum(torch.square(asset.data.joint_vel[:, asset_cfg.joint_ids]), dim=1)
        * should_stand
        * _command_duration_mask(env, duration, command_name)
    )


def ang_vel_stand_still(
    env: ManagerBasedRLEnv,
    duration: float,
    command_name: str,
    distance_threshold: float = 0.1,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalize xy-axis base angular velocity using L2-kernel."""
    # extract the used quantities (to enable type-hinting)
    command = env.command_manager.get_term(command_name)

    asset: Articulation = env.scene[asset_cfg.name]
    should_stand = torch.norm(command.pos_command_w[:, :2] - asset.data.root_pos_w[:, :2], dim=1) <= distance_threshold

    return (
        torch.square(asset.data.root_ang_vel_b[:, 2])
        * should_stand
        * _command_duration_mask(env, duration, command_name)
    )


# -- time efficiency reward
def time_efficiency_reward(
    env: ManagerBasedRLEnv, command_name: str, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """
    Reward for reaching the target quickly.
    """
    command = env.command_manager.get_term(command_name)
    asset: Articulation = env.scene[asset_cfg.name]
    # Calculate the distance to the target
    position_reached = torch.norm(command.pos_command_w[:, :2] - asset.data.root_pos_w[:, :2], dim=1) <= 0.1

    return command.time_left / env.max_episode_length_s * position_reached
