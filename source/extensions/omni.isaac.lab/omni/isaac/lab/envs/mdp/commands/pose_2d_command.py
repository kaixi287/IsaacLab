# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Sub-module containing command generators for the 2D-pose for locomotion tasks."""

from __future__ import annotations

import torch
from collections.abc import Sequence
from typing import TYPE_CHECKING

from omni.isaac.lab.assets import Articulation
from omni.isaac.lab.managers import CommandTerm
from omni.isaac.lab.markers import VisualizationMarkers
from omni.isaac.lab.terrains import TerrainImporter
from omni.isaac.lab.utils.math import quat_from_euler_xyz, quat_rotate_inverse, wrap_to_pi, yaw_quat

if TYPE_CHECKING:
    from omni.isaac.lab.envs import ManagerBasedEnv

    from .commands_cfg import TerrainBasedPose2dCommandCfg, UniformPose2dCommandCfg


class UniformPose2dCommand(CommandTerm):
    """Command generator that generates pose commands containing a 3-D position and heading.

    The command generator samples uniform 2D positions around the environment origin. It sets
    the height of the position command to the default root height of the robot. The heading
    command is either set to point towards the target or is sampled uniformly.
    This can be configured through the :attr:`Pose2dCommandCfg.simple_heading` parameter in
    the configuration.
    """

    cfg: UniformPose2dCommandCfg
    """Configuration for the command generator."""

    def __init__(self, cfg: UniformPose2dCommandCfg, env: ManagerBasedEnv):
        """Initialize the command generator class.

        Args:
            cfg: The configuration parameters for the command generator.
            env: The environment object.
        """
        # initialize the base class
        super().__init__(cfg, env)

        # obtain the robot and terrain assets
        # -- robot
        self.robot: Articulation = env.scene[cfg.asset_name]

        # crete buffers to store the command
        # -- commands: (x, y, z, heading)
        self._pos_command_w = torch.zeros(self.num_envs, 3, device=self.device)
        self.pos_command_b = torch.zeros_like(self._pos_command_w)
        self._heading_command_w = torch.zeros(self.num_envs, device=self.device)

        # -- metrics
        self.metrics["error_pos"] = torch.zeros(self.num_envs, device=self.device)

        # create buffers for heading command
        if self.cfg.include_heading:
            self.heading_command_b = torch.zeros_like(self._heading_command_w)
            self.metrics["error_heading"] = torch.zeros(self.num_envs, device=self.device)

    def __str__(self) -> str:
        msg = "PositionCommand:\n"
        msg += f"\tCommand dimension: {tuple(self.command.shape[1:])}\n"
        msg += f"\tResampling time range: {self.cfg.resampling_time_range}"
        return msg

    """
    Properties
    """

    @property
    def command(self) -> torch.Tensor:
        """The desired 2D-pose in base frame. Shape is (num_envs, 4)."""
        if self.cfg.include_heading:
            return torch.cat([self.pos_command_b, self.heading_command_b.unsqueeze(1)], dim=1)
        else:
            return self.pos_command_b

    @property
    def pos_command_w(self) -> torch.Tensor:
        return self._pos_command_w

    @property
    def heading_command_w(self) -> torch.Tensor:
        return self._heading_command_w

    """
    Implementation specific functions.
    """

    def _update_metrics(self):
        # logs data
        self.metrics["error_pos_2d"] = torch.norm(self._pos_command_w[:, :2] - self.robot.data.root_pos_w[:, :2], dim=1)
        if self.cfg.include_heading:
            self.metrics["error_heading"] = torch.abs(wrap_to_pi(self._heading_command_w - self.robot.data.heading_w))

    def _resample_command(self, env_ids: Sequence[int]):
        # obtain env origins for the environments
        self._pos_command_w[env_ids] = self._env.scene.env_origins[env_ids]
        # offset the position command by the default root height
        self._pos_command_w[env_ids, 2] += self.robot.data.default_root_state[env_ids, 2]
        # set heading command to current robot heading
        if self.cfg.include_heading:
            self._heading_command_w[env_ids] = self.robot.data.heading_w[env_ids]

        # Create mask for environments to resample commands
        command_resample_mask = torch.full((len(env_ids),), True, dtype=torch.bool, device=self.device)
        if self.cfg.standing_command_prob > 0.0:
            command_resample_mask = torch.rand(len(env_ids), device=self.device) >= self.cfg.standing_command_prob

        if command_resample_mask.any():
            env_ids_tensor = torch.tensor(env_ids, device=self.device)
            resample_env_ids = env_ids_tensor[command_resample_mask]
            r = torch.empty(len(resample_env_ids), device=self.device)
            if self.cfg.polar_sampling:
                # Sample random radii and angles for polar coordinates
                radius = r.uniform_(*self.cfg.polar_ranges.radius)
                angle = r.uniform_(*self.cfg.polar_ranges.theta)

                # radius = torch.tensor([4] * len(env_ids), device=self.device, dtype=torch.float)
                # angle = torch.tensor([-torch.pi/2, torch.pi/2, -torch.pi/2, torch.pi/2, -torch.pi/2, torch.pi/2, -torch.pi/2, torch.pi/2],
                #             device=self.device, dtype=torch.float)
                # angle = torch.tensor([torch.pi/2, torch.pi/4, torch.pi/2, 3*torch.pi/4, -torch.pi/2, -torch.pi/4, -torch.pi/2, -3*torch.pi/4],
                #     device=self.device, dtype=torch.float)

                # Apply the offsets to the position commands
                self._pos_command_w[resample_env_ids, 0] += radius * torch.cos(angle)
                self._pos_command_w[resample_env_ids, 1] += radius * torch.sin(angle)
            else:
                # offset the position command by the current root position
                self._pos_command_w[resample_env_ids, 0] += r.uniform_(*self.cfg.ranges.pos_x)
                self._pos_command_w[resample_env_ids, 1] += r.uniform_(*self.cfg.ranges.pos_y)

            if self.cfg.include_heading:
                if self.cfg.simple_heading:
                    # set heading command to point towards target
                    target_vec = self._pos_command_w[resample_env_ids] - self.robot.data.root_pos_w[resample_env_ids]
                    target_direction = torch.atan2(target_vec[:, 1], target_vec[:, 0])
                    flipped_target_direction = wrap_to_pi(target_direction + torch.pi)

                    # compute errors to find the closest direction to the current heading
                    # this is done to avoid the discontinuity at the -pi/pi boundary
                    curr_to_target = wrap_to_pi(target_direction - self.robot.data.heading_w[resample_env_ids]).abs()
                    curr_to_flipped_target = wrap_to_pi(
                        flipped_target_direction - self.robot.data.heading_w[resample_env_ids]
                    ).abs()

                    # set the heading command to the closest direction
                    self._heading_command_w[resample_env_ids] = torch.where(
                        curr_to_target < curr_to_flipped_target,
                        target_direction,
                        flipped_target_direction,
                    )
                else:
                    # random heading command
                    self._heading_command_w[resample_env_ids] = r.uniform_(*self.cfg.ranges.heading)

    def _update_command(self):
        """Re-target the position command to the current root state."""
        target_vec = self._pos_command_w - self.robot.data.root_pos_w[:, :3]
        self.pos_command_b[:] = quat_rotate_inverse(yaw_quat(self.robot.data.root_quat_w), target_vec)
        if self.cfg.include_heading:
            self.heading_command_b[:] = wrap_to_pi(self._heading_command_w - self.robot.data.heading_w)

    def _set_debug_vis_impl(self, debug_vis: bool):
        if debug_vis:
            # create markers if necessary for the first tome
            if not hasattr(self, "goal_pose_visualizer"):
                self.goal_pose_visualizer = VisualizationMarkers(self.cfg.goal_pose_visualizer_cfg)
                self.goal_pose_visualizer.set_visibility(True)

            if not hasattr(self, "curr_pose_visualizer"):
                self.curr_pose_visualizer = VisualizationMarkers(self.cfg.curr_pose_visualizer_cfg)
                self.curr_pose_visualizer.set_visibility(True)

            if self.cfg.connection_visualizer_cfg is not None and not hasattr(self, "connection_visualizer"):
                self.connection_visualizer = VisualizationMarkers(self.cfg.connection_visualizer_cfg)
                self.connection_visualizer.set_visibility(True)
        else:
            if hasattr(self, "goal_pose_visualizer"):
                self.goal_pose_visualizer.set_visibility(False)
            if hasattr(self, "curr_pose_visualizer"):
                self.curr_pose_visualizer.set_visibility(False)
            if hasattr(self, "connection_visualizer"):
                self.connection_visualizer.set_visibility(False)

    def _debug_vis_callback(self, event):
        # update the box marker
        _pos_command_w = self._pos_command_w.clone()
        _pos_command_w[:, 2] += 0.5
        self.goal_pose_visualizer.visualize(
            translations=_pos_command_w,
            orientations=quat_from_euler_xyz(
                torch.zeros_like(self.heading_command_w),
                torch.zeros_like(self.heading_command_w),
                self.heading_command_w,
            ),
        )

        # Visualize the current pose
        if not self.robot.is_initialized:
            return
        # get marker location
        # -- base state, set the marker height to the command marker height
        base_pos_w = self.robot.data.root_pos_w.clone()
        base_pos_w[:, 2] = _pos_command_w[:, 2]
        # -- resolve the scales and quaternions
        base_quat_w = self.robot.data.root_quat_w
        # display markers
        self.curr_pose_visualizer.visualize(base_pos_w, base_quat_w)

        # Visualize the connection between the robot and the goal
        if hasattr(self, "connection_visualizer"):
            direction_vector = _pos_command_w - base_pos_w  # Vector from robot to target
            distance = torch.norm(direction_vector, dim=-1, keepdim=True)  # Compute distance
            yaw_angles = torch.atan2(direction_vector[:, 1], direction_vector[:, 0])

            # Compute orientation using the provided quat_from_euler_xyz function
            connection_orientations = quat_from_euler_xyz(
                roll=torch.zeros_like(yaw_angles),  # No roll
                pitch=torch.zeros_like(yaw_angles),  # No pitch
                yaw=yaw_angles,  # Yaw angle derived from direction vector
            )

            # Scale the arrow to span the distance between robot and target
            connection_scales = torch.cat(
                [distance, torch.full_like(distance, 0.03), torch.full_like(distance, 0.03)], dim=-1
            )

            self.connection_visualizer.visualize(
                translations=base_pos_w + (_pos_command_w - base_pos_w) * 0.25,  # Midpoint between robot and target
                orientations=connection_orientations,
                scales=connection_scales,
            )


class TerrainBasedPose2dCommand(UniformPose2dCommand):
    """Command generator that generates pose commands based on the terrain.

    This command generator samples the position commands from the valid patches of the terrain.
    The heading commands are either set to point towards the target or are sampled uniformly.

    It expects the terrain to have a valid flat patches under the key 'target'.
    """

    cfg: TerrainBasedPose2dCommandCfg
    """Configuration for the command generator."""

    def __init__(self, cfg: TerrainBasedPose2dCommandCfg, env: ManagerBasedEnv):
        # initialize the base class
        super().__init__(cfg, env)

        # obtain the terrain asset
        self.terrain: TerrainImporter = env.scene["terrain"]

        # obtain the valid targets from the terrain
        if "target" not in self.terrain.flat_patches:
            raise RuntimeError(
                "The terrain-based command generator requires a valid flat patch under 'target' in the terrain."
                f" Found: {list(self.terrain.flat_patches.keys())}"
            )
        # valid targets: (terrain_level, terrain_type, num_patches, 3)
        self.valid_targets: torch.Tensor = self.terrain.flat_patches["target"]

    def _resample_command(self, env_ids: Sequence[int]):
        # sample new position targets from the terrain
        ids = torch.randint(0, self.valid_targets.shape[2], size=(len(env_ids),), device=self.device)
        self._pos_command_w[env_ids] = self.valid_targets[
            self.terrain.terrain_levels[env_ids], self.terrain.terrain_types[env_ids], ids
        ]
        # offset the position command by the current root height
        self._pos_command_w[env_ids, 2] += self.robot.data.default_root_state[env_ids, 2]

        if self.cfg.simple_heading:
            # set heading command to point towards target
            target_vec = self._pos_command_w[env_ids] - self.robot.data.root_pos_w[env_ids]
            target_direction = torch.atan2(target_vec[:, 1], target_vec[:, 0])
            flipped_target_direction = wrap_to_pi(target_direction + torch.pi)

            # compute errors to find the closest direction to the current heading
            # this is done to avoid the discontinuity at the -pi/pi boundary
            curr_to_target = wrap_to_pi(target_direction - self.robot.data.heading_w[env_ids]).abs()
            curr_to_flipped_target = wrap_to_pi(flipped_target_direction - self.robot.data.heading_w[env_ids]).abs()

            # set the heading command to the closest direction
            self._heading_command_w[env_ids] = torch.where(
                curr_to_target < curr_to_flipped_target,
                target_direction,
                flipped_target_direction,
            )
        else:
            # random heading command
            r = torch.empty(len(env_ids), device=self.device)
            self._heading_command_w[env_ids] = r.uniform_(*self.cfg.ranges.heading)
