# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from omni.isaac.lab.utils import configclass

from .rough_env_cfg import AnymalDRoughEnvCfg


@configclass
class AnymalDFlatEnvCfg(AnymalDRoughEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # override rewards
        self.rewards.flat_orientation_l2.weight = -5.0
        self.rewards.dof_torques_l2.weight = -2.5e-5
        self.rewards.feet_air_time.weight = 0.5
        # change terrain to flat
        self.scene.terrain.terrain_type = "plane"
        self.scene.terrain.terrain_generator = None
        # no height scan
        self.scene.height_scanner = None
        self.observations.policy.height_scan = None
        # no terrain curriculum
        self.curriculum.terrain_levels = None
        if getattr(self.events, "add_payload_to_body", None) is not None:
            self.events.add_payload_to_body.params["z_position_range"] = (0.1325, 0.1325)


class AnymalDFlatEnvCfg_PLAY(AnymalDFlatEnvCfg):
    def __post_init__(self) -> None:
        # post init of parent
        super().__post_init__()

        # Set to eval mode
        self.eval_mode = True

        # make a smaller scene for play
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        # disable randomization for play
        self.observations.policy.enable_corruption = False
        # remove random pushing event
        self.events.base_external_force_torque = None
        self.events.push_robot = None
        if getattr(self.events, "disable_joint", None) is not None:
            self.scene.robot.debug_vis = True
            self.scene.robot.in_distribution_joint_ids = [0, 4, 8]
            self.events.disable_joint.params["prob_no_disable"] = 0.0
        if getattr(self.events, "add_payload_to_body", None) is not None:
            self.scene.robot.debug_vis = True
            self.scene.robot.in_distribution_external_force_positions = [(0.0, 0.4), (0.0, 0.08)]
