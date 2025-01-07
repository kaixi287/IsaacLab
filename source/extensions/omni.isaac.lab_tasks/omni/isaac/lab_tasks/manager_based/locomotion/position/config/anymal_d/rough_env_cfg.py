# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
from omni.isaac.lab.utils import configclass

from .flat_env_cfg import AnymalDPosTrackingFlatEnvCfg

##
# Pre-defined configs
##
from omni.isaac.lab.terrains.config.rough import ROUGH_TERRAINS_CFG  # isort: skip


@configclass
class AnymalDPosTrackingRoughEnvCfg(AnymalDPosTrackingFlatEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # revert rewards
        # self.rewards.flat_orientation_l2.weight = -0.5
        self.rewards.dof_torques_l2.weight = -1.0e-5
        self.rewards.feet_air_time.weight = 0.125
        # revert terrain
        self.scene.terrain.terrain_type = "generator"
        self.scene.terrain.terrain_generator = ROUGH_TERRAINS_CFG
        self.scene.terrain.terrain_generator.curriculum = False
        self.scene.terrain.terrain_generator.difficulty_range = (0.0, 0.0)
        self.scene.terrain.terrain_generator.sub_terrains["pyramid_stairs"].proportion = 0.0
        self.scene.terrain.terrain_generator.sub_terrains["pyramid_stairs_inv"].proportion = 0.0
        self.scene.terrain.terrain_generator.sub_terrains["boxes"].proportion = 0.0
        self.scene.terrain.terrain_generator.sub_terrains["random_rough"].proportion = 1.0
        self.scene.terrain.terrain_generator.sub_terrains["random_rough"].noise_range = (0.0, 0.05)
        self.scene.terrain.terrain_generator.sub_terrains["hf_pyramid_slope"].proportion = 0.0
        self.scene.terrain.terrain_generator.sub_terrains["hf_pyramid_slope_inv"].proportion = 0.0


class AnymalDPosTrackingRoughEnvCfg_PLAY(AnymalDPosTrackingRoughEnvCfg):
    def __post_init__(self) -> None:
        # post init of parent
        super().__post_init__()

        # Set to eval mode
        self.eval_mode = True

        # make a smaller scene for play
        # self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        # disable randomization for play
        self.observations.policy.enable_corruption = False
        self.terminations.base_contact = None
        self.terminations.illegal_force = None
        self.terminations.illegal_force_feet = None
        # self.events.disable_joint.params["prob_no_disable"] = 0.0
        if getattr(self.events, "disable_joint", None) is not None:
            self.scene.robot.debug_vis = True
            self.scene.robot.in_distribution_joint_ids = [0, 4, 8]
            self.events.disable_joint.params["prob_no_disable"] = 0.0
        if getattr(self.events, "add_payload_to_body", None) is not None:
            self.scene.robot.debug_vis = True
            self.scene.robot.in_distribution_external_force_positions = [(0.0, 0.4), (0.0, 0.08)]
