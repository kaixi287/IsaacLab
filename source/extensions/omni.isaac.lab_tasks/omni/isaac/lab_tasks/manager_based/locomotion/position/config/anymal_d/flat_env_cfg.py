# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from omni.isaac.lab.managers import RewardTermCfg as RewTerm
from omni.isaac.lab.managers import SceneEntityCfg
from omni.isaac.lab.managers import TerminationTermCfg as DoneTerm
from omni.isaac.lab.utils import configclass

import omni.isaac.lab_tasks.manager_based.locomotion.position.mdp as mdp
from omni.isaac.lab_tasks.manager_based.locomotion.position.position_env_cfg import (
    PosTrackingEnvCfg,
    RewardsCfg,
    TerminationsCfg,
)

##
# Pre-defined configs
##
from omni.isaac.lab_assets.anymal import ANYMAL_D_CFG  # isort: skip


##
# MDP settings
##


@configclass
class AnymalDTerminationsCfg(TerminationsCfg):
    """Termination terms for ANYMAL_D."""

    illegal_force_feet = DoneTerm(
        func=mdp.illegal_force,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*FOOT"), "max_force": 1500},
    )
    illegal_force = DoneTerm(
        func=mdp.illegal_force,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*"), "max_force": 5000},
    )


@configclass
class AnymalDRewards(RewardsCfg):
    # TODO: still needed?
    feet_acc = RewTerm(
        func=mdp.feet_acc, weight=-0.002, params={"asset_cfg": SceneEntityCfg("robot", body_names=[".*_FOOT"])}
    )


##
# Environment configuration
##


@configclass
class AnymalDPosTrackingFlatEnvCfg(PosTrackingEnvCfg):
    """Configuration for the locomotion velocity-tracking environment."""

    terminations: TerminationsCfg = AnymalDTerminationsCfg()
    rewards: RewardsCfg = AnymalDRewards()

    def __post_init__(self):
        """Post initialization."""
        # post init of parent
        super().__post_init__()
        # switch robot to anymal-d
        self.scene.robot = ANYMAL_D_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        # change terrain to flat
        self.scene.terrain.terrain_type = "plane"
        self.scene.terrain.terrain_generator = None
        # no height scan
        self.scene.height_scanner = None
        self.observations.policy.height_scan = None
        self.observations.critic.height_scan = None


class AnymalDPosTrackingFlatEnvCfg_PLAY(AnymalDPosTrackingFlatEnvCfg):
    def __post_init__(self) -> None:
        # post init of parent
        super().__post_init__()

        # make a smaller scene for play
        # self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        # disable randomization for play
        self.observations.policy.enable_corruption = False
        self.terminations.base_contact = None
        self.terminations.illegal_force = None
        self.terminations.illegal_force_feet = None
        if getattr(self.events, "disable_joint", None) is not None:
            self.events.disable_joint.params["prob_no_disable"] = 0.0
            self.scene.robot.debug_vis = True
            self.scene.robot.in_distribution_joint_ids = [0, 1, 3, 4, 7, 8]

        if getattr(self.events, "add_payload_to_base", None) is not None:
            self.scene.robot.debug_vis = True
            self.scene.robot.in_distribution_external_force_positions = [(0.0, 0.4), (0.0, 0.08)]
