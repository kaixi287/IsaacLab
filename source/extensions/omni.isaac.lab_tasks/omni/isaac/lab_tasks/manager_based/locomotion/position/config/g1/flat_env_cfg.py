# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from omni.isaac.lab.managers import RewardTermCfg as RewTerm
from omni.isaac.lab.managers import SceneEntityCfg
from omni.isaac.lab.utils import configclass

import omni.isaac.lab_tasks.manager_based.locomotion.position.mdp as mdp
from omni.isaac.lab_tasks.manager_based.locomotion.position.position_env_cfg import PosTrackingEnvCfg, RewardsCfg

##
# Pre-defined configs
##
from omni.isaac.lab_assets import G1_MINIMAL_CFG  # isort: skip

##
# MDP settings
##


@configclass
class G1Rewards(RewardsCfg):
    feet_air_time = RewTerm(
        func=mdp.feet_air_time_positive_biped,
        weight=0.75,
        params={
            "command_name": "pose_command",
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_ankle_roll_link"),
            "threshold": 0.4,
        },
    )
    feet_slide = RewTerm(
        func=mdp.feet_slide,
        weight=-0.1,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_ankle_roll_link"),
            "asset_cfg": SceneEntityCfg("robot", body_names=".*_ankle_roll_link"),
        },
    )
    # Penalize knee and hip joint accelerations
    dof_acc_l2 = RewTerm(
        func=mdp.joint_acc_l2,
        weight=-1.0e-7,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*_hip_.*", ".*_knee_joint"])},
    )
    # base_acc = RewTerm(
    #     func=mdp.base_acc, weight=-0.001, params={"asset_cfg": SceneEntityCfg("robot", body_names=["torso_link"])}
    # )
    # applied_torque_limits = RewTerm(func=mdp.applied_torque_limits, weight=-0.05)
    # dof_vel_l2 = RewTerm(
    #     func=mdp.joint_vel_l2,
    #     weight=-0.0005,
    #     params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*_hip_.*", ".*_knee_joint"]
    #     )})
    # dof_vel_limits = RewTerm(
    #     func=mdp.joint_vel_limits,
    #     weight=-1.0,
    #     params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*_ankle_pitch_joint", ".*_ankle_roll_joint"]), "soft_ratio": 0.9})
    # Penalize ankle joint limits
    dof_pos_limits = RewTerm(
        func=mdp.joint_pos_limits,
        weight=-1.0,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*_ankle_pitch_joint", ".*_ankle_roll_joint"])},
    )
    # Penalize deviation from default of the joints that are not essential for locomotion
    joint_deviation_hip = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-0.02,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*_hip_yaw_joint", ".*_hip_roll_joint"])},
    )
    joint_deviation_arms = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-0.1,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                joint_names=[
                    ".*_shoulder_pitch_joint",
                    ".*_shoulder_roll_joint",
                    ".*_shoulder_yaw_joint",
                    ".*_elbow_pitch_joint",
                    ".*_elbow_roll_joint",
                ],
            )
        },
    )
    joint_deviation_fingers = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-0.05,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                joint_names=[
                    ".*_five_joint",
                    ".*_three_joint",
                    ".*_six_joint",
                    ".*_four_joint",
                    ".*_zero_joint",
                    ".*_one_joint",
                    ".*_two_joint",
                ],
            )
        },
    )
    joint_deviation_torso = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-0.1,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names="torso_joint")},
    )
    # feet_acc = RewTerm(
    #     func=mdp.feet_acc, weight=-0.002, params={"asset_cfg": SceneEntityCfg("robot", body_names=".*_ankle_roll_link")}
    # )

    # dont_move = RewTerm(
    #     func=mdp.stand_still, weight=-0.05, params={"duration": 1.0, "distance_threshold": 0.1, "command_name": "pose_command"}
    # )
    # stand_still = RewTerm(
    #     func=mdp.stand_still_pose_reward,
    #     weight=50.0,
    #     params={"duration": 3.0, "distance_threshold": 0.05, "command_name": "pose_command"},
    # )
    # ang_vel_stand_still = RewTerm(mdp.ang_vel_stand_still, weight=-0.1, params={"duration": 3.0, "distance_threshold": 0.01, "command_name": "pose_command"})


##
# Environment configuration
##


@configclass
class G1PosTrackingFlatEnvCfg(PosTrackingEnvCfg):
    """Configuration for the locomotion position-tracking environment."""

    rewards: RewardsCfg = G1Rewards()

    def __post_init__(self):
        """Post initialization."""
        # post init of parent
        super().__post_init__()
        # Scene
        self.scene.robot = G1_MINIMAL_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        # change terrain to flat
        self.scene.terrain.terrain_type = "plane"
        self.scene.terrain.terrain_generator = None
        # no height scan
        self.scene.height_scanner = None
        self.observations.policy.height_scan = None
        self.observations.critic.height_scan = None

        self.events.push_robot = None
        self.events.add_base_mass = None
        self.events.reset_robot_joints.params["position_range"] = (1.0, 1.0)
        self.events.reset_base.params = {
            "pose_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "yaw": (-3.14, 3.14)},
            "velocity_range": {
                "x": (0.0, 0.0),
                "y": (0.0, 0.0),
                "z": (0.0, 0.0),
                "roll": (0.0, 0.0),
                "pitch": (0.0, 0.0),
                "yaw": (0.0, 0.0),
            },
        }

        # Termination
        self.terminations.base_contact.params = {
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names="torso_link"),
            "threshold": 1.0,
        }

        # Rewards
        self.rewards.lin_vel_z_l2.weight = -0.2
        self.rewards.ang_vel_xy_l2.weight = -0.005
        self.rewards.action_rate_l2.weight = -0.005
        self.rewards.flat_orientation_l2.weight = -1.0
        self.rewards.dof_torques_l2.weight = -2.0e-6
        self.rewards.dof_torques_l2.params["asset_cfg"] = SceneEntityCfg(
            "robot", joint_names=[".*_hip_.*", ".*_knee_joint", ".*_ankle_.*"]
        )

        # Reduce the weight for hip joint deviation if hip joint is disabled
        # if getattr(self.events, "disable_joint", None) is not None:
        #     self.rewards.joint_deviation_hip.weight = -0.01
        # self.rewards.dont_wait = None
        # self.rewards.stand_still.params["duration"] = 3.0
        # self.rewards.stand_still.params["distance_threshold"] = 0.01
        # self.rewards.stand_still.weight = -0.05
        # self.rewards.move_in_direction.weight = 1.5
        # self.rewards.tracking_pos.weight = 15.0


class G1PosTrackingFlatEnvCfg_PLAY(G1PosTrackingFlatEnvCfg):
    def __post_init__(self) -> None:
        # post init of parent
        super().__post_init__()

        # Set to eval mode
        self.eval_mode = True
        if getattr(self.events, "disable_joint", None) is not None:
            self.events.disable_joint.params["prob_no_disable"] = 0.0
            self.scene.robot.debug_vis = True
            self.scene.robot.in_distribution_joint_ids = [0, 3, 7]

        if getattr(self.events, "add_payload_to_body", None) is not None:
            self.scene.robot.debug_vis = True
            self.scene.robot.in_distribution_external_force_positions = [(-0.07, 0.07), (0.08, 0.11)]

        # make a smaller scene for play
        # self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        # disable randomization for play
        self.observations.policy.enable_corruption = False
