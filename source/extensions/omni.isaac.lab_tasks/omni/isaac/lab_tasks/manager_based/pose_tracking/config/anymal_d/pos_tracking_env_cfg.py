# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import math
from dataclasses import MISSING

import omni.isaac.lab.sim as sim_utils
from omni.isaac.lab.assets import ArticulationCfg, AssetBaseCfg
from omni.isaac.lab.envs import ManagerBasedRLEnvCfg
from omni.isaac.lab.managers import CurriculumTermCfg as CurrTerm
from omni.isaac.lab.managers import EventTermCfg as EventTerm
from omni.isaac.lab.managers import ObservationGroupCfg as ObsGroup
from omni.isaac.lab.managers import ObservationTermCfg as ObsTerm
from omni.isaac.lab.managers import RewardTermCfg as RewTerm
from omni.isaac.lab.managers import SceneEntityCfg
from omni.isaac.lab.managers import TerminationTermCfg as DoneTerm
from omni.isaac.lab.scene import InteractiveSceneCfg
from omni.isaac.lab.sensors import ContactSensorCfg, RayCasterCfg, patterns
from omni.isaac.lab.terrains import TerrainImporterCfg
from omni.isaac.lab.utils import configclass
from omni.isaac.lab.utils.assets import ISAAC_NUCLEUS_DIR, ISAACLAB_NUCLEUS_DIR
from omni.isaac.lab.utils.noise import AdditiveUniformNoiseCfg as Unoise

import omni.isaac.lab_tasks.manager_based.pose_tracking.mdp as mdp
import omni.isaac.lab_tasks.manager_based.navigation.mdp as mdp_nav
from omni.isaac.lab_tasks.manager_based.locomotion.velocity.config.anymal_d.flat_env_cfg import AnymalDFlatEnvCfg

LOW_LEVEL_ENV_CFG = AnymalDFlatEnvCfg()

##
# Pre-defined configs
##
from omni.isaac.lab.terrains.config.rough import ROUGH_TERRAINS_CFG  # isort: skip


##
# MDP settings
##


@configclass
class CommandsCfg:
    """Command terms for the MDP."""

    pose_command = mdp.UniformPose2dCommandCfg(
        asset_name="robot",
        simple_heading=False,
        resampling_time_range=(8.0, 8.0),
        debug_vis=True,
        polar_sampling=True,
        ranges=mdp.UniformPose2dCommandCfg.Ranges(pos_x=(-3.0, 3.0), pos_y=(-3.0, 3.0), heading=(-math.pi, math.pi)),
        polar_ranges=mdp.UniformPose2dCommandCfg.PolarRanges(radius=(1.0, 5.0), theta=(-math.pi, math.pi), heading=(-math.pi, math.pi)),
        include_heading=False
    )


@configclass
class ActionsCfg:
    """Action specifications for the MDP."""

    joint_pos = mdp.JointPositionActionCfg(asset_name="robot", joint_names=[".*"], scale=0.5, use_default_offset=True)


@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""

        # observation terms (order preserved)
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel, noise=Unoise(n_min=-0.1, n_max=0.1))
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, noise=Unoise(n_min=-0.2, n_max=0.2))
        projected_gravity = ObsTerm(
            func=mdp.projected_gravity,
            noise=Unoise(n_min=-0.05, n_max=0.05),
        )
        # pose_commands = ObsTerm(func=mdp.generated_commands, params={"command_name": "pose_command"})
        pos_commands = ObsTerm(func=mdp.pos_commands, params={"command_name": "pose_command"})
        # heading_commands = ObsTerm(func=mdp.heading_commands_sin, params={"command_name": "pose_command"})
        time_to_target = ObsTerm(func=mdp.time_to_target, params={"command_name": "pose_command"}, noise=Unoise(n_min=-0.1, n_max=0.1))
        joint_pos = ObsTerm(func=mdp.joint_pos_rel, noise=Unoise(n_min=-0.01, n_max=0.01))
        joint_vel = ObsTerm(func=mdp.joint_vel_rel, noise=Unoise(n_min=-1.5, n_max=1.5))
        actions = ObsTerm(func=mdp.last_action)
        height_scan = ObsTerm(
            func=mdp.height_scan,
            params={"sensor_cfg": SceneEntityCfg("height_scanner")},
            noise=Unoise(n_min=-0.1, n_max=0.1),
            clip=(-1.0, 1.0),
        )
        
        def __post_init__(self, enable_corruption=True, concatenate_terms=True):
            self.enable_corruption = enable_corruption
            self.concatenate_terms = concatenate_terms

    # observation groups
    policy: PolicyCfg = PolicyCfg()
    critic: PolicyCfg = PolicyCfg(enable_corruption=False)


@configclass
class EventCfg:
    """Configuration for events."""

    # startup
    physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "static_friction_range": (0.8, 0.8),
            "dynamic_friction_range": (0.6, 0.6),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 64,
        },
    )

    add_base_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="base"),
            "mass_distribution_params": (-5.0, 5.0),
            "operation": "add",
        },
    )

    # reset
    reset_base = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "yaw": (-3.14, 3.14)},
            "velocity_range": {
                "x": (-0.5, 0.5),
                "y": (-0.5, 0.5),
                "z": (-0.5, 0.5),
                "roll": (-0.5, 0.5),
                "pitch": (-0.5, 0.5),
                "yaw": (-0.5, 0.5),
            },
        },
    )

    reset_robot_joints = EventTerm(
        func=mdp.reset_joints_by_scale,
        mode="reset",
        params={
            "position_range": (0.5, 1.5),
            "velocity_range": (0.0, 0.0),
        },
    )

    block_joint = EventTerm(
        func=mdp.block_joint,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "joint_to_block": -1, # Index of joint to disable
            "prob_no_block": 0.2,
        },
    )

@configclass
class RewardsCfg:
    """Reward terms for the MDP."""

    # -- task
    tracking_pos = RewTerm(
        func=mdp.tracking_pos2,
        weight=10.0,
        params={"duration": 3.0, "command_name": "pose_command"},
    )
    # tracking_heading = RewTerm(
    #     func=mdp.tracking_heading2,
    #     weight=5.0,
    #     params={"duration": 3.0, "command_name": "pose_command", "max_pos_distance": 0.5},
    # )
    # # -- penalties
    # lin_vel_z_l2 = RewTerm(func=mdp.lin_vel_z_l2, weight=-2.0)
    # ang_vel_xy_l2 = RewTerm(func=mdp.ang_vel_xy_l2, weight=-0.05)
    dof_torques_l2 = RewTerm(func=mdp.joint_torques_l2, weight=-1.0e-5)
    # dof_acc_l2 = RewTerm(func=mdp.joint_acc_l2, weight=-2.5e-7)
    action_rate_l2 = RewTerm(func=mdp.action_rate_l2, weight=-0.01)
    # undesired_contacts = RewTerm(
    #     func=mdp.undesired_contacts,
    #     weight=-1.0,
    #     params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*THIGH"), "threshold": 1.0},
    # )
    # # -- optional penalties
    flat_orientation_l2 = RewTerm(func=mdp.flat_orientation_l2, weight=-1.0)
    # dof_pos_limits = RewTerm(func=mdp.joint_pos_limits, weight=0.0)

    termination_penalty = RewTerm(func=mdp.is_terminated, weight=-200.0)
    
    dont_wait = RewTerm(func=mdp.dont_wait, weight=-1.0, params={"min_vel": 0.2, "command_name": "pose_command"})
    move_in_direction = RewTerm(func=mdp.move_in_direction, weight=1.0, params={"command_name": "pose_command"}) 
    
    # -- reward for time efficiency
    # time_efficiency_reward = RewTerm(func=mdp.time_efficiency_reward, weight=2.0, params={"command_name": "pose_command"})
    
    # parkour tuning rewards
    dof_vel_l2 = RewTerm(func=mdp.joint_vel_l2, weight=-0.001)
    base_acc = RewTerm(
        func=mdp.base_acc, 
        weight=-0.001,
        params={"asset_cfg": SceneEntityCfg("robot", body_names=["base"])}
    )
    collision = RewTerm(func=mdp.collision, weight=-0.5, params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*(THIGH|SHANK)")})
    applied_torque_limits = RewTerm(func=mdp.applied_torque_limits, weight=-0.2)
    dof_vel_limits = RewTerm(func=mdp.joint_vel_limits, weight=-1.0, params={"soft_ratio": 0.9})
    feet_acc = RewTerm(
        func=mdp.feet_acc,
        weight=-0.002,
        params={"asset_cfg": SceneEntityCfg("robot", body_names=[".*_FOOT"])}
    )
    contact_forces = RewTerm(
        func=mdp.contact_forces,
        weight=-0.00001,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*"), "threshold": 700.0},
    )
    # stand_still = RewTerm(func=mdp.stand_still_pose, weight=-1.0, params={"duration": 1.0, "command_name": "pose_command"})
    feet_balance = RewTerm(
        func=mdp.feet_balance,
        weight=-1000,
        params={"duration": 1.0, "command_name": "pose_command", "asset_cfg": SceneEntityCfg("robot", body_names=[".*_FOOT"])}
    )


@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    base_contact = DoneTerm(
        func=mdp.illegal_contact,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names="base"), "threshold": 1.0},
    )
    illegal_force_feet = DoneTerm(
        func=mdp.illegal_force,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*FOOT"), "max_force": 1500},
    )
    illegal_force = DoneTerm(
        func=mdp.illegal_force,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*"), "max_force": 5000},
    )
    # command_resample = DoneTerm(func=mdp.command_resample, params={"command_name": "pose_command", "num_resamples": 1})


@configclass
class CurriculumCfg:
    """Curriculum terms for the MDP."""

    pass


##
# Environment configuration
##


@configclass
class PosTrackingEnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for the locomotion velocity-tracking environment."""

    # Scene settings
    scene: SceneEntityCfg = LOW_LEVEL_ENV_CFG.scene
    # Basic settings
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    # MDP settings
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()
    curriculum: CurriculumCfg = CurriculumCfg()

    def __post_init__(self):
        """Post initialization."""
        # general settings
        self.decimation = LOW_LEVEL_ENV_CFG.decimation
        self.episode_length_s = self.commands.pose_command.resampling_time_range[1]
        # simulation settings
        self.sim.dt = LOW_LEVEL_ENV_CFG.sim.dt
        self.sim.render_interval = LOW_LEVEL_ENV_CFG.decimation * 10
        self.sim.disable_contact_processing = True
        self.sim.physics_material = self.scene.terrain.physics_material
        # update sensor update periods
        # we tick all the sensors based on the smallest update period (physics update period)
        if self.scene.height_scanner is not None:
            self.scene.height_scanner.update_period = self.decimation * self.sim.dt
        else:
            self.observations.policy.height_scan = None
            self.observations.critic.height_scan = None
            
        if self.scene.contact_forces is not None:
            self.scene.contact_forces.update_period = self.sim.dt

        # check if terrain levels curriculum is enabled - if so, enable curriculum for terrain generator
        # this generates terrains with increasing difficulty and is useful for training
        if getattr(self.curriculum, "terrain_levels", None) is not None:
            if self.scene.terrain.terrain_generator is not None:
                self.scene.terrain.terrain_generator.curriculum = True
        else:
            if self.scene.terrain.terrain_generator is not None:
                self.scene.terrain.terrain_generator.curriculum = False
        
        if getattr(self.events, "block_joint", None) is not None:
            self.scene.robot.debug_vis = True

class PosTrackingEnvCfg_PLAY(PosTrackingEnvCfg):
    def __post_init__(self) -> None:
        # post init of parent
        super().__post_init__()

        # make a smaller scene for play
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        # disable randomization for play
        self.observations.policy.enable_corruption = False
        # generate symmetric commands
        # self.commands.pose_command.generate_symmetric_commands = True
