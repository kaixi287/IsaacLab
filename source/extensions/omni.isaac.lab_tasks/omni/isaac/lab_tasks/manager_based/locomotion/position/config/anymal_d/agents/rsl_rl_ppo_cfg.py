# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from omni.isaac.lab.utils import configclass

from omni.isaac.lab_tasks.utils.wrappers.rsl_rl import (
    RslRlOnPolicyRunnerCfg,
    RslRlPpoActorCriticCfg,
    RslRlPpoAlgorithmCfg,
)


@configclass
class AnymalDPosTrackingEnvPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 48
    max_iterations = 2000
    save_interval = 100
    experiment_name = "anymal_d_pos_tracking"
    empirical_normalization = False
    policy = RslRlPpoActorCriticCfg(
        # class_name = "ActorCriticRecurrent",
        init_noise_std=0.5,
        actor_hidden_dims=[128, 128, 128],
        critic_hidden_dims=[128, 128, 128],
        activation="elu",
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.005,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
        # -- Symmetry Augmentation
        symmetry_cfg=dict(
            use_data_augmentation=False,  # this adds symmetric trajectories to the batch
            use_mirror_loss=False,  # this adds symmetry loss term to the loss function
            data_augmentation_func=(  # specify the data augmentation function if any
                "omni.isaac.lab_tasks.manager_based.locomotion.position.config.anymal_d.symmetry:get_symmetric_states"
            ),
            mirror_loss_coeff=0.0,  # coefficient for symmetry loss term
        ),
    )
