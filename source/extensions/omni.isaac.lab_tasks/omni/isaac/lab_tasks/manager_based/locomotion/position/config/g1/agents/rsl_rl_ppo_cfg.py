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
class G1PosTrackingEnvPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 24
    max_iterations = 8000
    save_interval = 500
    experiment_name = "g1_pos_tracking"
    empirical_normalization = False
    policy = RslRlPpoActorCriticCfg(
        init_noise_std=1.0,
        actor_hidden_dims=[256, 128, 128],
        critic_hidden_dims=[256, 128, 128],
        activation="elu",
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.008,
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
            # data_augmentation_func="omni.isaac.lab_tasks.manager_based.locomotion.position.config.g1.symmetry:get_symmetric_states",  # specify the data augmentation function if any
            data_augmentation_func=None,  # specify the data augmentation function if any
            mirror_loss_coeff=0.0  # coefficient for symmetry loss term
        )
    )
