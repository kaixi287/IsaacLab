import torch
from typing import Optional, Tuple

from rsl_rl.env import VecEnv


# Anymal_D joints: ['LF_HAA', 'LH_HAA', 'RF_HAA', 'RH_HAA', 'LF_HFE', 'LH_HFE', 'RF_HFE', 'RH_HFE', 'LF_KFE', 'LH_KFE', 'RF_KFE', 'RH_KFE']

def _switch_legs_lr(dof):
    dof_switched = torch.zeros_like(dof, device=dof.device)
    
    # Left Front <-> Right Front (HAA, HFE, KFE)
    dof_switched[..., 0] = dof[..., 2] * -1  # LF_HAA <-> RF_HAA
    dof_switched[..., 2] = dof[..., 0] * -1
    dof_switched[..., 4] = dof[..., 6]  # LF_HFE <-> RF_HFE
    dof_switched[..., 6] = dof[..., 4]
    dof_switched[..., 8] = dof[..., 10] # LF_KFE <-> RF_KFE
    dof_switched[..., 10] = dof[..., 8]

    # Left Hind <-> Right Hind (HAA, HFE, KFE)
    dof_switched[..., 1] = dof[..., 3] * -1  # LH_HAA <-> RH_HAA
    dof_switched[..., 3] = dof[..., 1] * -1
    dof_switched[..., 5] = dof[..., 7]  # LH_HFE <-> RH_HFE
    dof_switched[..., 7] = dof[..., 5]
    dof_switched[..., 9] = dof[..., 11] # LH_KFE <-> RH_KFE
    dof_switched[..., 11] = dof[..., 9]
    return dof_switched

def _switch_legs_fb(dof):
    dof_switched = torch.zeros_like(dof, device=dof.device)
    # Front left <-> Hind left (HAA, HFE, KFE)
    dof_switched[..., 0] = dof[..., 1]  # LF_HAA <-> LH_HAA
    dof_switched[..., 1] = dof[..., 0]
    dof_switched[..., 4] = dof[..., 5] * -1  # LF_HFE <-> LH_HFE
    dof_switched[..., 5] = dof[..., 4] * -1
    dof_switched[..., 8] = dof[..., 9] * -1  # LF_KFE <-> LH_KFE
    dof_switched[..., 9] = dof[..., 8] * -1

    # Front right <-> Hind right (HAA, HFE, KFE)
    dof_switched[..., 2] = dof[..., 3]  # RF_HAA <-> RH_HAA
    dof_switched[..., 3] = dof[..., 2]
    dof_switched[..., 6] = dof[..., 7] * -1  # RF_HFE <-> RH_HFE
    dof_switched[..., 7] = dof[..., 6] * -1
    dof_switched[..., 10] = dof[..., 11] * -1  # RF_KFE <-> RH_KFE
    dof_switched[..., 11] = dof[..., 10] * -1
    return dof_switched

def _transform_obs_left_right(obs, has_height_scan=False):
    obs = obs.clone()
    # Flip lin vel y [1], ang vel x,z [3, 5], gravity y [7]
    obs[..., [1, 3, 5, 7]] *= -1
    # Flip commands pos y [10], commands heading sin [11]
    obs[..., [10, 11]] *= -1
    # dof pos
    obs[..., 14:26] = _switch_legs_lr(obs[..., 14:26])
    # dof vel
    obs[..., 26:38] = _switch_legs_lr(obs[..., 26:38])
    # last actions
    obs[..., 38:50] = _switch_legs_lr(obs[..., 38:50])
    # height_scan
    if has_height_scan:
        obs[..., 50:]  = obs[..., 50:].view(*obs.shape[:-1], 21, 11).flip(dims=[-1]).view(*obs.shape[:-1], 21*11)
    return obs

def _transform_obs_front_back(obs, has_height_scan=False):
    obs = obs.clone()
    # Flip lin vel x [0], ang vel y,z [4, 5], gravity x [6]
    obs[..., [0, 4, 5, 6]] *= -1
    # Flip commands pos x [9], commands heading sin [11]
    obs[..., [9, 11]] *= -1
    # dof pos
    obs[..., 14:26] = _switch_legs_fb(obs[..., 14:26])
    # dof vel
    obs[..., 26:38] = _switch_legs_fb(obs[..., 26:38])
    # last actions
    obs[..., 38:50] = _switch_legs_fb(obs[..., 38:50])
    # height_scan
    if has_height_scan:
        obs[..., 50:]  = obs[..., 50:].view(*obs.shape[:-1], 21, 11).flip(dims=[-2]).view(*obs.shape[:-1], 21*11)
    return obs

def _transform_actions_left_right(actions):
    actions = actions.clone()
    actions[:] = _switch_legs_lr(actions[:])
    return actions

def _transform_actions_front_back(actions):
    actions = actions.clone()
    actions[:] = _switch_legs_fb(actions[:])
    return actions

@torch.no_grad()
def get_symmetric_states(
    obs: Optional[torch.Tensor] = None, 
    actions: Optional[torch.Tensor] = None, 
    env: Optional[VecEnv] = None, 
    is_critic: bool = False
) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
    obs_aug, actions_aug = None, None

    if obs is not None:
        # Check if height_scan is enabled
        has_height_scan = False
        if env is not None:
            if is_critic and hasattr(env.cfg.observations, 'critic'):
                has_height_scan = hasattr(env.cfg.observations.critic, 'height_scan') and env.cfg.observations.critic.height_scan
            else:
                has_height_scan = hasattr(env.cfg.observations.policy, 'height_scan') and env.cfg.observations.policy.height_scan
        
        num_envs = obs.shape[-2]
        obs_aug = torch.zeros(*obs.shape[:-2], num_envs * 4, obs.shape[-1], device=obs.device)
        obs_aug[..., :num_envs, :] = obs
        obs_aug[..., num_envs:2*num_envs, :] = _transform_obs_left_right(obs, has_height_scan)
        obs_aug[..., 2*num_envs:3*num_envs, :] = _transform_obs_front_back(obs, has_height_scan)
        obs_aug[..., 3*num_envs:, :] = _transform_obs_front_back(obs_aug[..., num_envs:2*num_envs, :], has_height_scan)

    if actions is not None:
        num_envs = actions.shape[-2]
        actions_aug = torch.zeros(*actions.shape[:-2], num_envs * 4, actions.shape[-1], device=actions.device)
        actions_aug[..., :num_envs, :] = actions
        actions_aug[..., num_envs:2*num_envs, :] = _transform_actions_left_right(actions)
        actions_aug[..., 2*num_envs:3*num_envs, :] = _transform_actions_front_back(actions)
        actions_aug[..., 3*num_envs:, :] = _transform_actions_front_back(actions_aug[..., num_envs:2*num_envs, :])

    return obs_aug, actions_aug