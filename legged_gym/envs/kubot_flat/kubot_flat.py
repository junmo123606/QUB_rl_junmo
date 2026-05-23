import torch
import numpy as np
import os
import math

from isaacgym.torch_utils import *
from isaacgym import gymtorch, gymapi, gymutil

from legged_gym import LEGGED_GYM_ROOT_DIR, envs
from legged_gym.envs.base.base_task import BaseTask
from legged_gym.utils.terrain import Terrain
from legged_gym.utils.helpers import class_to_dict
from legged_gym.utils.math import (
    quat_apply_yaw,
    wrap_to_pi,
    torch_rand_sqrt_float,
)
from .kubot_flat_config import KubotCfg

class KubotEnv(BaseTask):

    def __init__(
        self, cfg: KubotCfg, sim_params, physics_engine, sim_device, headless
    ):
        self.cfg = cfg
        self.sim_params = sim_params
        self.height_samples = None

        self.init_done = False
        self._parse_cfg(self.cfg)
        super().__init__(self.cfg, sim_params, physics_engine, sim_device, headless)
        self.pi = torch.acos(torch.zeros(1, device=self.device)) * 2

        self.group_idx = torch.arange(0, self.cfg.env.num_envs)

        if not self.headless:
            self.set_camera(self.cfg.viewer.pos, self.cfg.viewer.lookat)
        self._init_buffers()
        self._prepare_reward_function()
        self.init_done = True

        self._print_robot_indices()

    def _print_robot_indices(self):
        print("=" * 70)
        print("[KubotEnv] Robot index verification")
        print("-" * 70)
        print("DOF order:")
        for i, name in enumerate(self.dof_names):
            print(f"  [{i:2d}] {name}")
        print("-" * 70)
        print(f"feet_indices              : {self.feet_indices.cpu().tolist()}")
        if hasattr(self, "penalised_contact_indices"):
            print(f"penalised_contact_indices : {self.penalised_contact_indices.cpu().tolist()}")
        if hasattr(self, "termination_contact_indices"):
            print(f"termination_contact_indices: {self.termination_contact_indices.cpu().tolist()}")
        ankle_idx = [i for i, n in enumerate(self.dof_names) if "ankle" in n]
        print(f"ankle DOF indices (auto)  : {ankle_idx}")
        print("=" * 70)

    def step(self, actions):
        if hasattr(self, '_action_clip'):
            self._action_clip(actions)
        else:
            self.actions = torch.clip(actions, -self.cfg.normalization.clip_actions, self.cfg.normalization.clip_actions)
            
        self.render()
        self.pre_physics_step()
        for _ in range(self.cfg.control.decimation):
            self.action_fifo = torch.cat(
                (self.actions.unsqueeze(1), self.action_fifo[:, :-1, :]), dim=1
            )
            self.envs_steps_buf += 1
            self.torques = self._compute_torques(
                self.action_fifo[torch.arange(self.num_envs), self.action_delay_idx, :]
            ).view(self.torques.shape)
            self.gym.set_dof_actuation_force_tensor(
                self.sim, gymtorch.unwrap_tensor(self.torques)
            )
            if self.cfg.domain_rand.push_robots:
                self._push_robots()
            self.gym.simulate(self.sim)
            if self.device == "cpu":
                self.gym.fetch_results(self.sim, True)
            self.gym.refresh_dof_state_tensor(self.sim)
            self.compute_dof_vel()
        self.post_physics_step()

        clip_obs = self.cfg.normalization.clip_observations
        self.obs_buf = torch.clip(self.obs_buf, -clip_obs, clip_obs)
        return (
            self.obs_buf,
            self.rew_buf,
            self.reset_buf,
            self.extras,
            self.obs_history,
            self.commands[:, :3] * self.commands_scale,
            self.critic_obs_buf
        )

    def _resample_commands(self, env_ids):
        self.commands[env_ids, 0] = (
            self.command_ranges["lin_vel_x"][env_ids, 1]
            - self.command_ranges["lin_vel_x"][env_ids, 0]
        ) * torch.rand(len(env_ids), device=self.device) + self.command_ranges["lin_vel_x"][env_ids, 0]
        self.commands[env_ids, 1] = (
            self.command_ranges["lin_vel_y"][env_ids, 1]
            - self.command_ranges["lin_vel_y"][env_ids, 0]
        ) * torch.rand(len(env_ids), device=self.device) + self.command_ranges["lin_vel_y"][env_ids, 0]
        self.commands[env_ids, 2] = (
            self.command_ranges["ang_vel_yaw"][env_ids, 1]
            - self.command_ranges["ang_vel_yaw"][env_ids, 0]
        ) * torch.rand(len(env_ids), device=self.device) + self.command_ranges["ang_vel_yaw"][env_ids, 0]
        if self.cfg.commands.heading_command:
            self.commands[env_ids, 3] = torch_rand_float(
                self.command_ranges["heading"][0],
                self.command_ranges["heading"][1],
                (len(env_ids), 1),
                device=self.device,
            ).squeeze(1)

        zero_command_idx = (
            (
                torch_rand_float(0, 1, (len(env_ids), 1), device=self.device)
                > self.cfg.commands.zero_command_prob
            )
            .squeeze(1)
            .nonzero(as_tuple=False)
            .flatten()
        )
        self.commands[zero_command_idx, :3] = 0
        if self.cfg.commands.heading_command:
            forward = quat_apply(
                self.base_quat[zero_command_idx], self.forward_vec[zero_command_idx]
            )
            heading = torch.atan2(forward[:, 1], forward[:, 0])
            self.commands[zero_command_idx, 3] = heading

    def _compute_torques(self, actions):
        actions_scaled = actions * self.cfg.control.action_scale

        control_type = self.cfg.control.control_type
        if control_type == "P":
            torques = (
                self.p_gains * (actions_scaled + self.default_dof_pos - self.dof_pos)
                - self.d_gains * self.dof_vel
            )
        elif control_type == "V":
            torques = (
                self.p_gains * (actions_scaled - self.dof_vel)
                - self.d_gains * (self.dof_vel - self.last_dof_vel) / self.sim_params.dt
            )
        elif control_type == "T":
            torques = actions_scaled
        else:
            raise NameError(f"Unknown controller type: {control_type}")
        return torch.clip(
            torques * self.torques_scale, -self.torque_limits, self.torque_limits
        )

    def _get_noise_scale_vec(self, cfg):
        noise_vec = torch.zeros_like(self.obs_buf[0])
        self.add_noise = self.cfg.noise.add_noise
        noise_scales = self.cfg.noise.noise_scales
        noise_level = self.cfg.noise.noise_level

        n = self.cfg.env.num_actions

        idx = 0
        noise_vec[idx:idx + 3] = noise_scales.ang_vel * noise_level * self.obs_scales.ang_vel
        idx += 3
        noise_vec[idx:idx + 3] = noise_scales.gravity * noise_level
        idx += 3
        noise_vec[idx:idx + n] = noise_scales.dof_pos * noise_level * self.obs_scales.dof_pos
        idx += n
        noise_vec[idx:idx + n] = noise_scales.dof_vel * noise_level * self.obs_scales.dof_vel

        return noise_vec

    def reset_idx(self, env_ids):
        if len(env_ids) == 0:
            return
        if self.cfg.terrain.curriculum:
            self._update_terrain_curriculum(env_ids)
        if self.cfg.commands.curriculum:
            time_out_env_ids = self.time_out_buf.nonzero(as_tuple=False).flatten()
            self.update_command_curriculum(time_out_env_ids)

        self._reset_dofs(env_ids)
        self._reset_root_states(env_ids)
        self._resample_commands(env_ids)
        self._resample_gaits(env_ids)

        self.last_actions[env_ids] = 0.0
        self.last_dof_pos[env_ids] = self.dof_pos[env_ids]
        self.last_base_position[env_ids] = self.base_position[env_ids]
        self.last_foot_positions[env_ids] = self.foot_positions[env_ids]
        self.last_dof_vel[env_ids] = 0.0
        self.feet_air_time[env_ids] = 0.0
        self.episode_length_buf[env_ids] = 0
        self.envs_steps_buf[env_ids] = 0
        self.reset_buf[env_ids] = 1
        self.obs_history[env_ids] = 0
        obs_buf, _ = self.compute_group_observations()
        self.obs_history[env_ids] = obs_buf[env_ids].repeat(1, self.obs_history_length)
        self.gait_indices[env_ids] = 0
        self.fail_buf[env_ids] = 0
        self.action_fifo[env_ids] = 0
        self.dof_pos_int[env_ids] = 0

        self.extras["episode"] = {}
        for key in self.episode_sums.keys():
            self.extras["episode"]["rew_" + key] = (
                torch.mean(self.episode_sums[key][env_ids]) / self.max_episode_length_s
            )
            self.episode_sums[key][env_ids] = 0.0
        if self.cfg.terrain.curriculum:
            self.extras["episode"]["group_terrain_level"] = torch.mean(
                self.terrain_levels[self.group_idx].float()
            )
            self.extras["episode"]["group_terrain_level_stair_up"] = torch.mean(
                self.terrain_levels[self.stair_up_idx].float()
            )
        if self.cfg.terrain.curriculum and self.cfg.commands.curriculum:
            self.extras["episode"]["max_command_x"] = torch.mean(
                self.command_ranges["lin_vel_x"][self.smooth_slope_idx, 1].float()
            )
        if self.cfg.env.send_timeouts:
            self.extras["time_outs"] = self.time_out_buf | self.edge_reset_buf

    def compute_observations(self):
        self.obs_buf, self.critic_obs_buf = self.compute_group_observations()
        if self.cfg.env.obs_history_length > 1:
            self.obs_history = torch.cat((self.obs_buf, self.obs_history[:, :-self.cfg.env.num_observations]), dim=-1)

    def compute_group_observations(self):
        obs_buf = torch.cat(
            (
                self.base_ang_vel * self.obs_scales.ang_vel,
                self.projected_gravity,
                (self.dof_pos - self.default_dof_pos) * self.obs_scales.dof_pos,
                self.dof_vel * self.obs_scales.dof_vel,
                self.actions,
                self.clock_inputs_sin.view(self.num_envs, 1),
                self.clock_inputs_cos.view(self.num_envs, 1),
                self.gaits,
            ),
            dim=-1,
        )
        critic_obs_buf = torch.cat((
            self.base_lin_vel * self.obs_scales.lin_vel, obs_buf), dim=-1)
        return obs_buf, critic_obs_buf

    def _get_ankle_dof_indices(self):
        if not hasattr(self, "_ankle_dof_indices_cached"):
            ankle_idx = [i for i, name in enumerate(self.dof_names) if "ankle" in name]
            self._ankle_dof_indices_cached = torch.tensor(
                ankle_idx, dtype=torch.long, device=self.device
            )
        return self._ankle_dof_indices_cached

    # ===================== reward functions =====================

    def _reward_lin_vel_z(self):
        return torch.square(self.base_lin_vel[:, 2])

    def _reward_ang_vel_xy(self):
        return torch.sum(torch.square(self.base_ang_vel[:, :2]), dim=1)

    def _reward_orientation(self):
        return torch.sum(torch.square(self.projected_gravity[:, :2]), dim=1)

    def _reward_base_height(self):
        base_height = torch.mean(self.root_states[:, 2].unsqueeze(1) - self.measured_heights, dim=1)
        return torch.square(base_height - self.cfg.rewards.base_height_target)

    def _reward_torques(self):
        return torch.sum(torch.square(self.torques), dim=1)

    def _reward_dof_acc(self):
        return torch.sum(torch.square(self.dof_acc), dim=1)

    def _reward_action_rate(self):
        return torch.sum(torch.square(self.actions - self.last_actions[:, :, 0]), dim=1)

    def _reward_action_smooth(self):
        return torch.sum(
            torch.square(
                self.actions - 2 * self.last_actions[:, :, 0] + self.last_actions[:, :, 1]), dim=1)

    def _reward_keep_balance(self):
        base_height = self.root_states[:, 2]
        height_ok = (base_height > self.cfg.rewards.alive_min_height).float()
        upright_ok = (self.projected_gravity[:, 2] < -self.cfg.rewards.alive_max_tilt).float()
        return height_ok * upright_ok

    def _reward_dof_pos_limits(self):
        out_of_limits = -(self.dof_pos - self.dof_pos_limits[:, 0]).clip(max=0.0)
        out_of_limits += (self.dof_pos - self.dof_pos_limits[:, 1]).clip(min=0.0)
        return torch.sum(out_of_limits, dim=1)

    def _reward_tracking_lin_vel(self):
        lin_vel_error = torch.sum(
            torch.square(self.commands[:, :2] - self.base_lin_vel[:, :2]), dim=1
        )
        return torch.exp(-lin_vel_error / self.cfg.rewards.tracking_sigma)

    def _reward_tracking_ang_vel(self):
        ang_vel_error = torch.square(self.commands[:, 2] - self.base_ang_vel[:, 2])
        return torch.exp(-ang_vel_error / self.cfg.rewards.ang_tracking_sigma)

    def _reward_tracking_contacts_shaped_force(self):
        foot_forces = torch.norm(self.contact_forces[:, self.feet_indices, :], dim=-1)
        desired_contact = self.desired_contact_states

        reward = 0
        if self.reward_scales["tracking_contacts_shaped_force"] > 0:
            for i in range(len(self.feet_indices)):
                reward += (1 - desired_contact[:, i]) * torch.exp(
                    -foot_forces[:, i] ** 2 / self.cfg.rewards.gait_force_sigma)
        else:
            for i in range(len(self.feet_indices)):
                reward += (1 - desired_contact[:, i]) * (
                    1 - torch.exp(-foot_forces[:, i] ** 2 / self.cfg.rewards.gait_force_sigma))

        return reward / len(self.feet_indices)

    def _reward_tracking_contacts_shaped_vel(self):
        foot_velocities = torch.norm(self.foot_velocities, dim=-1)
        desired_contact = self.desired_contact_states
        reward = 0
        if self.reward_scales["tracking_contacts_shaped_vel"] > 0:
            for i in range(len(self.feet_indices)):
                reward += desired_contact[:, i] * torch.exp(
                    -foot_velocities[:, i] ** 2 / self.cfg.rewards.gait_vel_sigma
                )
        else:
            for i in range(len(self.feet_indices)):
                reward += desired_contact[:, i] * (
                    1 - torch.exp(-foot_velocities[:, i] ** 2 / self.cfg.rewards.gait_vel_sigma))
        return reward / len(self.feet_indices)

    def _reward_feet_distance(self):
        feet_distance = torch.norm(self.foot_positions[:, 0, :2] - self.foot_positions[:, 1, :2], dim=-1)
        reward = torch.clip(self.cfg.rewards.min_feet_distance - feet_distance, 0, 1)
        return reward

    def _reward_feet_distance_max(self):
        feet_distance = torch.norm(
            self.foot_positions[:, 0, :2] - self.foot_positions[:, 1, :2], dim=-1
        )
        return torch.clip(feet_distance - self.cfg.rewards.max_feet_distance, 0, 2)

    def _reward_feet_regulation(self):
        feet_height = self.cfg.rewards.base_height_target * 0.001
        reward = torch.sum(
            torch.exp(-self.foot_heights / feet_height)
            * torch.square(torch.norm(self.foot_velocities[:, :, :2], dim=-1)), dim=1)
        return reward

    def _reward_collision(self):
        return torch.sum(
            torch.norm(self.contact_forces[:, self.penalised_contact_indices, :], dim=-1) > 1.0, dim=1)

    def _reward_foot_landing_vel(self):
        z_vels = self.foot_velocities[:, :, 2]
        contacts = self.contact_forces[:, self.feet_indices, 2] > 0.1
        about_to_land = (self.foot_heights < self.cfg.rewards.about_landing_threshold) & (~contacts) & (z_vels < 0.0)
        landing_z_vels = torch.where(about_to_land, z_vels, torch.zeros_like(z_vels))
        reward = torch.sum(torch.square(landing_z_vels), dim=1)
        return reward

    def _reward_posture(self):
        return torch.sum(torch.square(self.dof_pos - self.default_dof_pos), dim=1)

    def _reward_ankle_regularization(self):
        ankle_idx = self._get_ankle_dof_indices()
        return torch.sum(
            torch.square(self.dof_pos[:, ankle_idx] - self.default_dof_pos[:, ankle_idx]),
            dim=1,
        )

    def _reward_ang_vel_z(self):
        return torch.square(self.base_ang_vel[:, 2] - self.commands[:, 2])

    def _reward_yaw_drift(self):
        roll, pitch, yaw = get_euler_xyz(self.base_quat)
        yaw_wrapped = torch.atan2(torch.sin(yaw), torch.cos(yaw))
        return torch.square(yaw_wrapped)

    def _reward_symmetry(self):
        l_actions = self.actions[:, 1:7]
        r_actions = self.actions[:, 7:13]
        if not hasattr(self, "_sym_mirror"):
            self._sym_mirror = torch.tensor(
                [1.0, -1.0, -1.0, 1.0, 1.0, -1.0],
                device=self.device, dtype=torch.float,
            )
        diff = l_actions - r_actions * self._sym_mirror
        return torch.sum(torch.square(diff), dim=1)

    def _reward_feet_air_time(self):
        contact = self.contact_forces[:, self.feet_indices, 2] > 1.0
        if not hasattr(self, "_last_contacts_air"):
            self._last_contacts_air = torch.zeros_like(contact)
        contact_filt = contact | self._last_contacts_air
        self._last_contacts_air = contact
        first_contact = (self.feet_air_time > 0.0) * contact_filt
        rew_air = torch.sum((self.feet_air_time - 0.4) * first_contact.float(), dim=1)
        rew_air *= (torch.norm(self.commands[:, :2], dim=1) > 0.1).float()
        return rew_air
