# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

from legged_gym.envs.base.base_config import BaseConfig
import os

class KubotCfg(BaseConfig):
    class env:
        num_envs = 4096
        num_observations = 72
        num_critic_observations = 3 + num_observations
        num_height_samples = 117
        num_actions = 20
        env_spacing = 3.0
        send_timeouts = True
        episode_length_s = 20
        obs_history_length = 10
        dof_vel_use_pos_diff = False
        fail_to_terminal_time_s = 2.0

    class terrain:
        mesh_type = "plane"
        horizontal_scale = 0.1
        vertical_scale = 0.005
        border_size = 25
        curriculum = False
        static_friction = 1.0
        dynamic_friction = 1.0
        restitution = 0.0
        measure_heights = False
        critic_measure_heights = False
        terrain_length = 8.0
        terrain_width = 8.0
        measured_points_x = [-0.6, -0.5, -0.4, -0.3, -0.2, -0.1, 0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
        measured_points_y = [-0.4, -0.3, -0.2, -0.1, 0.0, 0.1, 0.2, 0.3, 0.4]
        selected = False
        terrain_kwargs = None
        max_init_terrain_level = 9
        num_rows = 10
        num_cols = 20
        terrain_proportions = [0.1, 0.1, 0.35, 0.25, 0.2]
        slope_treshold = 0.75

    class commands:
        curriculum = False
        num_commands = 3
        resampling_time = 5.0
        heading_command = False
        zero_command_prob = 0.5
        smooth_max_lin_vel_x = 2.0
        smooth_max_lin_vel_y = 1.0
        non_smooth_max_lin_vel_x = 1.0
        non_smooth_max_lin_vel_y = 1.0
        max_ang_vel_yaw = 3.0
        curriculum_threshold = 0.75
        min_norm = 0.1
        class ranges:
            lin_vel_x = [0.0, 0.3] 
            lin_vel_y = [-0.1, 0.1]
            ang_vel_yaw = [-0.3, 0.3]
            heading = [-3.14, 3.14]

    class gait:
        num_gait_params = 4
        resampling_time = 5
        class ranges:
            frequencies = [1.4, 1.7]
            offsets = [0.5, 0.5]
            durations = [0.55, 0.65]
            swing_height = [0.06, 0.10]
            p = [0.5, 0.5]
            d = [0.5, 0.5]
            f = [0.5, 0.5]
            z = [0.5, 0.5]
            h = [0.5, 0.5]

    class init_state:
        pos = [0.0, 0.0, 0.55]
        rot = [0.0, 0.0, 0.0, 1.0]
        lin_vel = [0.0, 0.0, 0.0]
        ang_vel = [0.0, 0.0, 0.0]

        default_joint_angles = {
            "Neck_yaw_joint": 0.0, "Head_pitch_joint": 0.0,
            "L_Shoulder_pitch_joint": 0.15, "L_Arm_roll_joint": 0.0, "L_Hand_pitch_joint": 0.0,
            "R_Shoulder_pitch_joint": 0.15, "R_Arm_roll_joint": 0.0, "R_Hand_pitch_joint": 0.0,
            "L_Hip_yaw_joint": 0.0, "L_Hip_roll_joint": 0.0, "L_Hip_pitch_joint": -0.10,
            "L_Knee_pitch_joint": 0.30, "L_Ankle_pitch_joint": -0.20, "L_Ankle_roll_joint": 0.0,
            "R_Hip_yaw_joint": 0.0, "R_Hip_roll_joint": 0.0, "R_Hip_pitch_joint": -0.10,
            "R_Knee_pitch_joint": 0.30, "R_Ankle_pitch_joint": -0.20, "R_Ankle_roll_joint": 0.0,
        }

    class control:
        action_scale = 0.25
        control_type = "P"

        stiffness = {
            "Neck": 20.0, "Head": 20.0, "Shoulder": 40.0, "Arm": 40.0, "Hand": 20.0,
            "Hip_yaw": 100.0, "Hip_roll": 150.0, "Hip_pitch": 150.0,
            "Knee_pitch": 200.0, "Ankle_pitch": 80.0, "Ankle_roll": 80.0,
        }
        damping = {
            "Neck": 1.0, "Head": 1.0, "Shoulder": 2.0, "Arm": 2.0, "Hand": 1.0,
            "Hip_yaw": 3.0, "Hip_roll": 4.0, "Hip_pitch": 4.0,
            "Knee_pitch": 6.0, "Ankle_pitch": 3.0, "Ankle_roll": 3.0,
        }
        decimation = 4
        user_torque_limit = 120.0

        torque_limits = {
            "Neck": 100.0, "Head": 100.0, "Shoulder": 100.0, "Arm": 100.0, "Hand": 100.0,
            "Hip_yaw": 100.0, "Hip_roll": 150.0, "Hip_pitch": 200.0,
            "Knee_pitch": 200.0, "Ankle_pitch": 120.0, "Ankle_roll": 120.0,
        }
        max_power = 1000.0

    class asset:
        file = '{LEGGED_GYM_ROOT_DIR}/resources/robots/kubot/urdf/kubot25_pkgs.urdf'
        name = "kubot"
        foot_name = "Foot"
        foot_radius = 0.03
        penalize_contacts_on = ["Hip", "Knee", "Shoulder", "Arm", "Hand", "Neck", "Head"]
        terminate_after_contacts_on = ["base_link", "Neck", "Head"]
        disable_gravity = False
        collapse_fixed_joints = False
        fix_base_link = False
        default_dof_drive_mode = 3
        self_collisions = 0
        replace_cylinder_with_capsule = True
        flip_visual_attachments = False
        density = 0.001
        angular_damping = 0.0
        linear_damping = 0.0
        max_angular_velocity = 100.0
        max_linear_velocity = 100.0
        armature = 0.05
        thickness = 0.01

    class domain_rand:
        randomize_friction = False
        friction_range = [0.5, 1.25]
        randomize_base_mass = False
        added_mass_range = [-1.0, 1.0]
        randomize_base_com = False
        rand_com_vec = [0.03, 0.02, 0.03]
        push_robots = False
        push_interval_s = 10
        max_push_vel_xy = 1.0
        randomize_action_delay = False
        delay_ms_range = [0, 20]
        randomize_motor_torque = False
        randomize_motor_torque_range = [0.85, 1.15]
        randomize_Kp = False
        randomize_Kp_range = [0.9, 1.1]
        randomize_Kd = False
        randomize_Kd_range = [0.9, 1.1]
        randomize_restitution = False
        restitution_range = [0.0, 0.2]
        randomize_imu_offset = False
        randomize_imu_offset_range = [-0.05, 0.05]
        randomize_inertia = False
        randomize_inertia_range = [0.8, 1.2]
        randomize_default_dof_pos = False
        randomize_default_dof_pos_range = [-0.05, 0.05]

    class rewards:
        class scales:
            keep_balance = 10.0
            orientation = -10.0
            base_height = -5.0
            yaw_drift = -0.5
            tracking_lin_vel = 3.0
            tracking_ang_vel = 1.5
            feet_air_time = 2.0          
            tracking_contacts_shaped_force = 2.0
            tracking_contacts_shaped_vel = 2.0
            posture = -0.3
            ankle_regularization = -1.0
            symmetry = 1.0
            dof_pos_limits = -10.0
            torques = -2.0e-5
            action_rate = -0.02
            action_smooth = -0.02
            dof_acc = -2.5e-7
            lin_vel_z = -2.0
            ang_vel_xy = -0.1
            ang_vel_z = 0.0
            foot_landing_vel = -0.5
            feet_regulation = -0.05
            feet_distance = -1.0
            feet_distance_max = -1.0
            collision = -2.0

        only_positive_rewards = False
        clip_reward = 100
        clip_single_reward = 5
        tracking_sigma = 0.25
        ang_tracking_sigma = 0.25
        height_tracking_sigma = 0.1
        soft_dof_pos_limit = 0.9
        soft_dof_vel_limit = 1.0
        soft_torque_limit = 0.8
        base_height_target = 0.55
        feet_height_target = 0.08
        min_feet_distance = 0.18
        max_feet_distance = 0.40
        alive_min_height = 0.40
        alive_max_tilt = 0.70
        max_contact_force = 300.0
        kappa_gait_probs = 0.07
        gait_force_sigma = 50.0
        gait_vel_sigma = 0.5
        gait_height_sigma = 0.02
        about_landing_threshold = 0.05

    class normalization:
        class obs_scales:
            lin_vel = 2.0
            ang_vel = 0.25
            dof_pos = 1.0
            dof_vel = 0.05
            torque = 0.05
        clip_observations = 100.0
        clip_actions = 1.0

    class noise:
        add_noise = True
        noise_level = 0.1
        class noise_scales:
            dof_pos = 0.01
            dof_vel = 1.5
            lin_vel = 0.1
            ang_vel = 0.2
            gravity = 0.05

    class viewer:
        ref_env = 0
        pos = [3, -3, 2]
        lookat = [0, 0, 0]
        realtime_plot = True

    class sim:
        dt = 0.005
        substeps = 2
        gravity = [0.0, 0.0, -9.81]
        up_axis = 1

        class physx:
            num_threads = 10
            solver_type = 1
            num_position_iterations = 8
            num_velocity_iterations = 2
            contact_offset = 0.01
            rest_offset = 0.0
            bounce_threshold_velocity = 0.5
            max_depenetration_velocity = 1.0
            max_gpu_contact_pairs = 2**23
            default_buffer_size_multiplier = 5
            contact_collection = 2

class KubotCfgPPO(BaseConfig):
    seed = 1
    runner_class_name = "OnPolicyRunner"

    class MLP_Encoder:
        output_detach = True
        num_input_dim = KubotCfg.env.num_observations * KubotCfg.env.obs_history_length
        num_output_dim = 3
        hidden_dims = [256, 128]
        activation = "elu"
        orthogonal_init = False

    class policy:
        init_noise_std = 0.6
        actor_hidden_dims = [512, 256, 128]
        critic_hidden_dims = [512, 256, 128]
        activation = "elu"
        orthogonal_init = False

    class algorithm:
        value_loss_coef = 1.0
        use_clipped_value_loss = True
        clip_param = 0.2
        entropy_coef = 0.01
        num_learning_epochs = 5
        num_mini_batches = 4
        learning_rate = 1.0e-3
        schedule = "adaptive"
        gamma = 0.99
        lam = 0.95
        desired_kl = 0.01
        max_grad_norm = 1.0
        critic_take_latent = True
        est_learning_rate = 1.0e-3
        ts_learning_rate = 1.0e-4

    class runner:
        encoder_class_name = "MLP_Encoder"
        policy_class_name = "ActorCritic"
        algorithm_class_name = "PPO"
        num_steps_per_env = 24
        max_iterations = 15000
        logger = "tensorboard"
        exptid = ""
        wandb_project = "legged_gym_KUBOT"
        save_interval = 100
        experiment_name = "kubot_flat"
        run_name = ""
        resume = False
        load_run = "None"
        checkpoint = -1
        resume_path = "None"
