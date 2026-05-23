# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
#
# QUB humanoid - Phase 2 (제자리 보행)
#
# Phase 1 (v6): 안정적 standing 달성
# Phase 2: 잘 작동하는 v6 셋업 그대로 + gait 활성화 + 보행 reward 추가
#
# 변경 사항만 [P2] 주석. 나머지는 모두 Phase 1 그대로.
# Phase 1 체크포인트(May11_22-58-02_)에서 resume

from legged_gym.envs.base.base_config import BaseConfig
from legged_gym import LEGGED_GYM_ROOT_DIR
import os


class QubCfg(BaseConfig):
    class env:
        num_envs = 4096
        num_observations = 51
        num_critic_observations = 3 + num_observations
        num_height_samples = 117
        num_actions = 13
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
        curriculum = True
        static_friction = 1.5
        dynamic_friction = 1.5
        restitution = 0.0
        measure_heights = False
        critic_measure_heights = True
        measured_points_x = [-0.6, -0.5, -0.4, -0.3, -0.2, -0.1, 0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
        measured_points_y = [-0.4, -0.3, -0.2, -0.1, 0.0, 0.1, 0.2, 0.3, 0.4]
        selected = False
        terrain_kwargs = None
        max_init_terrain_level = 9
        terrain_length = 8.0
        terrain_width = 8.0
        num_rows = 10
        num_cols = 20
        terrain_proportions = [0.1, 0.1, 0.35, 0.25, 0.2]
        slope_treshold = 0.75

    class commands:
        curriculum = True
        smooth_max_lin_vel_x = 2.0
        smooth_max_lin_vel_y = 1.0
        non_smooth_max_lin_vel_x = 1.0
        non_smooth_max_lin_vel_y = 1.0
        max_ang_vel_yaw = 3.0
        curriculum_threshold = 0.75
        num_commands = 3
        resampling_time = 5.0
        heading_command = False
        min_norm = 0.1
        zero_command_prob = 0.0

        class ranges:
            # [P2] Phase 2 초반: command 여전히 0. 제자리에서 박자만 학습.
            # 박자가 잡힌 후 (~2000 iter) 다음 단계에서 lin_vel_x = [-0.2, 0.2] 부터 열기
            lin_vel_x = [-0.3, 0.3]
            lin_vel_y = [-0.2, 0.2]
            ang_vel_yaw = [-1.0, 1.0]
            heading = [-3.14, 3.14]

    class gait:
        num_gait_params = 4
        resampling_time = 5

        class ranges:
            # [P2] Gait 활성화 - 석사님 말씀대로 60% stance + 40% swing
            frequencies = [1.7, 1.7]      # 1.7Hz (사람 평지 걸음과 비슷)
            offsets = [0.5, 0.5]          # 좌우 다리 180도 위상차 (교대로)
            durations = [0.6, 0.6]        # 60% stance
            swing_height = [0.03, 0.03]   # swing 시 발 6cm 들기

    class init_state:
        pos = [0.0, 0.0, 0.95]
        rot = [0.0, 0.0, 0.0, 1.0]
        lin_vel = [0.0, 0.0, 0.0]
        ang_vel = [0.0, 0.0, 0.0]
        default_joint_angles = {
            "torso_yaw_joint": 0.0,
            "R_hip_pitch_joint": -0.1,
            "R_hip_roll_joint": -0.03,
            "R_hip_yaw_joint": 0.0,
            "R_knee_pitch_joint": -0.204,
            "R_ankle_pitch_joint": 0.1,
            "R_ankle_roll_joint": 0.0,
            "L_hip_pitch_joint": -0.1,
            "L_hip_roll_joint": 0.03,
            "L_hip_yaw_joint": 0.0,
            "L_knee_pitch_joint": -0.204,
            "L_ankle_pitch_joint": 0.1,
            "L_ankle_roll_joint": 0.0,
        }

    class control:
        # [P2] Phase 1 그대로 유지 - 잘 작동하는 셋업 건드리지 말기
        action_scale = 0.1
        control_type = "P"
        stiffness = {
            "torso_yaw_joint": 100.0,
            "R_hip_pitch_joint": 100.0,
            "R_hip_roll_joint": 100.0,
            "R_hip_yaw_joint": 60.0,
            "R_knee_pitch_joint": 150.0,
            "R_ankle_pitch_joint": 60.0,
            "R_ankle_roll_joint": 60.0,
            "L_hip_pitch_joint": 100.0,
            "L_hip_roll_joint": 100.0,
            "L_hip_yaw_joint": 60.0,
            "L_knee_pitch_joint": 150.0,
            "L_ankle_pitch_joint": 60.0,
            "L_ankle_roll_joint": 60.0,
        }
        damping = {
            "torso_yaw_joint": 5.0,
            "R_hip_pitch_joint": 5.0,
            "R_hip_roll_joint": 5.0,
            "R_hip_yaw_joint": 3.0,
            "R_knee_pitch_joint": 8.0,
            "R_ankle_pitch_joint": 4.0,
            "R_ankle_roll_joint": 4.0,
            "L_hip_pitch_joint": 5.0,
            "L_hip_roll_joint": 5.0,
            "L_hip_yaw_joint": 3.0,
            "L_knee_pitch_joint": 8.0,
            "L_ankle_pitch_joint": 4.0,
            "L_ankle_roll_joint": 4.0,
        }
        decimation = 4
        user_torque_limit = 120.0
        torque_limits = {
            "torso_yaw_joint": 60.0,
            "R_hip_pitch_joint": 60.0,
            "R_hip_roll_joint": 60.0,
            "R_hip_yaw_joint": 60.0,
            "R_knee_pitch_joint": 120.0,
            "R_ankle_pitch_joint": 17.0,
            "R_ankle_roll_joint": 17.0,
            "L_hip_pitch_joint": 60.0,
            "L_hip_roll_joint": 60.0,
            "L_hip_yaw_joint": 60.0,
            "L_knee_pitch_joint": 120.0,
            "L_ankle_pitch_joint": 17.0,
            "L_ankle_roll_joint": 17.0,
        }
        max_power = 1000.0

    class asset:
        file = f"{LEGGED_GYM_ROOT_DIR}/resources/robots/qub/urdf/QUB.urdf"
        name = "qub"
        foot_name = "foot"
        foot_radius = 0.03
        penalize_contacts_on = ["thigh", "calf"]
        terminate_after_contacts_on = ["base_link", "pelvis_link"]
        disable_gravity = False
        collapse_fixed_joints = True
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
        # 1. 바닥 마찰력 랜덤화 (얼음판부터 끈적한 고무바닥까지)
        randomize_friction = False
        friction_range = [0.2, 1.25] 

        # 2. 로봇 몸무게 랜덤화 (배터리나 부품이 추가/변경되는 상황 대비, -1kg ~ +3kg)
        randomize_base_mass = False
        added_mass_range = [-1.0, 3.0]

        # 3. 무게 중심 위치 무작위 이동 (조립 오차 대비)
        randomize_base_com = False
        rand_com_vec = [0.03, 0.02, 0.03] # x, y, z 방향으로 최대 3cm 오차

        # 4. 외부 충격 (투명 인간이 10초마다 1.0m/s 속도로 로봇을 뻥뻥 걷어참)
        push_robots = False
        push_interval_s = 10
        max_push_vel_xy = 1.0

        # 5. 모터 제어 지연 (PC와 CAN 통신 사이의 랙(Lag) 시뮬레이션)
        randomize_action_delay = False
        delay_ms_range = [0, 20] # 0 ~ 20ms 사이의 랜덤한 통신 지연 발생

        # 6. 모터 토크 및 Kp, Kd 오차 (실제 모터 스펙이 카탈로그와 다를 때 대비)
        randomize_motor_torque = False
        randomize_motor_torque_range = [0.85, 1.15] # 힘이 15% 빠지거나 15% 쎄짐
        randomize_Kp = False
        randomize_Kp_range = [0.9, 1.1]
        randomize_Kd = False
        randomize_Kd_range = [0.9, 1.1]

        # 나머지 안 쓰는 것들은 False
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
            # ============================================================
            # [P2] Phase 2: Standing reward 유지 + Gait reward 추가
            # 핵심: posture를 약화시켜야 다리 들 수 있음
            # ============================================================

            # --- alive (그대로) ---
            keep_balance = 2.0

            # --- 자세 (Phase 1 유지하되 posture만 약화) ---
            orientation = -1.0
            base_height = -3.0
            # [P2] -2.0 -> -0.5 : 다리를 들어야 보행이라 default pose에서 벗어남이 필요
            posture = -0.2
            ankle_regularization = -0.05   # [P2] -1.0 -> -0.5 약화

            # --- 주저앉기 차단 (그대로) ---
            feet_distance_max = -3.0

            # --- regularization (그대로) ---
            torques = -0.00005
            action_rate = -0.1
            dof_pos_limits = -1.0

            # --- [P2 신규] Gait reward (보행의 핵심) ---
            tracking_contacts_shaped_force = 3.0  # stance phase: 발 GRF 받기 (땅에 붙기)
            tracking_contacts_shaped_vel = 1.5    # swing phase: 발 속도 내기 (공중에 뜨기)
            foot_landing_vel = -0.5               # 부드러운 착지 (충격 줄임)
            feet_regulation = -0.05               # 발 컨트롤 (불필요한 움직임 억제)

            # --- 보행 후 활성화 ---
            tracking_lin_vel = 1.5    # command range 열 때 함께 활성화
            tracking_ang_vel = 1.5
            lin_vel_z = -2.0           # 보행 시 vz가 약간 있는 게 정상이라 일단 OFF
            ang_vel_xy = -0.05
            action_smooth = 0.0
            dof_acc = 0.0
            collision = 0.0
            feet_distance = 0.0
            ang_vel_z = 0.0
            yaw_drift = 0.0
            symmetry = 0.0

        only_positive_rewards = False
        clip_reward = 100
        clip_single_reward = 5
        tracking_sigma = 0.2
        ang_tracking_sigma = 0.25
        height_tracking_sigma = 0.01
        soft_dof_pos_limit = 0.9
        soft_dof_vel_limit = 1.0
        soft_torque_limit = 0.8
        base_height_target = 0.70
        feet_height_target = 0.10
        min_feet_distance = 0.20
        max_feet_distance = 0.35
        # alive gate
        alive_min_height = 0.55
        alive_max_tilt = 0.85
        max_contact_force = 100.0
        kappa_gait_probs = 0.07
        gait_force_sigma = 50.0
        gait_vel_sigma = 0.5
        gait_height_sigma = 0.02
        about_landing_threshold = 0.08

    class normalization:
        class obs_scales:
            lin_vel = 2.0
            ang_vel = 0.25
            dof_pos = 1.0
            dof_vel = 0.05
            torque = 0.05

        clip_observations = 30.0
        clip_actions = 1.0

    class noise:
        add_noise = True
        noise_level = 0.1

        class noise_scales:
            dof_pos = 0.01
            dof_vel = 1.0
            lin_vel = 0.1
            ang_vel = 0.2
            gravity = 0.05

    class viewer:
        ref_env = 0
        pos = [5, -5, 3]
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
            # 잘 작동하는 셋업 그대로
            num_position_iterations = 8
            num_velocity_iterations = 2
            contact_offset = 0.01
            rest_offset = 0.0
            bounce_threshold_velocity = 0.5
            max_depenetration_velocity = 1.0
            max_gpu_contact_pairs = 2**23
            default_buffer_size_multiplier = 5
            contact_collection = 2


class QubCfgPPO(BaseConfig):
    seed = 1
    runner_class_name = "OnPolicyRunner"

    class MLP_Encoder:
        output_detach = True
        num_input_dim = QubCfg.env.num_observations * QubCfg.env.obs_history_length
        num_output_dim = 3
        hidden_dims = [256, 128]
        activation = "elu"
        orthogonal_init = False

    class policy:
        init_noise_std = 0.8
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
        wandb_project = "legged_gym_QUB"
        save_interval = 100
        experiment_name = "qub_flat"
        run_name = ""
        # [P2] Phase 1 체크포인트에서 이어가기
        resume = True
        load_run = "May21_15-12-37_"   # Phase 1 standing 학습 폴더
        checkpoint = -1                  # 최신 체크포인트 사용
        resume_path = "None"
