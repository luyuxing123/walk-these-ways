# License: see [LICENSE, LICENSES/legged_gym/LICENSE]

from params_proto import PrefixProto, ParamsProto


class Cfg(PrefixProto, cli=False):
    class env(PrefixProto, cli=False):
        random_robot_size = True
        run_play = False
        play_robot_scale = 1

        num_envs = 4000
        num_observations = 42
        # if not None a privilige_obs_buf will be returned by step() (critic obs for assymetric training). None is returned otherwise
        num_privileged_obs = 211+18+1+1-187
        privileged_future_horizon = 1
        num_actions = 12
        num_observation_history = 30
        env_spacing = 3.  # not used with heightfields/trimeshes
        send_timeouts = True  # send time out information to the algorithm
        episode_length_s = 20  # episode length in seconds
        observe_vel = True
        observe_only_ang_vel = False
        observe_only_lin_vel = False
        observe_yaw = False
        observe_contact_states = False
        observe_command = False
        observe_two_prev_actions = False
        observe_height_command = False
        observe_gait_commands = True
        observe_timing_parameter = False
        observe_clock_inputs = False
        observe_imu = False
        record_video = False
        recording_width_px = 360
        recording_height_px = 240
        recording_mode = "COLOR"
        num_recording_envs = 1
        debug_viz = False
        all_agents_share = False
        num_scalar_observations = 70

        priv_observe_friction = False
        priv_observe_restitution = False
        priv_observe_friction_indep = False
        priv_observe_ground_friction = False
        priv_observe_ground_friction_per_foot = False
        priv_observe_base_mass = True
        priv_observe_com_displacement = False
        priv_observe_motor_strength = False
        priv_observe_motor_offset = False
        priv_observe_joint_friction = False
        priv_observe_Kp_factor = False
        priv_observe_Kd_factor = False
        priv_observe_contact_forces = False
        priv_observe_contact_states = False
        priv_observe_body_velocity = False
        priv_observe_foot_height = False
        priv_observe_body_height = True
        priv_observe_gravity = False
        priv_observe_terrain_type = False
        priv_observe_clock_inputs = False
        priv_observe_doubletime_clock_inputs = False
        priv_observe_halftime_clock_inputs = False
        priv_observe_desired_contact_states = False
        priv_observe_dummy_variable = False
        priv_observe_motion = False
        priv_observe_gravity_transformed_motion = False
        priv_observe_measure_heights = True  # 187
        priv_observe_foot_forces = True  # 12
        priv_observe_vel = True  # 3
        priv_observe_friction = True  # 1
        priv_observe_collision_state = True  # 8
        priv_observe_stair_height_width = False

    class terrain(PrefixProto, cli=False):
        mesh_type = 'trimesh'  # "heightfield" # none, plane, heightfield or trimesh
        horizontal_scale = 0.1  # [m]
        vertical_scale = 0.005  # [m]
        border_size = 0.0  # 25 # [m]
        curriculum = False
        static_friction = 1.0
        dynamic_friction = 1.0
        restitution = 0.0
        terrain_noise_magnitude = 0.0
        # rough terrain only:
        terrain_smoothness = 0.005
        measure_heights = True
        # 1mx1.6m rectangle (without center line)
        measured_points_x = [-0.8, -0.7, -0.6, -0.5, -0.4, -0.3, -0.2, -0.1, 0., 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
        measured_points_y = [-0.5, -0.4, -0.3, -0.2, -0.1, 0., 0.1, 0.2, 0.3, 0.4, 0.5]

        selected = True  # select a unique terrain type and pass all arguments
        selected_terrain_type = "pyramid_stairs"
        terrain_kwargs = {
            'random_uniform':
                {'min_height': -0.082,
                 'max_height': 0.082,
                 'step': 0.005,
                 'downsampled_scale': 0.2
                 },
            'pyramid_sloped':
                {'slope': -0.45,
                 'platform_size': 3.
                 },
            'pyramid_stairs':
                {'step_width': 0.3,
                 'step_height': 0.1,
                 'platform_size': 1.
                 },
            'discrete_obstacles':
                {
                    'max_height': 0.05,
                    'min_size': 1.,
                    'max_size': 2.,
                    'num_rects': 20,
                    'platform_size': 3.
                }
        }  # Dict of arguments for selected terrain


        min_init_terrain_level = 0
        max_init_terrain_level = 0  # starting curriculum state,初始课程难度，从0～3之间随机采样
        terrain_length = 6  #单个env的长度和宽度
        terrain_width = 6
        num_rows = 22  # number of terrain rows (levels)    地形的row col
        num_cols = 22  # number of terrain cols (types)
        # terrain types: [smooth slope, rough slope, stairs up, stairs down, discrete]
        terrain_proportions = [0.25,0.5,0.75,1]
        # trimesh only:
        slope_treshold = 0.75  # slopes above this threshold will be corrected to vertical surfaces
        difficulty_scale = 1.
        x_init_range = 0.2
        y_init_range = 0.2
        yaw_init_range = 3.14
        x_init_offset = 0.
        y_init_offset = 0.
        teleport_robots = False
        teleport_thresh = 0.3
        max_platform_height = 0.2
        center_robots = True
        center_span = 10    #机器人放置的row和col

    class commands(PrefixProto, cli=False):
        command_curriculum = True
        max_reverse_curriculum = 1.
        max_forward_curriculum = 1.
        yaw_command_curriculum = False
        max_yaw_curriculum = 1.
        exclusive_command_sampling = False
        num_commands = 14
        resampling_time = 20.  # time before command are changed[s]
        subsample_gait = False
        gait_interval_s = 10.  # time between resampling gait params
        vel_interval_s = 10.
        jump_interval_s = 20.  # time between jumps
        jump_duration_s = 0.1  # duration of jump
        jump_height = 0.3
        heading_command = False  # if true: compute ang vel command from heading error
        global_reference = False
        observe_accel = False
        distributional_commands = True
        curriculum_type = "RewardThresholdCurriculum"
        lipschitz_threshold = 0.9

        num_lin_vel_bins = 30
        lin_vel_step = 0.3
        num_ang_vel_bins = 30
        ang_vel_step = 0.3
        distribution_update_extension_distance = 1
        curriculum_seed = 100

        lin_vel_x = [-0.6, 0.6]  # min max [m/s]
        lin_vel_y = [-0.3, 0.3]  # min max [m/s]
        ang_vel_yaw = [-0.6, 0.6]  # min max [rad/s]
        body_height_cmd = [0.25, 0.3]
        gait_phase_cmd_range = [0.0, 1]
        gait_offset_cmd_range = [0.0, 1]
        gait_bound_cmd_range = [0.0, 1]
        gait_frequency_cmd_range = [1.5, 2]
        gait_duration_cmd_range = [0.5, 0.5]
        footswing_height_range = [0.05, 0.25]
        body_pitch_range = [-0.0, 0.0]
        body_roll_range = [-0.0, 0.0]
        aux_reward_coef_range = [0.0, 0.01]
        compliance_range = [0.0, 0.01]
        stance_width_range = [0.27, 0.33]
        stance_length_range = [0.35, 0.4]

        limit_vel_x = [-0.6, 0.6]
        limit_vel_y = [-0.3, 0.3]
        limit_vel_yaw = [-0.6, 0.6]
        limit_body_height = [0.25, 0.3]
        limit_gait_phase = [0, 1]
        limit_gait_offset = [0, 1]
        limit_gait_bound = [0, 1]
        limit_gait_frequency = [1.5, 2]
        limit_gait_duration = [0.5, 0.5]
        limit_footswing_height = [0.05, 0.25]
        limit_body_pitch = [-0.0, 0.0]
        limit_body_roll = [-0.0, 0.0]
        limit_aux_reward_coef = [0.0, 0.01]
        limit_compliance = [0.0, 0.01]
        limit_stance_width = [0.27, 0.33]
        limit_stance_length = [0.35, 0.4]

        num_bins_vel_x = 1
        num_bins_vel_y = 1
        num_bins_vel_yaw = 1
        num_bins_body_height = 1
        num_bins_gait_frequency = 1
        num_bins_gait_phase = 1
        num_bins_gait_offset = 1
        num_bins_gait_bound = 1
        num_bins_gait_duration = 1
        num_bins_footswing_height = 1
        num_bins_body_pitch = 1
        num_bins_body_roll = 1
        num_bins_aux_reward_coef = 1
        num_bins_compliance = 1
        num_bins_compliance = 1
        num_bins_stance_width = 1
        num_bins_stance_length = 1
        min_vel = 0.15

        heading = [-3.14, 3.14]
        impulse_height_commands = False
        exclusive_phase_offset = False
        binary_phases = True
        pacing_offset = False
        balance_gait_distribution = True
        gaitwise_curricula = True

    class curriculum_thresholds(PrefixProto, cli=False):
        tracking_ang_vel = 0.7
        tracking_lin_vel = 0.8
        tracking_contacts_shaped_vel = 0.90
        tracking_contacts_shaped_force = 0.90

    class init_state(PrefixProto, cli=False):
        pos = [0.0, 0.0, 1.]  # x,y,z [m]
        rot = [0.0, 0.0, 0.0, 1.0]  # x,y,z,w [quat]
        lin_vel = [0.0, 0.0, 0.0]  # x,y,z [m/s]
        ang_vel = [0.0, 0.0, 0.0]  # x,y,z [rad/s]
        # target angles when action = 0.0
        default_joint_angles = {"joint_a": 0., "joint_b": 0.}

    class control(PrefixProto, cli=False):
        control_type = 'P' #'P'  # P: position, V: velocity, T: torques
        # PD Drive parameters:
        stiffness = {'joint_a': 10.0, 'joint_b': 15.}  # [N*m/rad]
        damping = {'joint_a': 1.0, 'joint_b': 1.5}  # [N*m*s/rad]
        # action scale: target angle = actionScale * action + defaultAngle
        action_scale = 0.5
        hip_scale_reduction = 1.0
        # decimation: Number of control action updates @ sim DT per policy DT
        decimation = 4

    class asset(PrefixProto, cli=False):
        file = ""
        foot_name = "None"  # name of the feet bodies, used to index body state and contact force tensors
        penalize_contacts_on = []
        terminate_after_contacts_on = []
        disable_gravity = False
        # merge bodies connected by fixed joints. Specific fixed joints can be kept by adding " <... dont_collapse="true">
        collapse_fixed_joints = True
        fix_base_link = False  # fixe the base of the robot
        default_dof_drive_mode = 3  # see GymDofDriveModeFlags (0 is none, 1 is pos tgt, 2 is vel tgt, 3 effort)
        self_collisions = 0  # 1 to disable, 0 to enable...bitwise filter
        # replace collision cylinders with capsules, leads to faster/more stable simulation
        replace_cylinder_with_capsule = True
        flip_visual_attachments = True  # Some .obj meshes must be flipped from y-up to z-up

        density = 0.001
        angular_damping = 0.
        linear_damping = 0.
        max_angular_velocity = 1000.
        max_linear_velocity = 1000.
        armature = 0.
        thickness = 0.01

    class domain_rand(PrefixProto, cli=False):
        rand_interval_s = 10
        randomize_rigids_after_start = False
        randomize_friction_indep = False
        friction_range = [0.1, 3.0]  # increase range
        restitution_range = [0, 0.4]
        restitution = 0.5
        # add link masses, increase range, randomize inertia, randomize joint properties
        added_mass_range = [-3, 8]
        randomize_com_displacement = False
        # add link masses, increase range, randomize inertia, randomize joint properties
        com_displacement_range = [-0.15, 0.15]
        motor_strength_range = [0.9, 1.1]
        randomize_Kp_factor = False
        Kp_factor_range = [0.8, 1.3]
        randomize_Kd_factor = False
        Kd_factor_range = [0.5, 1.5]
        gravity_rand_interval_s = 4
        gravity_impulse_duration = 0.99
        gravity_range = [-1.0, 1.0]
        push_robots = False
        push_interval_s = 15
        max_push_vel_xy = 0.5
        lag_timesteps = 6
        ground_friction_range = [0.0, 0.0]
        randomize_motor_offset = True
        motor_offset_range = [-0.02, 0.02]
        tile_height_range = [-0.0, 0.0]
        tile_height_curriculum = False
        tile_height_update_interval = 1000000
        tile_height_curriculum_step = 0.01
        randomize_ground_friction = False

        randomize_lag_timesteps = True
        randomize_friction = True
        randomize_restitution = True
        randomize_base_mass = True
        randomize_motor_strength = True
        randomize_gravity = True

    class rewards(PrefixProto, cli=False):
        only_positive_rewards = False  # if true negative total rewards are clipped at zero (avoids early termination problems)
        only_positive_rewards_ji22_style = True
        sigma_rew_neg = 0.02
        reward_container_name = "CoRLRewards"
        tracking_sigma = 0.15  # tracking reward = exp(-error^2/sigma)
        tracking_sigma_lat = 0.25  # tracking reward = exp(-error^2/sigma)
        tracking_sigma_long = 0.25  # tracking reward = exp(-error^2/sigma)
        tracking_sigma_yaw = 0.25  # tracking reward = exp(-error^2/sigma)
        soft_dof_pos_limit = 0.9  # percentage of urdf limits, values above this limit are penalized
        soft_dof_vel_limit = 1.
        soft_torque_limit = 1.
        base_height_target = 0.28
        max_contact_force = 100.  # forces above this value are penalized
        use_terminal_body_height = True
        use_terminal_air = True
        terminal_body_height = 0.05
        use_terminal_foot_height = False
        terminal_foot_height = -0.005
        use_terminal_roll_pitch = True
        terminal_body_ori = 1.6
        kappa_gait_probs = 0.05
        gait_force_sigma = 50.
        gait_vel_sigma = 1.25
        footswing_height = 0.09
        max_collision_force = 2

    class reward_scales(ParamsProto, cli=False):
        tracking_lin_vel = 1      #1
        tracking_ang_vel = 0.5     #0.5
        torques = -1e-4  #-0.0001
        dof_acc = -2.5e-7
        action_smoothness_1 = -0.1 #-0.1
        base_height = -1    #-1
        collision = -1    #-5.0

        tracking_contacts_shaped_force = 1
        tracking_contacts_shaped_vel = 1

        lin_vel_z = -0.5 #-0.5
        ang_vel_xy = -2  #-0.05
        orientation = -2 #-0.5
        raibert_heuristic = -10    #-10
        dof_vel = -1e-4 #-1e-4

    class normalization(PrefixProto, cli=False):
        clip_observations = 100.
        clip_actions = 10.

        friction_range = [0, 1]
        ground_friction_range = [0, 1]
        restitution_range = [0, 1.0]
        added_mass_range = [-1., 3.]
        com_displacement_range = [-0.1, 0.1]
        motor_strength_range = [0.9, 1.1]
        motor_offset_range = [-0.05, 0.05]
        Kp_factor_range = [0.8, 1.3]
        Kd_factor_range = [0.5, 1.5]
        joint_friction_range = [0.0, 0.7]
        contact_force_range = [0.0, 50.0]
        contact_state_range = [0.0, 1.0]
        body_velocity_range = [-6.0, 6.0]
        foot_height_range = [0.0, 0.15]
        body_height_range = [0.0, 0.60]
        gravity_range = [-1.0, 1.0]
        motion = [-0.01, 0.01]

    class obs_scales(PrefixProto, cli=False):
        lin_vel = 2
        ang_vel = 0.25
        dof_pos = 1.0
        dof_vel = 0.05

        imu = 0.1
        height_measurements = 5.0
        friction_measurements = 1.0
        body_height_cmd = 2.0
        gait_phase_cmd = 1.0
        gait_freq_cmd = 1.0
        footswing_height_cmd = 0.15
        body_pitch_cmd = 0.3
        body_roll_cmd = 0.3
        aux_reward_cmd = 1.0
        compliance_cmd = 1.0
        stance_width_cmd = 1.0
        stance_length_cmd = 1.0
        segmentation_image = 1.0
        rgb_image = 1.0
        depth_image = 1.0

    class noise(PrefixProto, cli=False):
        add_noise = True
        noise_level = 1.0  # scales other values

    class noise_scales(PrefixProto, cli=False):
        dof_pos = 0.01
        dof_vel = 1.5
        lin_vel = 0.1
        ang_vel = 0.2
        imu = 0.1
        gravity = 0.05
        contact_states = 0.05
        height_measurements = 0.1
        friction_measurements = 0.0
        segmentation_image = 0.0
        rgb_image = 0.0
        depth_image = 0.0

    # viewer camera:
    class viewer(PrefixProto, cli=False):
        ref_env = 0
        pos = [60, 60, 10]  # [m]
        lookat = [75., 75, 3.]  # [m]

    class sim(PrefixProto, cli=False):
        dt = 0.005
        substeps = 1
        gravity = [0., 0., -9.81]  # [m/s^2]
        up_axis = 1  # 0 is y, 1 is z

        use_gpu_pipeline = True

        class physx(PrefixProto, cli=False):
            num_threads = 10
            solver_type = 1  # 0: pgs, 1: tgs
            num_position_iterations = 4
            num_velocity_iterations = 0
            contact_offset = 0.01  # [m]
            rest_offset = 0.0  # [m]
            bounce_threshold_velocity = 0.5  # 0.5 [m/s]
            max_depenetration_velocity = 1.0
            max_gpu_contact_pairs = 2 ** 23  # 2**24 -> needed for 8000 envs and more
            default_buffer_size_multiplier = 5
            contact_collection = 2  # 0: never, 1: last sub-step, 2: all sub-steps (default=2)
