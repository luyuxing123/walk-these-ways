import isaacgym

assert isaacgym
import torch
import numpy as np

import glob
import pickle as pkl
import threading
from go1_gym.envs import *
from go1_gym.envs.base.legged_robot_config import Cfg
from go1_gym.envs.go1.go1_config import config_go1
from go1_gym.envs.go1.velocity_tracking import VelocityTrackingEasyEnv
from go1_gym_learn.ppo_cse import ActorCritic
from go1_gym_learn.ppo_cse import RunnerArgs
from tqdm import tqdm
from go1_gym_learn.ppo_cse import RunnerArgs
from go1_gym_learn.ppo_cse.actor_critic import AC_Args

joy_cmd = [0.0, 0.0, 0.0]
stop=False
begin=True

def joy_callback(joy_msg):
    global joy_cmd
    global stop
    global begin
    joy_cmd[0] =  0.3*joy_msg.axes[1]
    joy_cmd[1] =  0.3*joy_msg.axes[0]
    joy_cmd[2] =  0.3*joy_msg.axes[3]  # 横向操作

    if(joy_cmd[0]>3 ):
        joy_cmd[0]=3
    if(joy_cmd[0]<-3):
        joy_cmd[0]=-3
    if(joy_cmd[1]>0.6):
        joy_cmd[1]=0.6
    if (joy_cmd[1] < -0.6):
        joy_cmd[1] = -0.6
    if (joy_cmd[2] > 3):
        joy_cmd[2] = 3
    if (joy_cmd[2] < -3 ):
        joy_cmd[2] = -3

    if(joy_msg.buttons[1]):
        stop=True
    if(joy_msg.buttons[0]):
        begin=True

use_jit = False
is_teacher = False
AC_Args.adaptation_lstm = False
AC_Args.use_help_decoder = True
AC_Args.perception_dim = 49
AC_Args.privileged_dim = 24#+2
AC_Args.terrain_dim = 187
AC_Args.actor_dim = AC_Args.perception_dim + AC_Args.privileged_lantent_size + AC_Args.terrain_lantent_size

if is_teacher:
    Cfg.env.num_observations = AC_Args.perception_dim + AC_Args.privileged_dim + AC_Args.terrain_dim
    Cfg.env.num_privileged_obs = 0
    Cfg.env.num_observation_history = 1
    Cfg.env.num_privileged_obs = 0
    Cfg.env.observe_vel = True
    Cfg.env.observe_foot_forces = True
    Cfg.env.observe_friction = True
    Cfg.env.observe_collision_state =True
    Cfg.env.observe_measure_heights = True
    Cfg.env.observe_stair_height_width = True

    Cfg.env.priv_observe_vel = False
    Cfg.env.priv_observe_foot_forces = False
    Cfg.env.priv_observe_friction = False
    Cfg.env.priv_observe_collision_state =False
    Cfg.env.priv_observe_measure_heights = False
    Cfg.env.priv_observe_stair_height_width = False
    Cfg.noise.add_noise = False
else:
    Cfg.env.num_observations = AC_Args.perception_dim
    Cfg.env.num_privileged_obs = AC_Args.privileged_dim + AC_Args.terrain_dim
    Cfg.env.num_observation_history=30
    Cfg.env.observe_vel = False
    Cfg.env.observe_foot_forces = False
    Cfg.env.observe_friction = False
    Cfg.env.observe_collision_state =False
    Cfg.env.observe_measure_heights = False
    Cfg.env.observe_stair_height_width = False

    Cfg.env.priv_observe_vel = True
    Cfg.env.priv_observe_foot_forces = True
    Cfg.env.priv_observe_friction = True
    Cfg.env.priv_observe_collision_state =True
    Cfg.env.priv_observe_measure_heights = True
    Cfg.env.priv_observe_stair_height_width = True
    Cfg.noise.add_noise = True

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_policy(logdir):
    if use_jit:
        body = torch.jit.load(logdir + '/checkpoints/body_latest.jit').to(device)
    else:
        body = ActorCritic(        Cfg.env.num_observations,
                                   Cfg.env.num_privileged_obs,
                                   Cfg.env.num_observation_history*Cfg.env.num_observations,
                                   Cfg.env.num_actions,
                                   is_teacher).to(device)
        # state_dict = torch.load(logdir + '/checkpoints/ac_weights_000400.pt')
        state_dict = torch.load(logdir + '/checkpoints/ac_weights_last.pt')
        body.load_state_dict(state_dict)
        body.eval()

    import os
    # if not is_teacher:
    #     adaptation_module = torch.jit.load(logdir + '/checkpoints/adaptation_module_latest.jit').to(device)

    def policy(obs, info={}):
        if is_teacher:
            action = body.act_teacher(obs["obs_history"]).to(device)
        else:
            # latent = adaptation_module.forward(obs["obs_history"].to(device))
            # latent = adaptation_module.forward(input)
            # latent = latent.unsqueeze(0)
            # print(latent)
            if use_jit:
                action = body(obs["obs_history"])
            else:
                action = body.act_student(obs["obs_history"])
        return action

    return policy

label = "gait-conditioned-agility/2024-10-26/train"
# label = "gait-conditioned-agility/pretrain-v0/train"

def load_env(label, headless=False):
    dirs = glob.glob(f"../runs/{label}/*")
    logdir = sorted(dirs)[-1]
    with open(logdir + "/parameters.pkl", 'rb') as file:
        pkl_cfg = pkl.load(file)
        print(pkl_cfg.keys())
        cfg = pkl_cfg["Cfg"]
        print(cfg.keys())


        for key, value in cfg.items():
            if hasattr(Cfg, key):
                for key2, value2 in cfg[key].items():
                    setattr(getattr(Cfg, key), key2, value2)

    # turn off DR for evaluation script
    Cfg.domain_rand.push_robots = False
    Cfg.domain_rand.randomize_friction = False
    Cfg.domain_rand.randomize_gravity = False
    Cfg.domain_rand.randomize_restitution = False
    Cfg.domain_rand.randomize_motor_offset = False
    Cfg.domain_rand.randomize_motor_strength = False
    Cfg.domain_rand.randomize_friction_indep = False
    Cfg.domain_rand.randomize_ground_friction = False
    Cfg.domain_rand.randomize_base_mass = False
    Cfg.domain_rand.randomize_Kd_factor = False
    Cfg.domain_rand.randomize_Kp_factor = False
    Cfg.domain_rand.randomize_joint_friction = False
    Cfg.domain_rand.randomize_com_displacement = False


    Cfg.env.num_recording_envs = 1
    Cfg.env.num_envs = 1
    Cfg.terrain.num_rows = 5
    Cfg.terrain.num_cols = 5
    Cfg.terrain.border_size = 0
    Cfg.terrain.center_robots = True
    Cfg.terrain.center_span = 1
    Cfg.terrain.teleport_robots = True
    Cfg.env.episode_length_s = 100

    Cfg.domain_rand.lag_timesteps = 6
    Cfg.domain_rand.randomize_lag_timesteps = True
    Cfg.control.control_type = "P"

    Cfg.viewer.pos=[10,10,6]
    Cfg.viewer.lookat=[11,5,3]

    Cfg.terrain.curriculum=False
    Cfg.terrain.selected = True
    Cfg.terrain.mesh_type = 'trimesh'
    Cfg.terrain.selected_terrain_type = "pyramid_stairs"
    Cfg.terrain.terrain_kwargs = {
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
             'platform_size': 2
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
    Cfg.terrain.terrain_length = 8.
    Cfg.terrain.terrain_width = 8.

    Cfg.env.use_teacher = True

    from go1_gym.envs.wrappers.history_wrapper import HistoryWrapper

    env = VelocityTrackingEasyEnv(sim_device='cuda:0', headless=False, cfg=Cfg)
    env = HistoryWrapper(env)

    # load policy
    from ml_logger import logger
    from go1_gym_learn.ppo_cse.actor_critic import ActorCritic

    policy = load_policy(logdir)

    return env, policy



def read_input():
    global begin
    while True:
        user_input = input()
        if user_input == 'a':
            begin = True


def play_go1(headless=True):
    from ml_logger import logger

    from pathlib import Path
    from go1_gym import MINI_GYM_ROOT_DIR
    import glob
    import os
    import rospy
    from sensor_msgs.msg import Joy

    thread = threading.Thread(target=read_input)
    thread.start()

    rospy.init_node('play')
    rospy.Subscriber('/joy', Joy, joy_callback, queue_size=10)

    env, policy = load_env(label, headless=headless)

    Cfg.control.stiffness={'joint': 100}

    num_eval_steps = 20000
    gaits = {"pronking": [0, 0, 0],
             "trotting": [0.5, 0, 0],
             "bounding": [0, 0.5, 0],
             "pacing": [0, 0, 0.5]}

    body_height_cmd = 0
    step_frequency_cmd = 2
    gait = torch.tensor(gaits["trotting"])
    footswing_height_cmd = 0.1
    pitch_cmd = 0.0
    roll_cmd = 0.0
    stance_width_cmd = 0.3
    stance_length_cmd = 0.38

    measured_x_vels = []
    target_x_vels = []
    joint_positions = []

    obs = env.reset()
    # print(obs)

    i=0

    while(1):
        with torch.no_grad():
            actions = policy(obs)
        env.commands[:, 0] = joy_cmd[0]
        env.commands[:, 1] = joy_cmd[1]
        env.commands[:, 2] = joy_cmd[2]
        # env.commands[:, 3] = body_height_cmd
        # env.commands[:, 4] = step_frequency_cmd
        # env.commands[:, 5:8] = gait
        # env.commands[:, 8] = 0.5
        # env.commands[:, 9] = footswing_height_cmd
        # env.commands[:, 10] = pitch_cmd
        # env.commands[:, 11] = roll_cmd


        if Cfg.commands.num_commands>=13:
            env.commands[:, 12] = stance_width_cmd
            env.commands[:, 13] = stance_length_cmd
            env.commands[:, 14] = 0

        if(not begin):
            actions=torch.tensor([[0,0,0,0,0,0,0,0,0,0,0,0]])

        obs, rew, done, info = env.step(actions)
        # print(actions)
        # print(env.foot_positions[:, :, 2])
        # print(env.base_pos[:, 2])

        measured_x_vels.append(env.base_lin_vel[0, 0].cpu().numpy())
        joint_positions.append(env.dof_pos[0, :].cpu().numpy())
        target_x_vels.append (joy_cmd[0])

        # print(env.projected_gravity)

        i=i+1

        # print(env.base_pos[:, 2])

        if(stop):
            break

    # plot target and measured forward velocity
    measured_x_vels = np.array(measured_x_vels)
    target_x_vels = np.array(target_x_vels)
    joint_positions = np.array(joint_positions)
    from matplotlib import pyplot as plt
    fig, axs = plt.subplots(2, 1, figsize=(12, 5))
    axs[0].plot(np.linspace(0, i * env.dt, i), measured_x_vels, color='black', linestyle="-", label="Measured")
    axs[0].plot(np.linspace(0, i * env.dt, i), target_x_vels, color='black', linestyle="--", label="Desired")
    axs[0].legend()
    axs[0].set_title("Forward Linear Velocity")
    axs[0].set_xlabel("Time (s)")
    axs[0].set_ylabel("Velocity (m/s)")

    # axs[1].plot(np.linspace(0, num_eval_steps * env.dt, num_eval_steps), joint_positions, linestyle="-", label="Measured")
    # axs[1].set_title("Joint Positions")
    # axs[1].set_xlabel("Time (s)")
    # axs[1].set_ylabel("Joint Position (rad)")

    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    # to see the environment rendering, set headless=False
    play_go1(headless=False)
