import isaacgym
import torch
import torch
if not torch.cuda.is_available():
    exit("cuda not available")

import numpy as np
import glob
import pickle as pkl
from go1_gym.envs import *
from go1_gym.envs.base.legged_robot_config import Cfg
from go1_gym.envs.go1.go1_config import config_go1
from go1_gym.envs.go1.velocity_tracking import VelocityTrackingEasyEnv
from go1_gym_learn.ppo_cse import ActorCritic


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

joy_cmd = [0.0, 0.0, 0.0]

def joy_callback(joy_msg):
    global joy_cmd
    global stop
    global begin
    joy_cmd[0] =  0.6*joy_msg.axes[1]
    joy_cmd[1] =  0.6*joy_msg.axes[0]
    joy_cmd[2] =  0.6*joy_msg.axes[3]  # 横向操作


def load_policy(logdir):
    body = ActorCritic(Cfg.env.num_observations,
                       Cfg.env.num_privileged_obs,
                       Cfg.env.num_observation_history * Cfg.env.num_observations,
                       Cfg.env.num_actions,
                       ).to(device)
    state_dict = torch.load(logdir + '/checkpoints/ac_weights_last.pt')
    # state_dict = torch.load(logdir + '/checkpoints/ac_weights_003200.pt')
    body.load_state_dict(state_dict)
    body.eval()

    def policy(obs, info={}):
        command = obs["privileged_obs"][:, -13:]
        vt = body.adaptation_module(obs["obs_history"])
        ht = body.history_encoder(obs["obs_history"])
        action = body.actor_body(torch.cat((ht,vt,command), dim=-1))
        return action

    return policy


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

    Cfg.domain_rand.lag_timesteps = 6
    Cfg.domain_rand.randomize_lag_timesteps = True
    Cfg.control.control_type = "actuator_net"

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
    Cfg.terrain.num_rows = 2
    Cfg.terrain.num_cols = 2

    Cfg.pos = [0.0, 0.0, 0.]

    from go1_gym.envs.wrappers.history_wrapper import HistoryWrapper

    env = VelocityTrackingEasyEnv(sim_device='cuda:0', headless=False, cfg=Cfg)
    env = HistoryWrapper(env)

    # load policy
    from ml_logger import logger
    from go1_gym_learn.ppo_cse.actor_critic import ActorCritic

    policy = load_policy(logdir)

    return env, policy


def play_go1(headless=True):
    from ml_logger import logger

    from pathlib import Path
    from go1_gym import MINI_GYM_ROOT_DIR
    import glob
    import os
    import rospy
    from sensor_msgs.msg import Joy

    rospy.init_node('play')
    rospy.Subscriber('/joy', Joy, joy_callback, queue_size=10)

    # label = "gait-conditioned-agility/pretrain-v0/train"
    label = "gait-conditioned-agility/2024-11-25/train"

    env, policy = load_env(label, headless=headless)

    num_eval_steps = 250
    gaits = {"pronking": [0, 0, 0],
             "trotting": [0.5, 0, 0],
             "bounding": [0, 0.5, 0],
             "pacing": [0, 0, 0.5]}

    x_vel_cmd, y_vel_cmd, yaw_vel_cmd = 1.5, 0.0, 0.0
    body_height_cmd = 0.3
    step_frequency_cmd = 2
    gait = torch.tensor(gaits["trotting"])
    footswing_height_cmd = 0.08
    pitch_cmd = 0.0
    roll_cmd = 0.0
    stance_width_cmd = 0.25

    measured_x_vels = np.zeros(num_eval_steps)
    target_x_vels = np.ones(num_eval_steps) * x_vel_cmd
    joint_positions = np.zeros((num_eval_steps, 12))

    obs = env.reset()

    while True:
        with torch.no_grad():
            actions = policy(obs)
        env.commands[:, 0] = joy_cmd[0]
        env.commands[:, 1] = joy_cmd[1]
        env.commands[:, 2] = joy_cmd[2]
        env.commands[:, 3] = 0.3
        env.commands[:, 4] = 1.5
        env.commands[:, 5:8] = gait
        env.commands[:, 8] = 0.5
        obs, rew, done, info,_,_ = env.step(actions)


if __name__ == '__main__':
    # to see the environment rendering, set headless=False
    play_go1(headless=False)

