import isaacgym

assert isaacgym
import torch
import numpy as np

import glob
import pickle as pkl
from go1_gym_learn.ppo_cse.actor_critic import AC_Args
from go1_gym_learn.ppo_cse.actor_critic import ActorCritic
from go1_gym.envs import *
from go1_gym.envs.base.legged_robot_config import Cfg
from go1_gym.envs.go1.go1_config import config_go1
from go1_gym.envs.go1.velocity_tracking import VelocityTrackingEasyEnv

from tqdm import tqdm

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

label = "gait-conditioned-agility/2024-10-27/train"
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
    Cfg.terrain.curriculum = False

    Cfg.viewer.pos=[10,10,6]
    Cfg.viewer.lookat=[11,5,3]

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

    # label = "gait-conditioned-agility/pretrain-v0/train"

    env, policy = load_env(label, headless=headless)

    num_eval_steps = 2500
    gaits = {"pronking": [0, 0, 0],
             "trotting": [0.5, 0, 0],
             "bounding": [0, 0.5, 0],
             "pacing": [0, 0, 0.5]}

    x_vel_cmd, y_vel_cmd, yaw_vel_cmd = 0.3, 0.0, 0.0
    body_height_cmd = 0.0
    step_frequency_cmd = 3.0
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
        env.commands[:, 0] = x_vel_cmd
        env.commands[:, 1] = y_vel_cmd
        env.commands[:, 2] = yaw_vel_cmd
        # env.commands[:, 3] = body_height_cmd
        # env.commands[:, 4] = step_frequency_cmd
        # env.commands[:, 5:8] = gait
        # env.commands[:, 8] = 0.5
        # env.commands[:, 9] = footswing_height_cmd
        # env.commands[:, 10] = pitch_cmd
        # env.commands[:, 11] = roll_cmd
        # env.commands[:, 12] = stance_width_cmd
        obs, rew, done, info = env.step(actions)

        # measured_x_vels[i] = env.base_lin_vel[0, 0]
        # joint_positions[i] = env.dof_pos[0, :].cpu()

    # plot target and measured forward velocity
    # from matplotlib import pyplot as plt
    # fig, axs = plt.subplots(2, 1, figsize=(12, 5))
    # axs[0].plot(np.linspace(0, num_eval_steps * env.dt, num_eval_steps), measured_x_vels, color='black', linestyle="-", label="Measured")
    # axs[0].plot(np.linspace(0, num_eval_steps * env.dt, num_eval_steps), target_x_vels, color='black', linestyle="--", label="Desired")
    # axs[0].legend()
    # axs[0].set_title("Forward Linear Velocity")
    # axs[0].set_xlabel("Time (s)")
    # axs[0].set_ylabel("Velocity (m/s)")
    #
    # axs[1].plot(np.linspace(0, num_eval_steps * env.dt, num_eval_steps), joint_positions, linestyle="-", label="Measured")
    # axs[1].set_title("Joint Positions")
    # axs[1].set_xlabel("Time (s)")
    # axs[1].set_ylabel("Joint Position (rad)")
    #
    # plt.tight_layout()
    # plt.show()


if __name__ == '__main__':
    # to see the environment rendering, set headless=False
    play_go1(headless=False)