def train_go1(headless=True):

    import isaacgym
    assert isaacgym
    import torch

    from go1_gym.envs.base.legged_robot_config import Cfg
    from go1_gym.envs.go1.go1_config import config_go1
    from go1_gym.envs.go1.velocity_tracking import VelocityTrackingEasyEnv

    from ml_logger import logger

    from go1_gym_learn.ppo_cse import Runner
    from go1_gym.envs.wrappers.history_wrapper import HistoryWrapper
    from go1_gym_learn.ppo_cse.actor_critic import AC_Args
    from go1_gym_learn.ppo_cse.ppo import PPO_Args
    from go1_gym_learn.ppo_cse import Runner_Student
    from go1_gym_learn.ppo_cse import RunnerArgs
    from go1_gym_learn.ppo_cse import RunnerArgs_Student
    import os

    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:64"

    config_go1(Cfg)

    use_teacher = False
    #第一次的
    # teacher_path = "/home/ask-spc/luyx/walk-these-ways/runs/gait-conditioned-agility/2024-10-26/train/065713.860842"
    #仅使用26.5cm宽度楼梯
    teacher_path = "/home/ask-spc/luyx/walk-these-ways/runs/gait-conditioned-agility/2024-10-27/train/091629.309567"
    AC_Args.perception_dim = 49
    AC_Args.privileged_dim = 24#+2
    AC_Args.terrain_dim = 187
    AC_Args.actor_dim = AC_Args.perception_dim + AC_Args.privileged_lantent_size + AC_Args.terrain_lantent_size
    # Cfg.env.num_envs = 1

    if use_teacher:
        Cfg.env.num_observations = AC_Args.perception_dim + AC_Args.privileged_dim + AC_Args.terrain_dim
        Cfg.env.num_privileged_obs = 0
        Cfg.env.num_observation_history = 1
        Cfg.env.num_privileged_obs = 0
        Cfg.env.observe_vel = True
        Cfg.env.observe_foot_forces = True
        Cfg.env.observe_friction = True
        Cfg.env.observe_collision_state =True
        Cfg.env.observe_measure_heights = True
        Cfg.env.observe_stair_height_width = False

        Cfg.env.priv_observe_vel = False
        Cfg.env.priv_observe_foot_forces = False
        Cfg.env.priv_observe_friction = False
        Cfg.env.priv_observe_collision_state =False
        Cfg.env.priv_observe_measure_heights = False
        Cfg.env.priv_observe_stair_height_width = False
        Cfg.noise.add_noise = False
    else:
        AC_Args.adaptation_lstm = False
        AC_Args.use_help_decoder = True
        if AC_Args.adaptation_lstm:
            Cfg.env.num_observation_history = 1
        else:
            Cfg.env.num_observation_history = AC_Args.adaptation_history_nums
        Cfg.env.num_envs = 4096
        Cfg.env.num_observations = AC_Args.perception_dim
        Cfg.env.num_privileged_obs = AC_Args.privileged_dim + AC_Args.terrain_dim
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
        Cfg.env.priv_observe_stair_height_width = False
        Cfg.noise.add_noise = True
        # Cfg.terrain.curriculum = False


    env = VelocityTrackingEasyEnv(sim_device='cuda:0', headless=False, cfg=Cfg)

    # log the experiment parameters
    logger.log_params(AC_Args=vars(AC_Args), PPO_Args=vars(PPO_Args), RunnerArgs=vars(RunnerArgs),
                      Cfg=vars(Cfg))

    env = HistoryWrapper(env)
    gpu_id = 0
    if use_teacher:
        runner = Runner(env, device=f"cuda:{gpu_id}")
        runner.learn(num_learning_iterations=100000, init_at_random_ep_len=True, eval_freq=100)
        logger.log_params(AC_Args=vars(AC_Args), PPO_Args=vars(PPO_Args), RunnerArgs=vars(RunnerArgs),
                          Cfg=vars(Cfg))
    else:
        runner = Runner_Student(env, teacher_path,device=f"cuda:{gpu_id}")
        runner.learn(num_learning_iterations=100000, init_at_random_ep_len=True, eval_freq=100)
        logger.log_params(AC_Args=vars(AC_Args), PPO_Args=vars(PPO_Args), RunnerArgs=vars(RunnerArgs_Student),
                          Cfg=vars(Cfg))


if __name__ == '__main__':
    from pathlib import Path
    from ml_logger import logger
    from go1_gym import MINI_GYM_ROOT_DIR

    stem = Path(__file__).stem
    logger.configure(logger.utcnow(f'gait-conditioned-agility/%Y-%m-%d/{stem}/%H%M%S.%f'),
                     root=Path(f"{MINI_GYM_ROOT_DIR}/runs").resolve(), )
    logger.log_text("""
                charts: 
                - yKey: train/episode/rew_total/mean
                  xKey: iterations
                - yKey: train/episode/rew_tracking_lin_vel/mean
                  xKey: iterations
                - yKey: train/episode/rew_tracking_ang_vel/mean
                  xKey: iterations
                - yKey: train/episode/rew_tracking_contacts_shaped_force/mean
                  xKey: iterations
                - yKey: train/episode/rew_action_smoothness_1/mean
                  xKey: iterations
                - yKey: train/episode/rew_action_smoothness_2/mean
                  xKey: iterations
                - yKey: train/episode/rew_tracking_contacts_shaped_vel/mean
                  xKey: iterations
                - type: video
                  glob: "videos/*.mp4"
                - yKey: adaptation_loss/mean
                  xKey: iterations
                - yKey: act_loss/mean
                  xKey: iterations
                - yKey: critic_loss/mean
                  xKey: iterations
                """, filename=".charts.yml", dedent=True)

    # to see the environment rendering, set headless=False
    train_go1(headless=False)
