import time
from collections import deque
import copy
import os
from copy import deepcopy
import torch
from ml_logger import logger
from params_proto import PrefixProto
from .actor_critic import AC_Args
from .actor_critic import ActorCritic
from .rollout_storage import RolloutStorage

import torch.nn as nn
from .ppo import PPO_Args


def class_to_dict(obj) -> dict:
    if not hasattr(obj, "__dict__"):
        return obj
    result = {}
    for key in dir(obj):
        if key.startswith("_") or key == "terrain":
            continue
        element = []
        val = getattr(obj, key)
        if isinstance(val, list):
            for item in val:
                element.append(class_to_dict(item))
        else:
            element = class_to_dict(val)
        result[key] = element
    return result


class DataCaches:
    def __init__(self, curriculum_bins):
        from go1_gym_learn.ppo.metrics_caches import SlotCache, DistCache

        self.slot_cache = SlotCache(curriculum_bins)
        self.dist_cache = DistCache()


caches = DataCaches(1)


class RunnerArgs_Student(PrefixProto, cli=False):
    # runner
    algorithm_class_name = 'RMA'
    num_steps_per_env = 24  # per iteration
    max_iterations = 1500  # number of policy updates

    # logging
    save_interval = 400  # check for potential saves every this many iterations
    save_video_interval = 100
    log_freq = 10

    # load and resume
    resume = False
    load_run = -1  # -1 = last run
    checkpoint = -1  # -1 = last saved model
    resume_path = "/home/ask-spc/luyx/walk-these-ways/runs/gait-conditioned-agility/2024-10-10/train/091732.372596"  # updated from load_run and chkpt
    resume_curriculum = True


class Runner_Student:

    def __init__(self, env, teacher_path,device='cpu'):
        from .ppo import PPO
        import pickle

        self.device = device
        self.env = env

        is_teacher = True

        teacher_policy = ActorCritic(self.env.num_obs,
                                      self.env.num_privileged_obs,
                                      self.env.num_obs_history,
                                      self.env.num_actions,
                                      is_teacher).to(self.device)

        teacher_pt_path = teacher_path + "/checkpoints/ac_weights_last.pt"

        state_dict = torch.load(teacher_pt_path)
        teacher_policy.load_state_dict(state_dict)
        teacher_policy.eval()

        is_teacher=False

        actor_critic = ActorCritic(self.env.num_obs,
                                      self.env.num_privileged_obs,
                                      self.env.num_obs_history,
                                      self.env.num_actions,
                                      is_teacher).to(self.device)

        actor_critic.actor_body = deepcopy(teacher_policy.actor_body)
        actor_critic.critic_body = deepcopy(teacher_policy.critic_body)

        if RunnerArgs_Student.resume:
            # load pretrained weights from resume_path1
            from ml_logger import ML_Logger
            loader = ML_Logger(root="http://127.0.0.1:8081",
                               prefix=RunnerArgs_Student.resume_path)
            # weights = loader.load_torch("checkpoints/ac_weights_last.pt")
            weights = torch.load(RunnerArgs_Student.resume_path + "/checkpoints/ac_weights_last.jit")
            # actor_critic.load_state_dict(state_dict=weights)
            actor_critic=weights

            if hasattr(self.env, "curricula") and RunnerArgs_Student.resume_curriculum:
                # load curriculum state
                # distributions = loader.load_pkl("curriculum/distribution.pkl")
                # distribution_last = distributions[-1]["distribution"]
                f = open(RunnerArgs_Student.resume_path + '/curriculum/distribution.pkl', 'rb')
                distributions = pickle.load(f)
                # 也进行了修改，为了找到weights_
                distribution_last = distributions["distribution"]
                gait_names = [key[8:] if key.startswith("weights_") else None for key in distribution_last.keys()]
                for gait_id, gait_name in enumerate(self.env.category_names):
                    self.env.curricula[gait_id].weights = distribution_last[f"weights_{gait_name}"]
                    print(gait_name)

        self.alg = PPO(actor_critic, device=self.device)
        self.alg_teacher = PPO(teacher_policy, device=self.device)

        self.num_steps_per_env = RunnerArgs_Student.num_steps_per_env

        # init storage and model
        self.alg.init_storage(self.env.num_train_envs, self.num_steps_per_env, [self.env.num_obs],
                              [self.env.num_privileged_obs], [self.env.num_obs_history], [self.env.num_actions])

        self.tot_timesteps = 0
        self.tot_time = 0
        self.current_learning_iteration = 0
        self.last_recording_it = 0

        self.env.reset()



    def update_student(self):
        if self.alg.actor_critic.is_recurrent:
            generator = self.alg.storage.reccurent_mini_batch_generator(PPO_Args.num_mini_batches,
                                                                        PPO_Args.num_learning_epochs)
        else:
            generator = self.alg.storage.mini_batch_generator(PPO_Args.num_mini_batches, PPO_Args.num_learning_epochs)
        for sample in generator:
            if self.alg.actor_critic.is_recurrent:
                obs_batch, critic_obs_batch, privileged_obs_batch, obs_history_batch, actions_batch, values_batch, advantages_batch, returns_batch, \
                    hid_states_batch,masks_batch =sample
                student_act_batch, student_latent_batch = self.alg.actor_critic.get_student_act(obs_history_batch,
                                                                                                masks_batch,
                                                                                                hid_states_batch[0])
                teacher_act_batch, teacher_latent_batch = self.alg_teacher.actor_critic.get_teacher_act(
                    obs_history_batch, privileged_obs_batch, masks_batch)
            else:
                obs_batch, critic_obs_batch, privileged_obs_batch, obs_history_batch, actions_batch, target_values_batch, advantages_batch, returns_batch, old_actions_log_prob_batch, \
                        old_mu_batch, old_sigma_batch, env_bins_batch =sample
                student_act_batch, student_latent_batch = self.alg.actor_critic.get_student_act(obs_history_batch,None,None)
                teacher_act_batch, teacher_latent_batch = self.alg_teacher.actor_critic.get_teacher_act(obs_batch,privileged_obs_batch,None)

            act_loss = (teacher_act_batch - student_act_batch).pow(2).mean()
            latent_loss = (teacher_latent_batch - student_latent_batch).pow(2).mean()
            if AC_Args.use_help_decoder:
                act_loss = act_loss + (torch.cat((obs_batch,self.alg_teacher.actor_critic.get_encoded_privileged(obs_batch,privileged_obs_batch)),dim=-1) - self.alg.actor_critic.get_act_decoder_output(actions_batch)).pow(2).mean()
                latent_loss = latent_loss + (privileged_obs_batch - self.alg.actor_critic.get_help_decoder_output(student_latent_batch)).pow(2).mean()
                # act_loss = act_loss + (torch.cat((obs_batch,self.alg_teacher.actor_critic.get_encoded_privileged(privileged_obs_batch)),dim=-1) - self.alg.actor_critic.get_act_decoder_output(student_act_batch)).mean()
            loss = latent_loss + act_loss

            self.alg.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.alg.actor_critic.parameters(), PPO_Args.max_grad_norm)
            self.alg.optimizer.step()
            self.alg.storage.clear()

        num_updates = PPO_Args.num_learning_epochs * PPO_Args.num_mini_batches
        mean_act_loss,mean_adaptation_module_loss = act_loss/ num_updates,latent_loss/num_updates
        return mean_act_loss,mean_adaptation_module_loss


    def learn(self, num_learning_iterations, init_at_random_ep_len=False, eval_freq=100, curriculum_dump_freq=500, eval_expert=False):
        from ml_logger import logger
        # initialize writer
        assert logger.prefix, "you will overwrite the entire instrument server"

        logger.start('start', 'epoch', 'episode', 'run', 'step')

        if init_at_random_ep_len:
            self.env.episode_length_buf = torch.randint_like(self.env.episode_length_buf,
                                                             high=int(self.env.max_episode_length))


        obs_dict = self.env.get_observations()  # TODO: check, is this correct on the first step?
        obs, privileged_obs, obs_history = obs_dict["obs"], obs_dict["privileged_obs"], obs_dict["obs_history"]
        obs, privileged_obs, obs_history = obs.to(self.device), privileged_obs.to(self.device), obs_history.to(
            self.device)
        self.alg.actor_critic.train()  # switch to train mode (for dropout for example)

        rewbuffer = deque(maxlen=100)
        lenbuffer = deque(maxlen=100)
        rewbuffer_eval = deque(maxlen=100)
        lenbuffer_eval = deque(maxlen=100)
        cur_reward_sum = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)
        cur_episode_length = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)

        tot_iter = self.current_learning_iteration + num_learning_iterations
        for it in range(self.current_learning_iteration, tot_iter):
            start = time.time()
            # Rollout
            with torch.inference_mode():
                for i in range(self.num_steps_per_env):
                    actions_train = self.alg.act(obs, privileged_obs,obs_history)

                    # actions_train , _ = self.alg_teacher.actor_critic.get_teacher_act(obs_history,privileged_obs)
                    num_train_envs = self.env.num_train_envs

                    ret = self.env.step(actions_train)
                    obs_dict, rewards, dones, infos = ret
                    obs, privileged_obs, obs_history = obs_dict["obs"], obs_dict["privileged_obs"], obs_dict[
                        "obs_history"]

                    obs, privileged_obs, obs_history, rewards, dones = obs.to(self.device), privileged_obs.to(
                        self.device), obs_history.to(self.device), rewards.to(self.device), dones.to(self.device)
                    self.alg.process_env_step(rewards[:num_train_envs], dones[:num_train_envs], infos)

                    if 'train/episode' in infos:
                        with logger.Prefix(metrics="train/episode"):
                            logger.store_metrics(**infos['train/episode'])

                    if 'eval/episode' in infos:
                        with logger.Prefix(metrics="eval/episode"):
                            logger.store_metrics(**infos['eval/episode'])

                    if 'curriculum' in infos:

                        cur_reward_sum += rewards
                        cur_episode_length += 1

                        new_ids = (dones > 0).nonzero(as_tuple=False)

                        new_ids_train = new_ids[new_ids < num_train_envs]
                        rewbuffer.extend(cur_reward_sum[new_ids_train].cpu().numpy().tolist())
                        lenbuffer.extend(cur_episode_length[new_ids_train].cpu().numpy().tolist())
                        cur_reward_sum[new_ids_train] = 0
                        cur_episode_length[new_ids_train] = 0

                        new_ids_eval = new_ids[new_ids >= num_train_envs]
                        rewbuffer_eval.extend(cur_reward_sum[new_ids_eval].cpu().numpy().tolist())
                        lenbuffer_eval.extend(cur_episode_length[new_ids_eval].cpu().numpy().tolist())
                        cur_reward_sum[new_ids_eval] = 0
                        cur_episode_length[new_ids_eval] = 0

                    if 'curriculum/distribution' in infos:
                        distribution = infos['curriculum/distribution']

                stop = time.time()
                collection_time = stop - start

                # Learning step
                start = stop
                self.alg.compute_returns(obs_history[:num_train_envs], privileged_obs[:num_train_envs])

                if it % curriculum_dump_freq == 0:
                    logger.save_pkl({"iteration": it,
                                     **caches.slot_cache.get_summary(),
                                     **caches.dist_cache.get_summary()},
                                    path=f"curriculum/info.pkl", append=True)

                    if 'curriculum/distribution' in infos:
                        logger.save_pkl({"iteration": it,
                                         "distribution": distribution},
                                         path=f"curriculum/distribution.pkl", append=True)


            #update
            mean_act_loss,mean_adaptation_module_loss = self.update_student()
            self.alg.storage.clear()

            stop = time.time()
            learn_time = stop - start

            logger.store_metrics(
                # total_time=learn_time - collection_time,
                time_elapsed=logger.since('start'),
                time_iter=logger.split('epoch'),
                adaptation_loss=mean_adaptation_module_loss,
                act_loss=mean_act_loss,
                # critic_loss=mean_critic_loss
            )

            if RunnerArgs_Student.save_video_interval:
                self.log_video(it)

            self.tot_timesteps += self.num_steps_per_env * self.env.num_envs
            if logger.every(RunnerArgs_Student.log_freq, "iteration", start_on=1):
                # if it % Config.log_freq == 0:
                logger.log_metrics_summary(key_values={"timesteps": self.tot_timesteps, "iterations": it})
                logger.job_running()

            if it % RunnerArgs_Student.save_interval == 0:
                with logger.Sync():
                    logger.torch_save(self.alg.actor_critic.state_dict(), f"checkpoints/ac_weights_{it:06d}.pt")
                    logger.duplicate(f"checkpoints/ac_weights_{it:06d}.pt", f"checkpoints/ac_weights_last.pt")

                    path = './tmp/legged_data'

                    os.makedirs(path, exist_ok=True)

                    # if not self.alg.actor_critic.is_recurrent:
                    #     adaptation_module_path = f'{path}/adaptation_module_latest.jit'
                    #     adaptation_module = copy.deepcopy(self.alg.actor_critic.adaptation_module).to('cpu')
                    #     traced_script_adaptation_module = torch.jit.script(adaptation_module)
                    #     traced_script_adaptation_module.save(adaptation_module_path)
                    # else:
                    #     lstm_module_path = f'{path}/lstm_module_latest.jit'
                    #     lstm_module = copy.deepcopy(self.alg.actor_critic.memory_encoder).to('cpu')
                    #     traced_script_lstm_module = torch.jit.script(lstm_module)
                    #     traced_script_lstm_module.save(lstm_module_path)
                    #
                    #     encoder_module_path = f'{path}/encoder_module_latest.jit'
                    #     encoder_module = copy.deepcopy(self.alg.actor_critic.latent_head).to('cpu')
                    #     traced_script_encoder_module = torch.jit.script(encoder_module)
                    #     traced_script_encoder_module.save(encoder_module_path)
                    #
                    # body_path = f'{path}/body_latest.jit'
                    # body_model = copy.deepcopy(self.alg.actor_critic.actor_body).to('cpu')
                    # traced_script_body_module = torch.jit.script(body_model)
                    # traced_script_body_module.save(body_path)
                    #
                    # logger.upload_file(file_path=adaptation_module_path, target_path=f"checkpoints/")
                    # logger.upload_file(file_path=body_path, target_path=f"checkpoints/")

            self.current_learning_iteration += num_learning_iterations



    def log_video(self, it):
        if it - self.last_recording_it >= RunnerArgs_Student.save_video_interval:
            self.env.start_recording()
            if self.env.num_eval_envs > 0:
                self.env.start_recording_eval()
            print("START RECORDING")
            self.last_recording_it = it

        frames = self.env.get_complete_frames()
        if len(frames) > 0:
            self.env.pause_recording()
            print("LOGGING VIDEO")
            logger.save_video(frames, f"videos/{it:05d}.mp4", fps=1 / self.env.dt)

        if self.env.num_eval_envs > 0:
            frames = self.env.get_complete_frames_eval()
            if len(frames) > 0:
                self.env.pause_recording_eval()
                print("LOGGING EVAL VIDEO")
                logger.save_video(frames, f"videos/{it:05d}_eval.mp4", fps=1 / self.env.dt)

    def get_inference_policy(self, device=None):
        self.alg.actor_critic.eval()  # switch to evaluation mode (dropout for example)
        if device is not None:
            self.alg.actor_critic.to(device)
        return self.alg.actor_critic.act_inference

    def get_expert_policy(self, device=None):
        self.alg.actor_critic.eval()  # switch to evaluation mode (dropout for example)
        if device is not None:
            self.alg.actor_critic.to(device)
        return self.alg.actor_critic.act_expert
