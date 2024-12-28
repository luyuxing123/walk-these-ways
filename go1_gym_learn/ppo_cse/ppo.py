import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from params_proto import PrefixProto

from go1_gym_learn.ppo_cse import ActorCritic
from go1_gym_learn.ppo_cse import RolloutStorage
from go1_gym_learn.ppo_cse import caches


class PPO_Args(PrefixProto):
    # algorithm
    value_loss_coef = 1.0
    use_clipped_value_loss = True
    clip_param = 0.2
    entropy_coef = 0.01
    num_learning_epochs = 5
    num_mini_batches = 4  # mini batch size = num_envs*nsteps / nminibatches
    learning_rate = 1.e-3  # 5.e-4
    adaptation_module_learning_rate = 1.e-3
    num_adaptation_module_substeps = 1
    schedule = 'adaptive'  # could be adaptive, fixed
    gamma = 0.99
    lam = 0.95
    desired_kl = 0.01
    max_grad_norm = 1.

    selective_adaptation_module_loss = False


class PPO:
    actor_critic: ActorCritic

    def __init__(self, actor_critic,discriminator,amp_normalizer, device='cpu'):

        self.device = device

        # PPO components
        self.actor_critic = actor_critic
        self.actor_critic.to(device)
        self.storage = None  # initialized later
        self.optimizer = optim.Adam(self.actor_critic.parameters(), lr=PPO_Args.learning_rate)
        self.adaptation_module_optimizer = optim.Adam(self.actor_critic.parameters(),
                                                      lr=PPO_Args.adaptation_module_learning_rate)
        if self.actor_critic.decoder:
            self.decoder_optimizer = optim.Adam(self.actor_critic.parameters(),
                                                          lr=PPO_Args.adaptation_module_learning_rate)
        self.discriminator = discriminator
        self.discriminator.to(self.device)

        self.transition = RolloutStorage.Transition()

        self.learning_rate = PPO_Args.learning_rate
        self.amp_normalizer = amp_normalizer

    def init_storage(self, num_envs, num_transitions_per_env, actor_obs_shape, privileged_obs_shape, obs_history_shape,
                     action_shape):
        self.storage = RolloutStorage(num_envs, num_transitions_per_env, actor_obs_shape, privileged_obs_shape,
                                      obs_history_shape, action_shape, self.device)

    def test_mode(self):
        self.actor_critic.test()

    def train_mode(self):
        self.actor_critic.train()

    def act(self, obs, privileged_obs, obs_history):
        # Compute the actions and values
        num_envs = obs.shape[0]
        action_normal = self.actor_critic.act(obs_history[:num_envs//2],privileged_obs[:num_envs//2],is_normal=True).detach()
        action_mean_normal = self.actor_critic.action_mean.detach()
        action_sigma_normal = self.actor_critic.action_std.detach()
        action_log_prob_normal = self.actor_critic.get_actions_log_prob(action_normal).detach()
        value_normal = self.actor_critic.evaluate(obs[:num_envs//2],privileged_obs[:num_envs//2],is_normal=True).detach()

        action_random = self.actor_critic.act(obs_history[num_envs//2:],privileged_obs[num_envs//2:],is_normal=False).detach()
        action_mean_random = self.actor_critic.action_mean.detach()
        action_sigma_random = self.actor_critic.action_std.detach()
        action_log_prob_random = self.actor_critic.get_actions_log_prob(action_random).detach()
        value_random = self.actor_critic.evaluate(obs[num_envs//2:],privileged_obs[num_envs//2:],is_normal=False).detach()

        self.transition.actions = torch.cat((action_normal,action_random),dim=0)
        self.transition.action_mean = torch.cat((action_mean_normal,action_mean_random),dim=0)
        self.transition.action_sigma = torch.cat((action_sigma_normal,action_sigma_random),dim=0)
        self.transition.actions_log_prob = torch.cat((action_log_prob_normal,action_log_prob_random),dim=0)
        self.transition.values = torch.cat((value_normal,value_random),dim=0)
        # need to record obs and critic_obs before env.step()
        self.transition.observations = obs
        self.transition.critic_observations = obs
        self.transition.privileged_observations = privileged_obs
        self.transition.observation_histories = obs_history
        self.transition.is_normal = torch.cat((torch.ones(num_envs//2,dtype=torch.bool),
                                               torch.zeros(num_envs//2,dtype=torch.bool)),dim=-1)
        return self.transition.actions

    def process_env_step(self, rewards, dones, infos):
        self.transition.rewards = rewards.clone()
        self.transition.dones = dones
        self.transition.env_bins = infos["env_bins"]
        # Bootstrapping on time outs
        if 'time_outs' in infos:
            self.transition.rewards += PPO_Args.gamma * torch.squeeze(
                self.transition.values * infos['time_outs'].unsqueeze(1).to(self.device), 1)

        # Record the transition
        self.storage.add_transitions(self.transition)
        self.transition.clear()
        self.actor_critic.reset(dones)

    def compute_returns(self, last_critic_obs, last_critic_privileged_obs):
        num_envs = last_critic_obs.shape[0]
        last_values_normal = self.actor_critic.evaluate(last_critic_obs[:num_envs//2], last_critic_privileged_obs[:num_envs//2],is_normal=True).detach()
        last_values_random = self.actor_critic.evaluate(last_critic_obs[num_envs//2:], last_critic_privileged_obs[num_envs//2:],is_normal=False).detach()
        last_values = torch.cat((last_values_normal,last_values_random),dim=0)
        self.storage.compute_returns(last_values, PPO_Args.gamma, PPO_Args.lam)

    def update(self):
        mean_value_loss = 0
        mean_surrogate_loss = 0
        mean_adaptation_module_loss = 0
        mean_decoder_loss = 0
        mean_decoder_loss_student = 0
        mean_adaptation_module_test_loss = 0
        mean_decoder_test_loss = 0
        mean_decoder_test_loss_student = 0
        generator = self.storage.mini_batch_generator(PPO_Args.num_mini_batches, PPO_Args.num_learning_epochs)
        for obs_batch, critic_obs_batch, privileged_obs_batch, obs_history_batch, actions_batch, target_values_batch, advantages_batch, returns_batch, old_actions_log_prob_batch, \
            old_mu_batch, old_sigma_batch, masks_batch, env_bins_batch, is_normal_batch in generator:

            normal_indices = is_normal_batch.nonzero(as_tuple=True)[0]
            random_indices = (is_normal_batch == 0).nonzero(as_tuple=True)[0]

            actions_log_prob_batch = torch.zeros_like(
                actions_batch[:, 0])  # 假设 actions_batch 的 shape 是 [batch_size, action_dim]
            mu_batch = torch.zeros_like(actions_batch)  # 假设 mu 的 shape 与 actions_batch 一致
            sigma_batch = torch.zeros_like(actions_batch)  # 假设 sigma 的 shape 与 actions_batch 一致
            entropy_batch = torch.zeros_like(actions_batch[:, 0])  # 假设 entropy 的 shape 为 [batch_size]
            value_batch = torch.zeros_like(actions_batch[:, 0]).unsqueeze(-1)

            # 处理 normal_indices
            if len(normal_indices) > 0:
                self.actor_critic.act(
                    obs_history_batch[normal_indices],
                    privileged_obs_batch[normal_indices],
                    is_normal=True)

                actions_log_prob_batch[normal_indices] = self.actor_critic.get_actions_log_prob(
                    actions_batch[normal_indices])
                mu_batch[normal_indices] = self.actor_critic.action_mean
                sigma_batch[normal_indices] = self.actor_critic.action_std
                entropy_batch[normal_indices] = self.actor_critic.entropy
                value_batch[normal_indices] = self.actor_critic.evaluate(obs_batch[normal_indices],
                                                                         privileged_obs_batch[normal_indices],
                                                                         is_normal=True)

            # 处理 random_indices
            if len(random_indices) > 0:
                self.actor_critic.act(
                    obs_history_batch[random_indices],
                    privileged_obs_batch[random_indices],
                    is_normal=False)

                actions_log_prob_batch[random_indices] = self.actor_critic.get_actions_log_prob(
                    actions_batch[random_indices])
                mu_batch[random_indices] = self.actor_critic.action_mean
                sigma_batch[random_indices] = self.actor_critic.action_std
                entropy_batch[random_indices] = self.actor_critic.entropy
                value_batch[random_indices] = self.actor_critic.evaluate(obs_batch[random_indices],
                                                                         privileged_obs_batch[random_indices],
                                                                         is_normal=False)
            # KL
            if PPO_Args.desired_kl != None and PPO_Args.schedule == 'adaptive':
                with torch.inference_mode():
                    kl = torch.sum(
                        torch.log(sigma_batch / old_sigma_batch + 1.e-5) + (
                                torch.square(old_sigma_batch) + torch.square(old_mu_batch - mu_batch)) / (
                                2.0 * torch.square(sigma_batch)) - 0.5, axis=-1)
                    kl_mean = torch.mean(kl)

                    if kl_mean > PPO_Args.desired_kl * 2.0:
                        self.learning_rate = max(1e-5, self.learning_rate / 1.5)
                    elif kl_mean < PPO_Args.desired_kl / 2.0 and kl_mean > 0.0:
                        self.learning_rate = min(1e-2, self.learning_rate * 1.5)

                    for param_group in self.optimizer.param_groups:
                        param_group['lr'] = self.learning_rate

            # Surrogate loss
            ratio = torch.exp(actions_log_prob_batch - torch.squeeze(old_actions_log_prob_batch))
            surrogate = -torch.squeeze(advantages_batch) * ratio
            surrogate_clipped = -torch.squeeze(advantages_batch) * torch.clamp(ratio, 1.0 - PPO_Args.clip_param,
                                                                               1.0 + PPO_Args.clip_param)
            surrogate_loss = torch.max(surrogate, surrogate_clipped).mean()

            # Value function loss
            if PPO_Args.use_clipped_value_loss:
                value_clipped = target_values_batch + \
                                (value_batch - target_values_batch).clamp(-PPO_Args.clip_param,
                                                                          PPO_Args.clip_param)
                value_losses = (value_batch - returns_batch).pow(2)
                value_losses_clipped = (value_clipped - returns_batch).pow(2)
                value_loss = torch.max(value_losses, value_losses_clipped).mean()
            else:
                value_loss = (returns_batch - value_batch).pow(2).mean()


            norm_obs = self.discriminator.norm_obs(torch.cat((obs_batch[normal_indices],obs_batch[random_indices]),dim=0))
            d_score = self.discriminator(norm_obs)
            normal_d = d_score[:norm_obs.shape[0]//2]
            random_d = d_score[norm_obs.shape[0]//2:]
            normal_loss = torch.nn.MSELoss()(
                normal_d, torch.ones(normal_d.size(), device=self.device))
            random_loss = torch.nn.MSELoss()(
                random_d, -1 * torch.ones(random_d.size(), device=self.device))
            amp_loss = 0.5 * (normal_loss + random_loss)
            grad_pen_loss = self.discriminator.compute_grad_pen(
                obs_batch[normal_indices,:30],obs_batch[normal_indices,30:], lambda_=10)


            loss = (surrogate_loss +
                    PPO_Args.value_loss_coef * value_loss -
                    PPO_Args.entropy_coef * entropy_batch.mean()
                    + amp_loss + grad_pen_loss)

            # Gradient step
            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.actor_critic.parameters(), PPO_Args.max_grad_norm)
            self.optimizer.step()

            mean_value_loss += value_loss.item()
            mean_surrogate_loss += surrogate_loss.item()

            data_size = privileged_obs_batch.shape[0]
            num_train = int(data_size // 5 * 4)

            # Adaptation module gradient step

            for epoch in range(PPO_Args.num_adaptation_module_substeps):

                adaptation_pred = self.actor_critic.adaptation_module(obs_history_batch)
                with torch.no_grad():
                    adaptation_target = privileged_obs_batch[:,-18:-15]
                    # residual = (adaptation_target - adaptation_pred).norm(dim=1)
                    # caches.slot_cache.log(env_bins_batch[:, 0].cpu().numpy().astype(np.uint8),
                    #                       sysid_residual=residual.cpu().numpy())

                selection_indices = torch.linspace(0, adaptation_pred.shape[1]-1, steps=adaptation_pred.shape[1], dtype=torch.long)
                if PPO_Args.selective_adaptation_module_loss:
                    # mask out indices corresponding to swing feet
                    selection_indices = 0

                adaptation_loss = F.mse_loss(adaptation_pred[:num_train, selection_indices], adaptation_target[:num_train, selection_indices])
                adaptation_test_loss = F.mse_loss(adaptation_pred[num_train:, selection_indices], adaptation_target[num_train:, selection_indices])



                self.adaptation_module_optimizer.zero_grad()
                adaptation_loss.backward()
                self.adaptation_module_optimizer.step()

                mean_adaptation_module_loss += adaptation_loss.item()
                mean_adaptation_module_test_loss += adaptation_test_loss.item()

        num_updates = PPO_Args.num_learning_epochs * PPO_Args.num_mini_batches
        mean_value_loss /= num_updates
        mean_surrogate_loss /= num_updates
        mean_adaptation_module_loss /= (num_updates * PPO_Args.num_adaptation_module_substeps)
        mean_decoder_loss /= (num_updates * PPO_Args.num_adaptation_module_substeps)
        mean_decoder_loss_student /= (num_updates * PPO_Args.num_adaptation_module_substeps)
        mean_adaptation_module_test_loss /= (num_updates * PPO_Args.num_adaptation_module_substeps)
        mean_decoder_test_loss /= (num_updates * PPO_Args.num_adaptation_module_substeps)
        mean_decoder_test_loss_student /= (num_updates * PPO_Args.num_adaptation_module_substeps)
        self.storage.clear()

        return mean_value_loss, mean_surrogate_loss, mean_adaptation_module_loss, mean_decoder_loss, mean_decoder_loss_student, mean_adaptation_module_test_loss, mean_decoder_test_loss, mean_decoder_test_loss_student
