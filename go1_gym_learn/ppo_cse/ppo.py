import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from params_proto import PrefixProto

from go1_gym_learn.ppo_cse import ActorCritic
from go1_gym_learn.ppo_cse import RolloutStorage
from go1_gym_learn.ppo_cse import caches
from torch.distributions import Normal


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

    def __init__(self, actor_critic, device='cpu'):

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
        self.transition = RolloutStorage.Transition()

        self.learning_rate = PPO_Args.learning_rate

    def init_storage(self, num_envs, num_transitions_per_env, actor_obs_shape, privileged_obs_shape, obs_history_shape,
                     action_shape):
        self.storage = RolloutStorage(num_envs, num_transitions_per_env, actor_obs_shape, privileged_obs_shape,
                                      obs_history_shape, action_shape, self.device)

    def test_mode(self):
        self.actor_critic.test()

    def train_mode(self):
        self.actor_critic.train()

    def act(self, obs, privileged_obs, obs_history,is_normal):
        # Compute the actions and values
        if is_normal:
            self.transition.actions = self.actor_critic.act(obs, privileged_obs, obs_history, True).detach()
            self.transition.values = self.actor_critic.evaluate(obs, privileged_obs, True).detach()
            self.transition.actions_log_prob = self.actor_critic.get_actions_log_prob(self.transition.actions).detach()
            self.transition.action_mean = self.actor_critic.action_mean.detach()
            self.transition.action_sigma = self.actor_critic.action_std.detach()
            self.transition.observations = obs
            self.transition.critic_observations = obs
            self.transition.privileged_observations = privileged_obs
            self.transition.observation_histories = obs_history
            self.transition.is_normal = torch.ones(2048,device=self.device)
        else:
            actions_temp = self.actor_critic.act(obs, privileged_obs, obs_history, False).detach()
            self.transition.values = self.actor_critic.evaluate(obs, privileged_obs, False).detach()
            self.transition.actions = torch.cat((self.transition.actions,actions_temp),dim=0)
            self.transition.values = torch.cat((self.transition.values,self.actor_critic.evaluate(obs, privileged_obs,False).detach()),dim=0)
            self.transition.actions_log_prob = torch.cat((self.transition.actions_log_prob, self.actor_critic.get_actions_log_prob(actions_temp).detach()),dim=0)
            self.transition.action_mean = torch.cat((self.transition.action_mean, self.actor_critic.action_mean.detach()),dim=0)
            self.transition.action_sigma = torch.cat( (self.transition.action_sigma,self.actor_critic.action_std.detach()),dim=0)
            self.transition.observations = torch.cat((self.transition.observations, obs),dim=0)
            self.transition.critic_observations = torch.cat((self.transition.critic_observations, obs),dim=0)
            self.transition.privileged_observations = torch.cat((self.transition.privileged_observations, privileged_obs),dim=0)
            self.transition.observation_histories = torch.cat((self.transition.observation_histories, obs_history),dim=0)
            self.transition.is_normal = torch.cat((self.transition.is_normal,torch.zeros(2048,device=self.device)),dim=-1)
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
        last_values_normal = self.actor_critic.evaluate(last_critic_obs[:2048], last_critic_privileged_obs[:2048],True).detach()
        last_value_adaptative = self.actor_critic.evaluate(last_critic_obs[2048:], last_critic_privileged_obs[2048:],False).detach()
        self.storage.compute_returns(torch.cat((last_values_normal,last_value_adaptative),dim=0), PPO_Args.gamma, PPO_Args.lam)

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

            is_normal_true_indices = [i for i, val in enumerate(is_normal_batch) if val]
            is_normal_false_indices = [i for i, val in enumerate(is_normal_batch) if not val]
            obs_true = obs_batch[is_normal_true_indices]
            privileged_obs_true = privileged_obs_batch[is_normal_true_indices]
            obs_history_true = obs_history_batch[is_normal_true_indices]
            obs_false = obs_batch[is_normal_false_indices]
            privileged_obs_false = privileged_obs_batch[is_normal_false_indices]
            obs_history_false = obs_history_batch[is_normal_false_indices]
            value_batch = torch.zeros((len(is_normal_batch), 1), device=is_normal_batch.device)
            action_temp = torch.zeros((len(is_normal_batch), 12), device=is_normal_batch.device)
            if len(is_normal_true_indices) > 0:
                action_true = self.actor_critic.act_update(obs_true, privileged_obs_true, obs_history_true, True)
                action_temp[is_normal_true_indices] = action_true
                value_true = self.actor_critic.evaluate(obs_true, privileged_obs_true,True)
                value_batch[is_normal_true_indices] = value_true
            if len(is_normal_false_indices) > 0:
                action_false = self.actor_critic.act_update(obs_false, privileged_obs_false, obs_history_false, False)
                action_temp[is_normal_false_indices] = action_false
                value_false = self.actor_critic.evaluate(obs_false, privileged_obs_false,False)
                value_batch[is_normal_false_indices] = value_false
            self.actor_critic.distribution = Normal(action_temp, action_temp * 0. + self.actor_critic.std)
            actions_log_prob_batch = self.actor_critic.get_actions_log_prob(actions_batch)
            mu_batch = self.actor_critic.action_mean
            sigma_batch = self.actor_critic.action_std
            entropy_batch = self.actor_critic.entropy

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

            loss = surrogate_loss + PPO_Args.value_loss_coef * value_loss - PPO_Args.entropy_coef * entropy_batch.mean()

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
                    adaptation_target = privileged_obs_batch[:,:3]
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
