import torch
import torch.nn as nn
from params_proto import PrefixProto
from torch.distributions import Normal
import torch.optim as optim


class AC_Args(PrefixProto, cli=False):
    # policy
    init_noise_std = 1.0
    actor_hidden_dims = [256, 128, 64]
    critic_hidden_dims = [512, 256, 128]
    adaptation_module_branch_hidden_dims = [256,128, 64]    #[64, 32]
    history_encoder_dims = [256, 128, 64]
    activation = 'elu'  # can be elu, relu, selu, crelu, lrelu, tanh, sigmoid


    use_decoder = False


class ActorCritic(nn.Module):
    is_recurrent = False

    def __init__(self, num_obs,
                 num_privileged_obs,
                 num_obs_history,
                 num_actions,
                 **kwargs):
        if kwargs:
            print("ActorCritic.__init__ got unexpected arguments, which will be ignored: " + str(
                [key for key in kwargs.keys()]))
        self.decoder = AC_Args.use_decoder
        super().__init__()

        self.num_obs_history = num_obs_history
        self.num_privileged_obs = num_privileged_obs

        activation = get_activation(AC_Args.activation)

        # Adaptation module
        adaptation_module_layers = []
        adaptation_module_layers.append(nn.Linear(self.num_obs_history, AC_Args.adaptation_module_branch_hidden_dims[0]))
        adaptation_module_layers.append(activation)
        for l in range(len(AC_Args.adaptation_module_branch_hidden_dims)):
            if l == len(AC_Args.adaptation_module_branch_hidden_dims) - 1:
                adaptation_module_layers.append(
                    nn.Linear(AC_Args.adaptation_module_branch_hidden_dims[l], 3))
            else:
                adaptation_module_layers.append(
                    nn.Linear(AC_Args.adaptation_module_branch_hidden_dims[l],
                              AC_Args.adaptation_module_branch_hidden_dims[l + 1]))
                adaptation_module_layers.append(activation)
        self.adaptation_module = nn.Sequential(*adaptation_module_layers)



        # Policy
        actor_layers = []
        actor_layers.append(nn.Linear(32+13+3, AC_Args.actor_hidden_dims[0]))
        actor_layers.append(activation)
        for l in range(len(AC_Args.actor_hidden_dims)):
            if l == len(AC_Args.actor_hidden_dims) - 1:
                actor_layers.append(nn.Linear(AC_Args.actor_hidden_dims[l], num_actions))
            else:
                actor_layers.append(nn.Linear(AC_Args.actor_hidden_dims[l], AC_Args.actor_hidden_dims[l + 1]))
                actor_layers.append(activation)
        self.actor_body = nn.Sequential(*actor_layers)

        # Value function
        critic_layers = []
        critic_layers.append(nn.Linear(13+42+24+187, AC_Args.critic_hidden_dims[0]))
        critic_layers.append(activation)
        for l in range(len(AC_Args.critic_hidden_dims)):
            if l == len(AC_Args.critic_hidden_dims) - 1:
                critic_layers.append(nn.Linear(AC_Args.critic_hidden_dims[l], 1))
            else:
                critic_layers.append(nn.Linear(AC_Args.critic_hidden_dims[l], AC_Args.critic_hidden_dims[l + 1]))
                critic_layers.append(activation)
        self.critic_body = nn.Sequential(*critic_layers)

        # history encoder
        historyencoder_layers = []
        historyencoder_layers.append(nn.Linear(42*5, AC_Args.history_encoder_dims[0]))
        historyencoder_layers.append(activation)
        for l in range(len(AC_Args.history_encoder_dims)):
            if l == len(AC_Args.history_encoder_dims) - 1:
                historyencoder_layers.append(nn.Linear(AC_Args.history_encoder_dims[l], 32))
            else:
                historyencoder_layers.append(nn.Linear(AC_Args.history_encoder_dims[l], AC_Args.history_encoder_dims[l + 1]))
                historyencoder_layers.append(activation)
        self.history_encoder = nn.Sequential(*historyencoder_layers)

        # forward_model_layers = []
        # forward_model_layers.append(nn.Linear(210, 256))
        # # forward_model_layers.append(nn.Linear(42+13,256))   # state+action->next state(dof vel,vel pos,base ang,projected gravity)
        # forward_model_layers.append(activation)
        # forward_model_layers.append(nn.Linear(256,128))
        # forward_model_layers.append(activation)
        # forward_model_layers.append(nn.Linear(128,64))
        # forward_model_layers.append(activation)
        # forward_model_layers.append(nn.Linear(64,30))
        # self.forward_model = nn.Sequential(*forward_model_layers)
        #
        # inverse_model_layers = []
        # inverse_model_layers.append(nn.Linear(210, 256))
        # # inverse_model_layers.append(nn.Linear(43+43,256))   # state+state(obs-action+command)->action
        # inverse_model_layers.append(activation)
        # inverse_model_layers.append(nn.Linear(256,128))
        # inverse_model_layers.append(activation)
        # inverse_model_layers.append(nn.Linear(128,64))
        # inverse_model_layers.append(activation)
        # inverse_model_layers.append(nn.Linear(64,12))
        # self.inverse_model = nn.Sequential(*inverse_model_layers)

        print(f"Adaptation Module: {self.adaptation_module}")
        print(f"Actor MLP: {self.actor_body}")
        print(f"Critic MLP: {self.critic_body}")

        # Action noise
        self.std = nn.Parameter(AC_Args.init_noise_std * torch.ones(num_actions))
        self.distribution = None
        # disable args validation for speedup
        Normal.set_default_validate_args = False

        # self.forward_module_optimizer = optim.Adam(self.forward_model.parameters(),
        #                                               lr=1.e-3)
        #
        # self.inverse_module_optimizer = optim.Adam(self.inverse_model.parameters(),
        #                                               lr=1.e-3)

    @staticmethod
    # not used at the moment
    def init_weights(sequential, scales):
        [torch.nn.init.orthogonal_(module.weight, gain=scales[idx]) for idx, module in
         enumerate(mod for mod in sequential if isinstance(mod, nn.Linear))]

    def reset(self, dones=None):
        pass

    def forward(self):
        raise NotImplementedError

    @property
    def action_mean(self):
        return self.distribution.mean

    @property
    def action_std(self):
        return self.distribution.stddev

    @property
    def entropy(self):
        return self.distribution.entropy().sum(dim=-1)


    # def compute_intrinsic_reward(self,obs,pri_obs,last_obs,last_pri_obs,next_action,next_obs):
    #     obs = obs.requires_grad_(True)
    #     pri_obs = pri_obs.requires_grad_(True)
    #     last_obs = last_obs.requires_grad_(True)
    #     last_pri_obs = last_pri_obs.requires_grad_(True)
    #     next_action = next_action.requires_grad_(True)
    #     next_obs = next_obs.requires_grad_(True)
    #
    #     for param in self.forward_model.parameters():
    #         param.requires_grad = True
    #     for param in self.inverse_model.parameters():
    #         param.requires_grad = True
    #     obs = obs.requires_grad_(True)
    #     pri_obs = pri_obs.requires_grad_(True)
    #     last_obs = last_obs.requires_grad_(True)
    #     last_pri_obs = last_pri_obs.requires_grad_(True)
    #     next_action = next_action.requires_grad_(True)
    #     next_obs = next_obs.requires_grad_(True)
    #
    #     for param in self.forward_model.parameters():
    #         param.requires_grad = True
    #     for param in self.inverse_model.parameters():
    #         param.requires_grad = True
    #
    #     # with torch.enable_grad():
    #     command = pri_obs[:, -13:].requires_grad_(True)
    #     current_state = obs[:, :30].requires_grad_(True)
    #     predict_state = self.forward_model(torch.cat((current_state,command,next_action),dim=-1)).requires_grad_(True)
    #     real_state = next_obs[:, :30].requires_grad_(True)
    #     forward_reward = torch.sum(torch.square(real_state - predict_state),dim=-1).requires_grad_(True)
    #
    #     last_command = last_pri_obs[:, -13:].requires_grad_(True)
    #     last_state = last_obs[:, :30].requires_grad_(True)
    #     predict_action = self.inverse_model(torch.cat((last_state,last_command,current_state,command),dim=-1)).requires_grad_(True)
    #     inverse_reward = torch.sum(torch.square(next_action - predict_action),dim=-1).requires_grad_(True)
    #
    #     forward_loss = forward_reward.mean().requires_grad_(True)
    #     inverse_loss = inverse_reward.mean().requires_grad_(True)
    #
    #     self.forward_module_optimizer.zero_grad()
    #     forward_loss.backward()
    #     self.forward_module_optimizer.step()
    #     self.inverse_module_optimizer.zero_grad()
    #     inverse_loss.backward()
    #     self.inverse_module_optimizer.step()
    #     return forward_reward,inverse_reward,forward_loss,inverse_loss

    def compute_intrinsic_reward(self,obs_his,next_action,next_obs):
        forward_reward = torch.sum(torch.square(next_obs[:,:30]-self.forward_model(obs_his)),dim=-1).requires_grad_(True)
        # inverse_reward = torch.sum(torch.square(next_action-self.inverse_model(obs_his)),dim=-1).requires_grad_(True)

        forward_loss = forward_reward.mean().requires_grad_(True)
        # inverse_loss = inverse_reward.mean().requires_grad_(True)

        self.forward_module_optimizer.zero_grad()
        forward_loss.backward()
        self.forward_module_optimizer.step()

        # self.inverse_module_optimizer.zero_grad()
        # inverse_loss.backward()
        # self.inverse_module_optimizer.step()
        inverse_reward=0
        inverse_loss=0
        return forward_reward,inverse_reward,forward_loss,inverse_loss

    def update_distribution(self, observation_history, command):
        ht = self.history_encoder(observation_history)
        vt = self.adaptation_module(observation_history)
        mean = self.actor_body(torch.cat((ht, vt, command), dim=-1))
        self.distribution = Normal(mean, mean * 0. + self.std)

    def act(self, observation_history, privileged_obs, **kwargs):
        command = privileged_obs[:, -13:]
        self.update_distribution(observation_history,command)
        return self.distribution.sample()

    def act_play(self, observation_history, privileged_obs):
        command = privileged_obs[:, -13:]
        ht = self.history_encoder(observation_history)
        vt = self.adaptation_module(observation_history)
        mean = self.actor_body(torch.cat((ht, vt, command), dim=-1))

    def get_actions_log_prob(self, actions):
        return self.distribution.log_prob(actions).sum(dim=-1)

    def act_expert(self, ob, policy_info={}):
        return self.act_teacher(ob["obs_history"], ob["privileged_obs"])

    def act_inference(self, ob, policy_info={}):
        return self.act_student(ob["obs_history"], policy_info=policy_info)

    def evaluate(self, observation, privileged_observations, **kwargs):
        value = self.critic_body(torch.cat((observation, privileged_observations), dim=-1))
        return value

    # def get_student_latent(self, observation_history):
    #     return self.adaptation_module(observation_history)

def get_activation(act_name):
    if act_name == "elu":
        return nn.ELU()
    elif act_name == "selu":
        return nn.SELU()
    elif act_name == "relu":
        return nn.ReLU()
    elif act_name == "crelu":
        return nn.ReLU()
    elif act_name == "lrelu":
        return nn.LeakyReLU()
    elif act_name == "tanh":
        return nn.Tanh()
    elif act_name == "sigmoid":
        return nn.Sigmoid()
    else:
        print("invalid activation function!")
        return None
