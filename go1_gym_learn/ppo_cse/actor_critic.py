import torch
import torch.nn as nn
from params_proto import PrefixProto
from torch.distributions import Normal
from go1_gym.envs.base.legged_robot_config import Cfg

class AC_Args(PrefixProto, cli=False):
    # policy
    init_noise_std = 1.0
    actor_hidden_dims = [256, 128, 64]
    critic_hidden_dims = [512, 256, 128]
    adaptation_module_branch_hidden_dims = [256,128, 64]    #[64, 32]
    history_encoder_dims = [256, 128]
    activation = 'elu'  # can be elu, relu, selu, crelu, lrelu, tanh, sigmoid
    use_decoder = False
    num_commands = 18
    num_observation_history = Cfg.env.num_observation_history
    num_observations = Cfg.env.num_observations
    gait_encoder_hidden_dims = [64,32]
    gait_generator_hidden_dims = [128,64]
    history_encoder_output = 64


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
        actor_layers.append(nn.Linear(42+AC_Args.history_encoder_output+3+16+3, AC_Args.actor_hidden_dims[0]))
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
        critic_layers.append(nn.Linear(self.num_privileged_obs + 42+16, AC_Args.critic_hidden_dims[0]))
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
        historyencoder_layers.append(nn.Linear(self.num_obs_history, AC_Args.history_encoder_dims[0]))
        historyencoder_layers.append(activation)
        for l in range(len(AC_Args.history_encoder_dims)):
            if l == len(AC_Args.history_encoder_dims) - 1:
                historyencoder_layers.append(nn.Linear(AC_Args.history_encoder_dims[l], AC_Args.history_encoder_output))
            else:
                historyencoder_layers.append(nn.Linear(AC_Args.history_encoder_dims[l], AC_Args.history_encoder_dims[l + 1]))
                historyencoder_layers.append(activation)
        self.history_encoder = nn.Sequential(*historyencoder_layers)

        gait_encoder_layers = []
        gait_encoder_layers.append(nn.Linear(15, AC_Args.gait_encoder_hidden_dims[0]))
        gait_encoder_layers.append(activation)
        for l in range(len(AC_Args.gait_encoder_hidden_dims)):
            if l == len(AC_Args.gait_encoder_hidden_dims) - 1:
                gait_encoder_layers.append(nn.Linear(AC_Args.gait_encoder_hidden_dims[l], 16))
            else:
                gait_encoder_layers.append(nn.Linear(AC_Args.gait_encoder_hidden_dims[l], AC_Args.gait_encoder_hidden_dims[l + 1]))
                gait_encoder_layers.append(activation)
        self.gait_encoder = nn.Sequential(*gait_encoder_layers)

        gait_generator_layers = []
        gait_generator_layers.append(nn.Linear(42+3, AC_Args.gait_generator_hidden_dims[0]))
        gait_generator_layers.append(activation)
        for l in range(len(AC_Args.gait_generator_hidden_dims)):
            if l == len(AC_Args.gait_generator_hidden_dims) - 1:
                gait_generator_layers.append(nn.Linear(AC_Args.gait_generator_hidden_dims[l], 16))
            else:
                gait_generator_layers.append(nn.Linear(AC_Args.gait_generator_hidden_dims[l], AC_Args.gait_generator_hidden_dims[l + 1]))
                gait_generator_layers.append(activation)
        self.gait_generator = nn.Sequential(*gait_generator_layers)

        print(f"Adaptation Module: {self.adaptation_module}")
        print(f"Actor MLP: {self.actor_body}")
        print(f"Critic MLP: {self.critic_body}")

        # Action noise
        self.std = nn.Parameter(AC_Args.init_noise_std * torch.ones(num_actions))
        self.distribution = None
        # disable args validation for speedup
        Normal.set_default_validate_args = False

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

    def update_distribution(self, observation_history, privileged_obs,is_normal):
        ct = privileged_obs[:, -AC_Args.num_commands:-AC_Args.num_commands+3]
        ht = self.history_encoder(observation_history)
        vt = self.adaptation_module(observation_history)
        current_obs = observation_history[:,(AC_Args.num_observation_history-1)*AC_Args.num_observations:]
        if is_normal:
            gt = privileged_obs[:, -AC_Args.num_commands+3:]
            zt = self.gait_encoder(gt)
            zt = zt / torch.norm(zt, p=2, dim=-1, keepdim=True)
            mean = self.actor_body(torch.cat((current_obs,ht,vt,zt,ct),dim=-1)) #42+32+3+16+3
        else:
            zt = self.gait_generator(torch.cat((current_obs,ct),dim=-1))
            zt = zt / torch.norm(zt, p=2, dim=-1, keepdim=True)
            mean = self.actor_body(torch.cat((current_obs,ht,vt,zt,ct),dim=-1))
        self.distribution = Normal(mean, mean * 0. + self.std)

    def act(self, observation_history, privileged_obs, is_normal,**kwargs):
        self.update_distribution(observation_history,privileged_obs, is_normal)
        return self.distribution.sample()

    def get_actions_log_prob(self, actions):
        return self.distribution.log_prob(actions).sum(dim=-1)

    def act_expert(self, ob, policy_info={}):
        return self.act_teacher(ob["obs_history"], ob["privileged_obs"])

    def act_inference(self, ob, policy_info={}):
        return self.act_student(ob["obs_history"], policy_info=policy_info)

    def act_student(self, observation_history, policy_info={}):
        latent = self.adaptation_module(observation_history)
        actions_mean = self.actor_body(torch.cat((observation_history, latent), dim=-1))
        policy_info["latents"] = latent.detach().cpu().numpy()
        return actions_mean

    def act_teacher(self, observation_history, privileged_info, policy_info={}):
        actions_mean = self.actor_body(torch.cat((observation_history, privileged_info), dim=-1))
        policy_info["latents"] = privileged_info
        return actions_mean

    def evaluate(self, observation_history, privileged_observations, is_normal,**kwargs):
        ct = privileged_observations[:, -AC_Args.num_commands:-AC_Args.num_commands+3]
        if is_normal:
            gt = gt = privileged_observations[:, -AC_Args.num_commands+3:]
            zt = self.gait_encoder(gt)
            zt = zt / torch.norm(zt, p=2, dim=-1, keepdim=True)
        else:
            zt = self.gait_generator(torch.cat((observation_history,ct),dim=-1))
            zt = zt / torch.norm(zt, p=2, dim=-1, keepdim=True)
        value = self.critic_body(torch.cat((observation_history, privileged_observations,zt), dim=-1))
        return value

    def get_student_latent(self, observation_history):
        return self.adaptation_module(observation_history)

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
