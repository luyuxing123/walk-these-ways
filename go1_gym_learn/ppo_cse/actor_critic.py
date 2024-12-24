import torch
import torch.nn as nn
from params_proto import PrefixProto
from torch.distributions import Normal


class AC_Args(PrefixProto, cli=False):
    # policy
    init_noise_std = 1.0
    actor_hidden_dims = [256, 128, 64]
    critic_hidden_dims = [512, 256, 128]
    activation = 'elu'  # can be elu, relu, selu, crelu, lrelu, tanh, sigmoid

    use_decoder = False

    history_encoder_dim = [256, 128]
    history_encoder_output_dim = 32
    adaptation_module_branch_hidden_dims = [64, 32]
    adaptation_module_output_dim = 3
    gait_encoder_dim = [64, 32]
    gait_generator_dim = [128, 64]

    actor_body_input_dim = 3+32+3+16   #ct,ht,vt,z
    critic_body_input_dim = 16+42+187+3+24   #zt,privileged obs，terrain，command,obs



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

        # history_encoder
        history_encoder_layers = []
        history_encoder_layers.append(nn.Linear(self.num_obs_history, AC_Args.history_encoder_dim[0]))
        history_encoder_layers.append(activation)
        for l in range(len(AC_Args.history_encoder_dim)):
            if l == len(AC_Args.history_encoder_dim) - 1:
                history_encoder_layers.append(
                    nn.Linear(AC_Args.history_encoder_dim[l], AC_Args.history_encoder_output_dim))
            else:
                history_encoder_layers.append(
                    nn.Linear(AC_Args.history_encoder_dim[l],
                              AC_Args.history_encoder_dim[l + 1]))
                history_encoder_layers.append(activation)
        self.history_encoder = nn.Sequential(*history_encoder_layers)

        # Adaptation module estimate vel
        adaptation_module_layers = []
        adaptation_module_layers.append(nn.Linear(self.num_obs_history, AC_Args.adaptation_module_branch_hidden_dims[0]))
        adaptation_module_layers.append(activation)
        for l in range(len(AC_Args.adaptation_module_branch_hidden_dims)):
            if l == len(AC_Args.adaptation_module_branch_hidden_dims) - 1:
                adaptation_module_layers.append(
                    nn.Linear(AC_Args.adaptation_module_branch_hidden_dims[l], AC_Args.adaptation_module_output_dim))
            else:
                adaptation_module_layers.append(
                    nn.Linear(AC_Args.adaptation_module_branch_hidden_dims[l],
                              AC_Args.adaptation_module_branch_hidden_dims[l + 1]))
                adaptation_module_layers.append(activation)
        self.adaptation_module = nn.Sequential(*adaptation_module_layers)

        # gait encoder
        gait_encoder_layers = []
        gait_encoder_layers.append(nn.Linear(8, AC_Args.gait_encoder_dim[0]))
        gait_encoder_layers.append(activation)
        for l in range(len(AC_Args.gait_encoder_dim)):
            if l == len(AC_Args.gait_encoder_dim) - 1:
                gait_encoder_layers.append(
                    nn.Linear(AC_Args.gait_encoder_dim[l], 16))
            else:
                gait_encoder_layers.append(
                    nn.Linear(AC_Args.gait_encoder_dim[l],
                              AC_Args.gait_encoder_dim[l + 1]))
                gait_encoder_layers.append(activation)
        self.gait_encoder = nn.Sequential(*gait_encoder_layers)

        # gait generator
        gait_generator_layers = []
        gait_generator_layers.append(nn.Linear(45, AC_Args.gait_generator_dim[0]))
        gait_generator_layers.append(activation)
        for l in range(len(AC_Args.gait_generator_dim)):
            if l == len(AC_Args.gait_generator_dim) - 1:
                gait_generator_layers.append(
                    nn.Linear(AC_Args.gait_generator_dim[l], 16))
            else:
                gait_generator_layers.append(
                    nn.Linear(AC_Args.gait_generator_dim[l],
                              AC_Args.gait_generator_dim[l + 1]))
                gait_generator_layers.append(activation)
        self.gait_generator = nn.Sequential(*gait_generator_layers)


        # Policy
        actor_layers = []
        actor_layers.append(nn.Linear(AC_Args.actor_body_input_dim, AC_Args.actor_hidden_dims[0]))
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
        critic_layers.append(nn.Linear(AC_Args.critic_body_input_dim, AC_Args.critic_hidden_dims[0]))
        critic_layers.append(activation)
        for l in range(len(AC_Args.critic_hidden_dims)):
            if l == len(AC_Args.critic_hidden_dims) - 1:
                critic_layers.append(nn.Linear(AC_Args.critic_hidden_dims[l], 1))
            else:
                critic_layers.append(nn.Linear(AC_Args.critic_hidden_dims[l], AC_Args.critic_hidden_dims[l + 1]))
                critic_layers.append(activation)
        self.critic_body = nn.Sequential(*critic_layers)

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

    def act_update(self, obs, privileged_obs, observation_history, is_normal):
        ct = privileged_obs[...,-3:]
        ht = self.history_encoder(observation_history)
        vt = self.adaptation_module(observation_history)
        if is_normal:
            zt = self.gait_encoder(privileged_obs[...,-11:-3])
            zt = zt / torch.norm(zt, p=2, dim=-1, keepdim=True)
        else:
            zt = self.gait_generator(torch.cat((privileged_obs[...,-3:], obs),dim=-1))
            zt = zt / torch.norm(zt, p=2, dim=-1, keepdim=True)
        mean = self.actor_body(torch.cat((ct, ht, vt, zt), dim=-1))
        return mean

    def update_distribution(self, obs, privileged_obs, observation_history, is_normal):
        ct = privileged_obs[...,-3:]
        ht = self.history_encoder(observation_history)
        vt = self.adaptation_module(observation_history)
        if is_normal:
            zt = self.gait_encoder(privileged_obs[...,-11:-3])
            zt = zt / torch.norm(zt, p=2, dim=-1, keepdim=True)
        else:
            zt = self.gait_generator(torch.cat((privileged_obs[...,-3:], obs),dim=-1))
            zt = zt / torch.norm(zt, p=2, dim=-1, keepdim=True)
        mean = self.actor_body(torch.cat((ct, ht, vt, zt), dim=-1))
        self.distribution = Normal(mean, mean * 0. + self.std)

    def act(self, obs, privileged_obs, observation_history, is_normal, **kwargs):
        self.update_distribution(obs, privileged_obs, observation_history,is_normal)
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

    def evaluate(self, obs, privileged_observations, is_normal, **kwargs):
        if is_normal:
            zt = self.gait_encoder(privileged_observations[...,-11:-3])
            zt = zt / torch.norm(zt, p=2, dim=-1, keepdim=True)
        else:
            zt = self.gait_generator(torch.cat((privileged_observations[...,-3:], obs),dim=-1))
            zt = zt / torch.norm(zt, p=2, dim=-1, keepdim=True)
        privileged_observations = privileged_observations[..., :211]
        ct = privileged_observations[...,-3:]
        value = self.critic_body(torch.cat((zt, privileged_observations,obs,ct), dim=-1))
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
