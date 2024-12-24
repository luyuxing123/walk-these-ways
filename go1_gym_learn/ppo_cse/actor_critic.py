import torch
import torch.nn as nn
from params_proto import PrefixProto
from torch.distributions import Normal
from ..utils import unpad_trajectories


class AC_Args(PrefixProto, cli=False):
    # policy
    init_noise_std = 1.0
    actor_hidden_dims = [512, 256, 128]
    critic_hidden_dims = [512, 256, 128]
    activation = 'elu'  # can be elu, relu, selu, crelu, lrelu, tanh, sigmoid
    adaptation_module_branch_hidden_dims = [256, 128]
    use_decoder = False

    adaptation_module_decoder_hidden_dims = [256, 128]  #[128, 256]
    action_decoder_hidden_ddims = [256, 128]    #[128, 256, 512]

    #added by luyx from wjz.add two encoder
    privileged_lantent_size = 8
    privileged_hidden_dims = [64, 32]
    terrain_lantent_size = 16
    terrain_hidden_dims = [256, 128]

    perception_dim=58-9
    privileged_dim=24   #+2
    terrain_dim = 187
    actor_dim = perception_dim + privileged_lantent_size + terrain_lantent_size

    adaptation_history_nums = 30
    adaptation_input = adaptation_history_nums*perception_dim
    adaptation_output = privileged_lantent_size + terrain_lantent_size

    #student推理teacher特权信息的网络是否需要lstm
    #True表示需要，输入观测历史到lstm，输出特权、地形信息
    #False表示不需要，直接使用mlp
    adaptation_lstm = True
    is_teacher = True
    use_help_decoder = True

class Memory(torch.nn.Module):
    def __init__(self, input_size, type='lstm', num_layers=1, hidden_size=256):
        super().__init__()
        # RNN
        rnn_cls = nn.GRU if type.lower() == 'gru' else nn.LSTM  # 默认第一个维度是时间序列长度
        self.rnn = rnn_cls(input_size=input_size,
                           hidden_size=hidden_size, num_layers=num_layers)
        self.hidden_states = None

    def forward(self, input, masks=None, hidden_states=None):
        batch_mode = masks is not None
        #batch_mode表示当前是否处于策略更新模式
        #如果是采样模式，直接输入obs即可，得到act
        #如果是策略更新模式，需要从缓冲区中获取历史轨迹，根据历史obs计算策略更新后obs对应的新的act（旧的act存储在缓冲区中）
        #因此，需要记录历史数据中相对应的hidden_states，使用相同的值计算act
        if batch_mode:
            if hidden_states is None:
                raise ValueError(
                    "Hidden states not passed to memory module during policy update")
            out, _ = self.rnn(input, hidden_states)
            out = unpad_trajectories(out, masks)
        else:
            out, self.hidden_states = self.rnn(
                input.unsqueeze(0), self.hidden_states)
        return out

    # def forward(self, input, h0,c0):
    #     out, (hx, cx) = self.rnn(input, (h0,c0))
    #     return out, (hx, cx)

def MLP(input_dim, output_dim, hidden_dims, activation_name, output_activation=None):
    activation = get_activation(activation_name)  # 激活函数类型
    layers = []
    layers.append(nn.Linear(input_dim, hidden_dims[0]))
    layers.append(activation)
    for l in range(len(hidden_dims)):
        if l == len(hidden_dims) - 1:
            layers.append(nn.Linear(hidden_dims[l], output_dim))
            if output_activation is not None:
                output_activation = get_activation(output_activation)
                layers.append(output_activation)
        else:
            layers.append(nn.Linear(hidden_dims[l], hidden_dims[l + 1]))
            layers.append(activation)

    return layers

class ActorCritic(nn.Module):
    is_recurrent = False

    def __init__(self, num_obs,
                 num_privileged_obs,
                 num_obs_history,
                 num_actions,
                 isteacher,
                 **kwargs):
        self.is_recurrent = AC_Args.adaptation_lstm #true表示循环神经网络（lstm），false表示简单的mlp
        self.current_teacher = isteacher
        AC_Args.is_teacher = isteacher
        if kwargs:
            print("ActorCritic.__init__ got unexpected arguments, which will be ignored: " + str(
                [key for key in kwargs.keys()]))
        self.decoder = AC_Args.use_decoder
        super().__init__()

        self.num_obs_history = AC_Args.adaptation_input
        self.num_privileged_obs = AC_Args.adaptation_output

        activation = get_activation(AC_Args.activation)

        # Policy
        actor_layers = []
        actor_layers.append(nn.Linear(AC_Args.actor_dim, AC_Args.actor_hidden_dims[0]))
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
        critic_layers.append(nn.Linear(AC_Args.actor_dim, AC_Args.critic_hidden_dims[0]))
        critic_layers.append(activation)
        for l in range(len(AC_Args.critic_hidden_dims)):
            if l == len(AC_Args.critic_hidden_dims) - 1:
                critic_layers.append(nn.Linear(AC_Args.critic_hidden_dims[l], 1) )
            else:
                critic_layers.append(nn.Linear(AC_Args.critic_hidden_dims[l], AC_Args.critic_hidden_dims[l + 1]))
                critic_layers.append(activation)
        self.critic_body = nn.Sequential(*critic_layers)

        if self.current_teacher:
            terrain_layers = []
            terrain_layers.append(nn.Linear(AC_Args.terrain_dim , AC_Args.terrain_hidden_dims[0]))
            terrain_layers.append(activation)
            for l in range(len(AC_Args.terrain_hidden_dims)):
                if l == len(AC_Args.terrain_hidden_dims) - 1:
                    terrain_layers.append(nn.Linear(AC_Args.terrain_hidden_dims[l], AC_Args.terrain_lantent_size))
                else:
                    terrain_layers.append(nn.Linear(AC_Args.terrain_hidden_dims[l], AC_Args.terrain_hidden_dims[l + 1]))
                    terrain_layers.append(activation)
            self.terrain_body = nn.Sequential(*terrain_layers)

            privileged_layers = []
            privileged_layers.append(nn.Linear(AC_Args.privileged_dim , AC_Args.privileged_hidden_dims[0]))
            privileged_layers.append(activation)
            for l in range(len(AC_Args.privileged_hidden_dims)):
                if l == len(AC_Args.privileged_hidden_dims) - 1:
                    privileged_layers.append(nn.Linear(AC_Args.privileged_hidden_dims[l], AC_Args.privileged_lantent_size))
                else:
                    privileged_layers.append(nn.Linear(AC_Args.privileged_hidden_dims[l], AC_Args.privileged_hidden_dims[l + 1]))
                    privileged_layers.append(activation)
            self.privileged_body = nn.Sequential(*privileged_layers)
        else:
            # Adaptation module for student to infer privileged and terrain information from teacher
            if not AC_Args.adaptation_lstm:
                adaptation_module_layers = []
                adaptation_module_layers.append(nn.Linear(self.num_obs_history, AC_Args.adaptation_module_branch_hidden_dims[0]))
                adaptation_module_layers.append(activation)
                for l in range(len(AC_Args.adaptation_module_branch_hidden_dims)):
                    if l == len(AC_Args.adaptation_module_branch_hidden_dims) - 1:
                        adaptation_module_layers.append(
                            nn.Linear(AC_Args.adaptation_module_branch_hidden_dims[l], self.num_privileged_obs))
                    else:
                        adaptation_module_layers.append(
                            nn.Linear(AC_Args.adaptation_module_branch_hidden_dims[l],
                                      AC_Args.adaptation_module_branch_hidden_dims[l + 1]))
                        adaptation_module_layers.append(activation)
                self.adaptation_module = nn.Sequential(*adaptation_module_layers)

                if AC_Args.use_help_decoder:
                    adaptation_decoder_layers = []
                    adaptation_decoder_layers.append(
                        nn.Linear(self.num_privileged_obs, AC_Args.adaptation_module_decoder_hidden_dims[0]))
                    adaptation_decoder_layers.append(activation)
                    for l in range(len(AC_Args.adaptation_module_decoder_hidden_dims)):
                        if l == len(AC_Args.adaptation_module_decoder_hidden_dims) - 1:
                            adaptation_decoder_layers.append(
                                nn.Linear(AC_Args.adaptation_module_decoder_hidden_dims[l], AC_Args.privileged_dim + AC_Args.terrain_dim))
                        else:
                            adaptation_decoder_layers.append(
                                nn.Linear(AC_Args.adaptation_module_decoder_hidden_dims[l],
                                          AC_Args.adaptation_module_decoder_hidden_dims[l + 1]))
                            adaptation_decoder_layers.append(activation)
                    self.adaptation_decoder = nn.Sequential(*adaptation_decoder_layers)

                    act_decoder_layers = []
                    act_decoder_layers.append(
                        nn.Linear(12, AC_Args.action_decoder_hidden_ddims[0]))
                    act_decoder_layers.append(activation)
                    for l in range(len(AC_Args.action_decoder_hidden_ddims)):
                        if l == len(AC_Args.action_decoder_hidden_ddims) - 1:
                            act_decoder_layers.append(
                                nn.Linear(AC_Args.action_decoder_hidden_ddims[l], AC_Args.actor_dim))
                        else:
                            act_decoder_layers.append(
                                nn.Linear(AC_Args.action_decoder_hidden_ddims[l],
                                          AC_Args.action_decoder_hidden_ddims[l + 1]))
                            act_decoder_layers.append(activation)
                    self.act_decoder = nn.Sequential(*act_decoder_layers)
            else:
                self.memory_encoder = Memory(
                    AC_Args.perception_dim, type='lstm', num_layers=3, hidden_size=256)
                self.latent_head = nn.Sequential(*MLP(256, 24,
                                                      [64], 'elu', 'tanh'))

        # print(f"Adaptation Module: {self.adaptation_module}")
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

    def get_help_decoder_output(self,input):
        return self.adaptation_decoder(input)

    def get_act_decoder_output(self,input):
        return self.act_decoder(input)

    def get_encoded_privileged(self,obs,privileged_obs):
        obs = torch.cat((obs,privileged_obs),dim=-1)
        privileged_input = obs[:,
                           -(AC_Args.privileged_dim + AC_Args.terrain_dim):-AC_Args.terrain_dim]
        privileged_output = self.privileged_body(privileged_input)
        terrain_input = obs[:, -AC_Args.terrain_dim:]
        terrain_output = self.terrain_body(terrain_input)
        return torch.cat(( privileged_output, terrain_output),dim=-1)

    def forward(self,obs):
        #play且使用jit时使用
        if self.current_teacher:
            perception_input = obs[:, :AC_Args.perception_dim]
            if AC_Args.teacher_use_privilege:
                privileged_input = obs[:,
                                   -(AC_Args.privileged_dim + AC_Args.terrain_dim):-AC_Args.terrain_dim]
                privileged_output = self.privileged_body(privileged_input)
                terrain_input = obs[:, -AC_Args.terrain_dim:]
                terrain_output = self.terrain_body(terrain_input)
                perception_input = torch.cat((perception_input, privileged_output, terrain_output), dim=-1)
            mean = self.actor_body(perception_input)
        else:
            if not AC_Args.adaptation_lstm:
                latent = self.adaptation_module(obs)
                obs = obs[:,(AC_Args.adaptation_history_nums-1)*AC_Args.perception_dim:]
            else:
                memory = self.memory_encoder(obs,None, None).squeeze(0)
                latent = self.latent_head(memory)
            mean = self.actor_body(torch.cat((obs,latent), dim=-1))
        return mean


    def act_student(self,obs):
        #不使用lstm
        if not AC_Args.adaptation_lstm:
            latent = self.adaptation_module(obs)
            obs = obs[:, (AC_Args.adaptation_history_nums-1) * AC_Args.perception_dim :]
            mean = self.actor_body(torch.cat((obs, latent), dim=-1))
        else:
            #仅输入当前的数据，不是历史数据
            #RNN需要每次输入数据长度相同,所以会有填充操作。输入给RNN之后，需要去除填充的值
            memory = self.memory_encoder(obs, None, None).squeeze(0)
            latent = self.latent_head(memory)
            mean = self.actor_body(torch.cat((obs, latent), dim=-1))
        return mean

    def act_teacher(self, obs):
        # 不使用lstm
        privileged_input = obs[:,
                           -(AC_Args.privileged_dim + AC_Args.terrain_dim):-AC_Args.terrain_dim]
        privileged_output = self.privileged_body(privileged_input)
        terrain_input = obs[:, -AC_Args.terrain_dim:]
        terrain_output = self.terrain_body(terrain_input)
        perception_input = obs[:, :AC_Args.perception_dim]
        mean = self.actor_body(torch.cat((perception_input, privileged_output, terrain_output), dim=-1))
        return mean


    def get_student_act(self,obs,masks_batch=None,hid_states_batch=None):
        #策略更新时使用的
        if not AC_Args.adaptation_lstm:
            latent = self.adaptation_module(obs)
            obs = obs[:, (AC_Args.adaptation_history_nums - 1) * AC_Args.perception_dim:]
            mean = self.actor_body(torch.cat((obs, latent), dim=-1))
        else:
            #仅输入当前的数据，不是历史数据
            #RNN需要每次输入数据长度相同,所以会有填充操作。输入给RNN之后，需要去除填充的值
            memory = self.memory_encoder(obs, masks_batch, hid_states_batch).squeeze(0)
            latent = self.latent_head(memory)
            unpad_obs = unpad_trajectories(obs, masks_batch)
            mean = self.actor_body(torch.cat((unpad_obs, latent), dim=-1))
        return mean,latent

    def get_teacher_act(self,obs,privileged_obs_batch,masks_batch):
        if masks_batch is not None:
            obs = unpad_trajectories(obs, masks_batch)
            privileged_obs_batch = unpad_trajectories(privileged_obs_batch, masks_batch)
        current_obs = torch.cat((obs , privileged_obs_batch),dim=-1)
        privileged_input = current_obs[...,
                           -(AC_Args.privileged_dim + AC_Args.terrain_dim):-AC_Args.terrain_dim]
        privileged_output = self.privileged_body(privileged_input)
        terrain_input = current_obs[..., -AC_Args.terrain_dim:]
        terrain_output = self.terrain_body(terrain_input)
        perception_input = current_obs[..., :AC_Args.perception_dim]
        mean = self.actor_body(torch.cat((perception_input, privileged_output, terrain_output), dim=-1))
        return mean, torch.cat((privileged_output, terrain_output),dim=-1)

    def get_student_value(self,obs,masks_batch=None,hid_states_batch=None):
        if not AC_Args.adaptation_lstm:
            latent = self.adaptation_module(obs)
            obs = obs[:, (AC_Args.adaptation_history_nums - 1) * AC_Args.perception_dim:]
            mean = self.critic_body(torch.cat((obs, latent), dim=-1))
        else:
            memory = self.memory_encoder(obs, masks_batch, hid_states_batch).squeeze(0)
            latent = self.latent_head(memory)
            unpad_obs = unpad_trajectories(obs, masks_batch)
            mean = self.actor_body(torch.cat((unpad_obs, latent), dim=-1))
        return mean

    def get_teacher_value(self,obs,privileged_obs_batch,masks_batch):
        obs = unpad_trajectories(obs, masks_batch)
        privileged_obs_batch = unpad_trajectories(privileged_obs_batch, masks_batch)
        current_obs = torch.cat((obs , privileged_obs_batch),dim=-1)
        privileged_input = current_obs[...,
                           -(AC_Args.privileged_dim + AC_Args.terrain_dim):-AC_Args.terrain_dim]
        privileged_output = self.privileged_body(privileged_input)
        terrain_input = current_obs[..., -AC_Args.terrain_dim:]
        terrain_output = self.terrain_body(terrain_input)
        perception_input = current_obs[..., :AC_Args.perception_dim]
        mean = self.critic_body(torch.cat((perception_input, privileged_output, terrain_output), dim=-1))
        return mean.detach()

    @property
    def action_mean(self):
        return self.distribution.mean

    @property
    def action_std(self):
        return self.distribution.stddev

    @property
    def entropy(self):
        return self.distribution.entropy().sum(dim=-1)

    def update_distribution(self, obs):
        if self.current_teacher:
            privileged_input = obs[:,
                               -(AC_Args.privileged_dim + AC_Args.terrain_dim):-AC_Args.terrain_dim]
            privileged_output = self.privileged_body(privileged_input)
            terrain_input = obs[:, -AC_Args.terrain_dim:]
            terrain_output = self.terrain_body(terrain_input)
            perception_input = obs[:, :AC_Args.perception_dim]
            mean = self.actor_body(torch.cat((perception_input, privileged_output, terrain_output), dim=-1))
        else:
            if not AC_Args.adaptation_lstm:
                latent = self.adaptation_module(obs)
                obs = obs[:, (AC_Args.adaptation_history_nums-1) * AC_Args.perception_dim:]
            else:
                memory = self.memory_encoder(obs, None, None).squeeze(0)
                latent = self.latent_head(memory)
            mean = self.actor_body(torch.cat((obs,latent), dim=-1))

        self.distribution = Normal(mean, mean * 0. + self.std)

    def act(self, obs, **kwargs):
        self.update_distribution(obs)
        return self.distribution.sample()

    def get_actions_log_prob(self, actions):
        return self.distribution.log_prob(actions).sum(dim=-1)

    def act_expert(self, ob, policy_info={}):
        return self.act_teacher(ob["obs_history"], ob["privileged_obs"])

    def act_inference(self, ob, policy_info={}):
        return self.act_student(ob["obs_history"], policy_info=policy_info)



    def evaluate(self, obs, privileged_observations, **kwargs):
        if self.current_teacher:
            privileged_input = obs[:,-(AC_Args.privileged_dim+ AC_Args.terrain_dim):-AC_Args.terrain_dim]
            privileged_output = self.privileged_body(privileged_input)
            terrain_input = obs[:,-AC_Args.terrain_dim:]
            terrain_output = self.terrain_body(terrain_input)
            perception_input = obs[:,:AC_Args.perception_dim]
            value = self.critic_body(torch.cat((perception_input,privileged_output,terrain_output), dim=-1))
        else:
            if not AC_Args.adaptation_lstm:
                latent = self.adaptation_module(obs)
                obs = obs[:, (AC_Args.adaptation_history_nums - 1) * AC_Args.perception_dim:]
            else:
                memory = self.memory_encoder(obs, None, None).squeeze(0)
                latent = self.latent_head(memory)
            value = self.critic_body(torch.cat((obs,latent), dim=-1))
        return value

    def get_student_latent(self, obs):
        return self.adaptation_module(obs)

    def get_hidden_states(self):
        return self.memory_encoder.hidden_states, None

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
