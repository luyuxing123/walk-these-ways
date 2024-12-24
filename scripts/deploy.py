import torch
import torch.nn as nn
import os
import subprocess
from go1_gym.envs.base.legged_robot_config import Cfg
import copy
from torch.distributions import Normal
from go1_gym_learn.ppo_cse.actor_critic import AC_Args


# class AC_Args():
#     # policy
#     init_noise_std = 1.0
#     actor_hidden_dims = [256, 128, 64]
#     critic_hidden_dims = [512, 256, 128]
#     adaptation_module_branch_hidden_dims = [256,128]
#     # adaptation_module_branch_hidden_dims = [64,32]
#     history_encoder_dims = [256, 128]
#     activation = 'elu'  # can be elu, relu, selu, crelu, lrelu, tanh, sigmoid



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
        super().__init__()

        self.num_obs_history = num_obs_history
        self.num_privileged_obs = num_privileged_obs

        activation = nn.ELU()

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

        # Action noise
        self.std = nn.Parameter(AC_Args.init_noise_std * torch.ones(num_actions))
        self.distribution = None
        # disable args validation for speedup
        Normal.set_default_validate_args = False


    def forward(self,obs_history,command):
        vt = self.adaptation_module(obs_history)
        ht = self.history_encoder(obs_history)
        action = self.actor_body(torch.cat((ht,vt,command),dim=-1))
        return action


def convert_onnx_to_mnn(onnx_path, mnn_path):
    command = '$HOME/Documents/MNN/build/MNNConvert -f ONNX --modelFile {0} --MNNModel {1} --bizCode biz'.format(onnx_path, mnn_path)
    file_name = onnx_path.split('/')[-1]
    result = subprocess.run(command, shell=True, check=True)
    if result.returncode == 0:
        print("{0} convert to MNN successfully.".format(file_name))
    else:
        raise ValueError("MNN converter of {0} failed with return code:{1}".format(file_name, result.returncode))


def play():
    with torch.inference_mode():
        base_path = '/home/ask-spc/luyx/walk-these-ways-asymmetric/runs/gait-conditioned-agility/2024-11-26/train/030015.889261'
        state_dict=torch.load(base_path+'/checkpoints/ac_weights_last.pt')
        # state_dict=torch.load(base_path+'/checkpoints/ac_weights_010000.pt')

        body = ActorCritic(Cfg.env.num_observations,
                           Cfg.env.num_privileged_obs,
                           Cfg.env.num_observation_history*Cfg.env.num_observations,
                           Cfg.env.num_actions,
                        )

        body.load_state_dict(state_dict)

        obs_history = torch.randn(42*5)
        command = torch.randn(13)
        model_actor_body = copy.deepcopy(body)
        torch.onnx.export(model_actor_body, (obs_history, command), "/home/ask-spc/luyx/walk-these-ways-asymmetric/onnx/mlp.onnx"
                          , input_names=['obs_history', 'command'])
        convert_onnx_to_mnn("/home/ask-spc/luyx/walk-these-ways-asymmetric/onnx/mlp.onnx",
                            "/home/ask-spc/luyx/walk-these-ways-asymmetric/mnn/mlp.mnn")

        os.system('scp ../mnn/mlp.mnn  /home/ask-spc/luyx/catkin_ws/src/walk-these-ways-asymmetric-gazebo/policies/')


if __name__ == '__main__':
    play()

