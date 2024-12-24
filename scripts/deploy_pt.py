import os
import copy
import isaacgym
from go1_gym.envs.base.legged_robot_config import Cfg
from go1_gym.envs.go1.velocity_tracking import VelocityTrackingEasyEnv
from go1_gym_learn.ppo_cse import Runner
import torch
import subprocess

from go1_gym_learn.ppo_cse import ActorCritic
from go1_gym_learn.ppo_cse import AC_Args




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
        base_path = '/home/ask-spc/luyx/walk-these-ways/runs/gait-conditioned-agility/2024-10-26/train/235347.995669'
        state_dict=torch.load(base_path+'/checkpoints/ac_weights_last.pt',map_location='cpu')

        AC_Args.adaptation_lstm = False
        AC_Args.use_help_decoder = True

        body = ActorCritic(Cfg.env.num_observations,
                           Cfg.env.num_privileged_obs,
                           Cfg.env.num_observation_history*Cfg.env.num_observations,
                           Cfg.env.num_actions,
                        False).to('cpu')

        body.load_state_dict(state_dict)


        if AC_Args.adaptation_lstm:
            action_body = body.actor_body
            memory_encoder = body.memory_encoder
            latent_head = body.latent_head

            #mlp
            mlp_input = torch.randn(73)
            model_mlp = copy.deepcopy(action_body).to('cpu')
            torch.onnx.export(model_mlp, mlp_input, "/home/ask-spc/luyx/walk-these-ways/onnx/lstm/mlp.onnx")
            convert_onnx_to_mnn("/home/ask-spc/luyx/walk-these-ways/onnx/lstm/mlp.onnx",
                                "/home/ask-spc/luyx/walk-these-ways/mnn/lstm/mlp.mnn")


            lstm_input = torch.randn(1,49)
            h0 = torch.randn(3, 256)  # 初始的 h 隐藏状态
            c0 = torch.randn(3, 256)  # 初始的 c 细胞状态
            model_lstm = copy.deepcopy(memory_encoder).to('cpu')
            torch.onnx.export(model_lstm, (lstm_input,h0,c0), "/home/ask-spc/luyx/walk-these-ways/onnx/lstm/lstm.onnx"
                            ,input_names=['input', 'h0', 'c0'],
                            output_names=['output', 'hx', 'cx'] )
            # Convert to MNN
            convert_onnx_to_mnn("/home/ask-spc/luyx/walk-these-ways/onnx/lstm/lstm.onnx",
                                "/home/ask-spc/luyx/walk-these-ways/mnn/lstm/lstm.mnn")


            encoder_input = torch.randn(256)
            model_encoder = copy.deepcopy(latent_head).to('cpu')
            torch.onnx.export(model_encoder, encoder_input, "/home/ask-spc/luyx/walk-these-ways/onnx/lstm/encoder.onnx")
            # Convert to MNN
            convert_onnx_to_mnn("/home/ask-spc/luyx/walk-these-ways/onnx/lstm/encoder.onnx",
                                "/home/ask-spc/luyx/walk-these-ways/mnn/lstm/encoder.mnn")

            os.system('scp ../mnn/lstm/encoder.mnn ../mnn/lstm/lstm.mnn ../mnn/lstm/mlp.mnn /home/ask-spc/luyx/catkin_ws/src/walk-these-ways-ts-gazebo/policies/lstm')

        else:
            action_body = body.actor_body
            adaptation_model = body.adaptation_module

            # mlp
            mlp_input = torch.randn(73)
            model_mlp = copy.deepcopy(action_body).to('cpu')
            torch.onnx.export(model_mlp, mlp_input, "/home/ask-spc/luyx/walk-these-ways/onnx/only_mlp/mlp.onnx")
            convert_onnx_to_mnn("/home/ask-spc/luyx/walk-these-ways/onnx/only_mlp/mlp.onnx",
                                "/home/ask-spc/luyx/walk-these-ways/mnn/only_mlp/mlp.mnn")

            adaption_input = torch.randn(1, 1470)
            model_adaption = copy.deepcopy(adaptation_model).to('cpu')
            torch.onnx.export(model_adaption, adaption_input, "/home/ask-spc/luyx/walk-these-ways/onnx/only_mlp/adaptation.onnx")
            # Convert to MNN
            convert_onnx_to_mnn("/home/ask-spc/luyx/walk-these-ways/onnx/only_mlp/adaptation.onnx",
                                "/home/ask-spc/luyx/walk-these-ways/mnn/only_mlp/adaptation.mnn")

            os.system('scp ../mnn/only_mlp/adaptation.mnn ../mnn/only_mlp/mlp.mnn /home/ask-spc/luyx/catkin_ws/src/walk-these-ways-ts-gazebo/policies/only_mlp')


if __name__ == '__main__':
    play()
