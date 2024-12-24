import os
import copy

import isaacgym
from go1_gym.envs.base.legged_robot_config import Cfg
from go1_gym.envs.go1.velocity_tracking import VelocityTrackingEasyEnv
from go1_gym_learn.ppo_cse import Runner
import torch
import subprocess

from ml_logger import logger
from go1_gym.envs.wrappers.history_wrapper import HistoryWrapper
from go1_gym_learn.ppo_cse.actor_critic import AC_Args
from go1_gym_learn.ppo_cse.ppo import PPO_Args
from go1_gym_learn.ppo_cse import RunnerArgs
from go1_gym.envs.go1.go1_config import config_go1
import numpy as np



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
        base_path = '/home/ask-spc/luyx/walk-these-ways/runs/gait-conditioned-agility/2024-10-01/train/035000.050673'
        body_model_path=os.path.join(base_path,'checkpoints/body_latest.jit')
        adaptation_model_path=os.path.join(base_path,'checkpoints/adaptation_module_latest.jit')


        body_model = torch.jit.load(body_model_path)
        adaptation_model = torch.jit.load(adaptation_model_path)

        #mlp
        mlp_input = torch.randn(1, 2012)
        model_mlp = copy.deepcopy(body_model).to('cpu')
        torch.onnx.export(model_mlp, mlp_input, "/home/ask-spc/luyx/walk-these-ways/onnx/mlp.onnx")
        convert_onnx_to_mnn("/home/ask-spc/luyx/walk-these-ways/onnx/mlp.onnx",
                            "/home/ask-spc/luyx/walk-these-ways/mnn/mlp.mnn")

        #adaption

        adaption_input = torch.randn(1, 2010)
        model_adaption = copy.deepcopy(adaptation_model).to('cpu')
        torch.onnx.export(model_adaption, adaption_input, "/home/ask-spc/luyx/walk-these-ways/onnx/adaptation.onnx")
        # Convert to MNN
        convert_onnx_to_mnn("/home/ask-spc/luyx/walk-these-ways/onnx/adaptation.onnx",
                            "/home/ask-spc/luyx/walk-these-ways/mnn/adaptation.mnn")

if __name__ == '__main__':
    play()
