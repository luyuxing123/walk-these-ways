import torch
import torch.nn as nn
import torch.utils.data
from torch import autograd
from .running_standard_scaler import RunningStandardScaler


class AMPDiscriminator(nn.Module):
    def __init__(
            self, input_dim, amp_reward_coef, hidden_layer_sizes, device, task_reward_lerp=0.0):
        super(AMPDiscriminator, self).__init__()

        self.device = device
        self.input_dim = input_dim

        self.amp_reward_coef = amp_reward_coef
        amp_layers = []
        curr_in_dim = input_dim
        for hidden_dim in hidden_layer_sizes:
            amp_layers.append(nn.Linear(curr_in_dim, hidden_dim))
            amp_layers.append(nn.ReLU())
            curr_in_dim = hidden_dim
        self.trunk = nn.Sequential(*amp_layers).to(device)
        self.amp_linear = nn.Linear(hidden_layer_sizes[-1], 1).to(device)

        self.trunk.train()
        self.amp_linear.train()
        self.obs_normalizer = RunningStandardScaler(input_dim)
        self.task_reward_lerp = task_reward_lerp

    def forward(self, x):
        h = self.trunk(x)
        d = self.amp_linear(h)
        return d

    def compute_grad_pen(self,
                         expert_state,
                         expert_next_state,
                         lambda_=10):
        expert_data = torch.cat([expert_state, expert_next_state], dim=-1)
        expert_data.requires_grad = True

        disc = self.amp_linear(self.trunk(expert_data))
        ones = torch.ones(disc.size(), device=disc.device)
        grad = autograd.grad(
            outputs=disc, inputs=expert_data,
            grad_outputs=ones, create_graph=True,
            retain_graph=True, only_inputs=True)[0]

        # Enforce that the grad norm approaches 0.
        grad_pen = lambda_ * (grad.norm(2, dim=1) - 0).pow(2).mean()
        return grad_pen

    def predict_amp_reward(
            self, state, next_state,  normalizer=None):
        with torch.no_grad():
            self.eval()
            norm_amp_obs = self.norm_obs(torch.cat([state, next_state], dim=-1))
            h = self.trunk(norm_amp_obs)
            d = self.amp_linear(h)

            #这里的0.01等于amp奖励函数系数0.5乘以dt 0.02
            style_reward = 0.01 * torch.clamp(1 - (1 / 4) * torch.square(d - 1), min=0)
            self.train()
        return style_reward.squeeze(), d.squeeze()

    def _lerp_reward(self, disc_r, task_r):
        r = (1.0 - self.task_reward_lerp) * disc_r + self.task_reward_lerp * task_r
        return r

    def norm_obs(self, obs, **kwargs):
        if self.obs_normalizer is not None:
            with torch.no_grad():
                obs = self.obs_normalizer(obs, **kwargs)
        return obs