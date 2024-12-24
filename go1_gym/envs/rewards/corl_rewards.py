import torch
import numpy as np
from go1_gym.utils.math_utils import quat_apply_yaw, wrap_to_pi, get_scale_shift
from isaacgym.torch_utils import *
from isaacgym import gymapi

class CoRLRewards:
    def __init__(self, env):
        self.env = env

    def load_env(self, env):
        self.env = env

    # ------------ reward functions----------------
    def _reward_tracking_lin_vel(self):
        # Tracking of linear velocity commands (xy axes)
        lin_vel_error = torch.sum(torch.square(self.env.commands[:, :2] - self.env.base_lin_vel[:, :2]), dim=1)
        return torch.exp(-lin_vel_error / self.env.cfg.rewards.tracking_sigma)

    def _reward_tracking_ang_vel(self):
        # Tracking of angular velocity commands (yaw)
        ang_vel_error = torch.square(self.env.commands[:, 2] - self.env.base_ang_vel[:, 2])
        return torch.exp(-ang_vel_error / self.env.cfg.rewards.tracking_sigma_yaw)

    def _reward_lin_vel_z(self):
        # Penalize z axis base linear velocity
        return torch.square(self.env.base_lin_vel[:, 2])

    def _reward_ang_vel_xy(self):
        # Penalize xy axes base angular velocity
        return torch.sum(torch.square(self.env.base_ang_vel[:, :2]), dim=1)

    def _reward_orientation(self):
        # Penalize non flat base orientation
        return torch.sum(torch.square(self.env.projected_gravity[:, :2]), dim=1)

    def _reward_torques(self):
        # Penalize torques
        return torch.sum(torch.square(self.env.torques), dim=1)

    def _reward_dof_acc(self):
        # Penalize dof accelerations
        return torch.sum(torch.square((self.env.last_dof_vel - self.env.dof_vel) / self.env.dt), dim=1)

    def _reward_action_rate(self):
        # Penalize changes in actions
        return torch.sum(torch.square(self.env.last_actions - self.env.actions), dim=1)

    def _reward_collision(self):
        # Penalize collisions on selected bodies
        return torch.sum(1. * (torch.norm(self.env.contact_forces[:, self.env.penalised_contact_indices, :], dim=-1) > 0.1),
                         dim=1)

    def _reward_dof_pos_limits(self):
        # Penalize dof positions too close to the limit
        out_of_limits = -(self.env.dof_pos - self.env.dof_pos_limits[:, 0]).clip(max=0.)  # lower limit
        out_of_limits += (self.env.dof_pos - self.env.dof_pos_limits[:, 1]).clip(min=0.)
        return torch.sum(out_of_limits, dim=1)

    def _reward_jump(self):
        body_height = self.env.base_pos[:, 2]
        jump_height_target = self.env.commands[:, 3]
        reward = - torch.square(body_height - jump_height_target)
        return reward

    def _reward_tracking_contacts_shaped_force(self):
        foot_forces = torch.norm(self.env.contact_forces[:, self.env.feet_indices, :], dim=-1)
        desired_contact = self.env.desired_contact_states

        reward = 0
        for i in range(4):
            reward += - (1 - desired_contact[:, i]) * (
                        1 - torch.exp(-1 * foot_forces[:, i] ** 2 / self.env.cfg.rewards.gait_force_sigma))
        return reward / 4

    def _reward_tracking_contacts_shaped_vel(self):
        foot_velocities = torch.norm(self.env.foot_velocities, dim=2).view(self.env.num_envs, -1)
        desired_contact = self.env.desired_contact_states
        reward = 0
        for i in range(4):
            reward += - (desired_contact[:, i] * (
                        1 - torch.exp(-1 * foot_velocities[:, i] ** 2 / self.env.cfg.rewards.gait_vel_sigma)))
        return reward / 4

    def _reward_dof_pos(self):
        # Penalize dof positions
        return torch.sum(torch.square(self.env.dof_pos - self.env.default_dof_pos), dim=1)

    def _reward_dof_vel(self):
        # Penalize dof velocities
        return torch.sum(torch.square(self.env.dof_vel), dim=1)

    def _reward_base_height(self):
        # Penalize base height away from target
        base_height = torch.mean(self.env.root_states[:, 2].unsqueeze(1) - self.env.measured_heights, dim=1)
        error = torch.abs(self.env.commands[:, 3] - base_height)
        rew = 1 - torch.exp(-1 * error)
        return rew

    def _reward_action_smoothness_1(self):
        # Penalize changes in actions
        diff = torch.square(self.env.joint_pos_target[:, :self.env.num_actuated_dof] - self.env.last_joint_pos_target[:, :self.env.num_actuated_dof])
        diff = diff * (self.env.last_actions[:, :self.env.num_dof] != 0)  # ignore first step
        return torch.sum(diff, dim=1)

    def _reward_action_smoothness_2(self):
        # Penalize changes in actions
        diff = torch.square(self.env.joint_pos_target[:, :self.env.num_actuated_dof] - 2 * self.env.last_joint_pos_target[:, :self.env.num_actuated_dof] + self.env.last_last_joint_pos_target[:, :self.env.num_actuated_dof])
        diff = diff * (self.env.last_actions[:, :self.env.num_dof] != 0)  # ignore first step
        diff = diff * (self.env.last_last_actions[:, :self.env.num_dof] != 0)  # ignore second step
        return torch.sum(diff, dim=1)

    def _reward_feet_slip(self):
        contact = self.env.contact_forces[:, self.env.feet_indices, 2] > 1.
        contact_filt = torch.logical_or(contact, self.env.last_contacts)
        self.env.last_contacts = contact
        foot_velocities = torch.square(torch.norm(self.env.foot_velocities[:, :, 0:2], dim=2).view(self.env.num_envs, -1))
        rew_slip = torch.sum(contact_filt * foot_velocities, dim=1)
        return rew_slip

    def _reward_feet_contact_vel(self):
        reference_heights = 0
        near_ground = self.env.foot_positions[:, :, 2] - reference_heights < 0.03
        foot_velocities = torch.square(torch.norm(self.env.foot_velocities[:, :, 0:3], dim=2).view(self.env.num_envs, -1))
        rew_contact_vel = torch.sum(near_ground * foot_velocities, dim=1)
        return rew_contact_vel

    def _reward_feet_contact_forces(self):
        # penalize high contact forces
        return torch.sum((torch.norm(self.env.contact_forces[:, self.env.feet_indices, :],
                                     dim=-1) - self.env.cfg.rewards.max_contact_force).clip(min=0.), dim=1)

    def _reward_feet_clearance_cmd_linear(self):
        phases = 1 - torch.abs(1.0 - torch.clip((self.env.foot_indices * 2.0) - 1.0, 0.0, 1.0) * 2.0)
        foot_height = (self.env.foot_positions[:, :, 2]).view(self.env.num_envs, -1)# - reference_heights
        target_height = self.env.commands[:, 9].unsqueeze(1) * phases + 0.02 # offset for foot radius 2cm
        rew_foot_clearance = torch.square(target_height - foot_height) * (1 - self.env.desired_contact_states)
        return torch.sum(rew_foot_clearance, dim=1)

    def _reward_feet_impact_vel(self):
        prev_foot_velocities = self.env.prev_foot_velocities[:, :, 2].view(self.env.num_envs, -1)
        contact_states = torch.norm(self.env.contact_forces[:, self.env.feet_indices, :], dim=-1) > 1.0

        rew_foot_impact_vel = contact_states * torch.square(torch.clip(prev_foot_velocities, -100, 0))

        return torch.sum(rew_foot_impact_vel, dim=1)


    def _reward_collision(self):
        # Penalize collisions on selected bodies
        return torch.sum(1. * (torch.norm(self.env.contact_forces[:, self.env.penalised_contact_indices, :], dim=-1) > 0.1),
                         dim=1)

    def _reward_orientation_control(self):
        # Penalize non flat base orientation
        roll_pitch_commands = self.env.commands[:, 10:12]
        quat_roll = quat_from_angle_axis(-roll_pitch_commands[:, 1],
                                         torch.tensor([1, 0, 0], device=self.env.device, dtype=torch.float))
        quat_pitch = quat_from_angle_axis(-roll_pitch_commands[:, 0],
                                          torch.tensor([0, 1, 0], device=self.env.device, dtype=torch.float))

        desired_base_quat = quat_mul(quat_roll, quat_pitch)
        desired_projected_gravity = quat_rotate_inverse(desired_base_quat, self.env.gravity_vec)

        return torch.sum(torch.square(self.env.projected_gravity[:, :2] - desired_projected_gravity[:, :2]), dim=1)

    def _reward_raibert_heuristic(self):
        #cur_footsteps_translated：当前机器人各脚的位置与机器人底座（base）的相对位置。
        #self.env.foot_positions是机器人的脚位置，self.env.base_pos是底座位置，通过相减获得脚相对底座的位置。
        #4096*4*3
        # if self.env.cfg.commands.num_commands<=3:
        #     return torch.tensor(0)
        cur_footsteps_translated = self.env.foot_positions - self.env.base_pos.unsqueeze(1)
        footsteps_in_body_frame = torch.zeros(self.env.num_envs, 4, 3, device=self.env.device)
        for i in range(4):
            #四只脚相对于机器人的身体框架（body frame）的位置
            footsteps_in_body_frame[:, i, :] = quat_apply_yaw(quat_conjugate(self.env.base_quat),
                                                              cur_footsteps_translated[:, i, :])

        # nominal positions: [FR, FL, RR, RL]
        #desired_stance_width：期望的步伐宽度
        #desired_ys_nom：每只脚的期望横向（Y轴）位置（中心为原点）
        if self.env.cfg.commands.num_commands >= 13:
            desired_stance_width = self.env.commands[:, 12:13]
            desired_ys_nom = torch.cat([desired_stance_width / 2, -desired_stance_width / 2, desired_stance_width / 2, -desired_stance_width / 2], dim=1)
        else:
            desired_stance_width = 0.3
            desired_ys_nom = torch.tensor([desired_stance_width / 2,  -desired_stance_width / 2, desired_stance_width / 2, -desired_stance_width / 2], device=self.env.device).unsqueeze(0)

        #desired_stance_length：期望的步伐长度
        #desired_xs_nom：每只脚的期望纵向（X轴）位置
        if self.env.cfg.commands.num_commands >= 14:
            desired_stance_length = self.env.commands[:, 13:14]
            desired_xs_nom = torch.cat([desired_stance_length / 2, desired_stance_length / 2, -desired_stance_length / 2, -desired_stance_length / 2], dim=1)
        else:
            #luyx changed
            desired_stance_length = 0.38
            desired_xs_nom = torch.tensor([desired_stance_length / 2,  desired_stance_length / 2, -desired_stance_length / 2, -desired_stance_length / 2], device=self.env.device).unsqueeze(0)

        # raibert offsets   Raibert偏移量计算
        #phase：与步伐相位相关的值
        #foot_indices：由一个全局时间变量加上各个脚的角度所得的值，0-0.5表示站立相，0.5-1表示摆动相
        #todo 修改phases
        phases = torch.abs(1.0 - (self.env.foot_indices * 2.0)) * 1.0 - 0.5
        # phases = torch.where(self.env.foot_indices <= 0.5, torch.tensor(0.0,  device='cuda:0'), torch.tensor(1.0, device='cuda:0'))

        frequencies = self.env.cfg.commands.limit_gait_frequency[0]   #步伐频率
        yaw_vel_des = self.env.commands[:, 2:3]
        #原代码：
        # x_vel_des = self.env.commands[:, 0:1]
        # y_vel_des = yaw_vel_des * desired_stance_length / 2
        # desired_xs_offset = phases * x_vel_des * (0.5 / freuencies.unsqueeze(1))
        # desired_ys_offset = phases * y_vel_des * (0.5 / frequencies.unsqueeze(1))
        # desired_ys_offset[:, 2:4] *= -1

        #修改后：
        x_vel_des_cmd = self.env.commands[:, 0:1]
        x_vel_des_yaw = yaw_vel_des * desired_stance_width / 2
        y_vel_des_cmd = self.env.commands[:, 1:2]
        y_vel_des_yaw = yaw_vel_des * desired_stance_length / 2

        #todo：具体的计算方式
        # desired_ys_offset_cmd = phases * y_vel_des_cmd * 0.02
        # desired_ys_offset_yaw = phases * y_vel_des_yaw * 0.02
        desired_ys_offset_cmd = phases * y_vel_des_cmd * (0.5 / frequencies)
        desired_ys_offset_yaw = phases * y_vel_des_yaw * (0.5 / frequencies)
        desired_ys_offset_yaw[:, 2:4] *= -1
        desired_ys_offset=desired_ys_offset_cmd+desired_ys_offset_yaw

        # desired_xs_offset_cmd = phases * x_vel_des_cmd * 0.02
        # desired_xs_offset_yaw = phases * x_vel_des_yaw * 0.02
        desired_xs_offset_cmd = phases * x_vel_des_cmd * (0.5 / frequencies)
        desired_xs_offset_yaw = phases * x_vel_des_yaw * (0.5 / frequencies)
        desired_xs_offset_yaw[:,0:1] *= -1
        desired_xs_offset_yaw[:,2:3] *= -1
        desired_xs_offset = desired_xs_offset_cmd + desired_xs_offset_yaw

        desired_ys_nom = desired_ys_nom + desired_ys_offset
        desired_xs_nom = desired_xs_nom + desired_xs_offset

        #最终期望步伐位置的组合，包含X和Y两个方向
        #unsqueeze：在第二维增加一个维度，变为num_envs x 4 x 1（4只脚）
        #相加过后变为num_envs x 4 x 2
        desired_footsteps_body_frame = torch.cat((desired_xs_nom.unsqueeze(2), desired_ys_nom.unsqueeze(2)), dim=2)
        err_raibert_heuristic = torch.abs(desired_footsteps_body_frame - footsteps_in_body_frame[:, :, 0:2])
        reward = torch.sum(torch.square(err_raibert_heuristic), dim=(1, 2))
        return reward



    # def _reward_raibert_heuristic(self):
    #     cur_footsteps_translated = self.env.foot_positions - self.env.base_pos.unsqueeze(1)
    #     footsteps_in_body_frame = torch.zeros(self.env.num_envs, 4, 3, device=self.env.device)
    #     for i in range(4):
    #         footsteps_in_body_frame[:, i, :] = quat_apply_yaw(quat_conjugate(self.env.base_quat),
    #                                                           cur_footsteps_translated[:, i, :])
    #
    #     last_footsteps_translated = self.env.last_foot_positions - self.env.prev_base_pos.unsqueeze(1)
    #     last_footsteps_in_body_frame = torch.zeros(self.env.num_envs, 4, 3, device=self.env.device)
    #     for i in range(4):
    #         last_footsteps_in_body_frame[:, i, :] = quat_apply_yaw(quat_conjugate(self.env.base_quat),
    #                                                           last_footsteps_translated[:, i, :])
    #
    #     desired_stance_width = 0.3
    #     desired_stance_length = 0.38
    #
    #     phases = torch.where(self.env.foot_indices <= 0.5, torch.tensor(0.0,  device='cuda:0'), torch.tensor(1.0, device='cuda:0'))
    #     yaw_vel_des = self.env.commands[:, 2:3]
    #
    #     x_vel_des_cmd = self.env.commands[:, 0:1]
    #     x_vel_des_yaw = yaw_vel_des * desired_stance_width / 2
    #     y_vel_des_cmd = self.env.commands[:, 1:2]
    #     y_vel_des_yaw = yaw_vel_des * desired_stance_length / 2
    #
    #     #任意时可只有两条腿动，两条腿的位移平均分配给四条腿，所以/2
    #     desired_ys_offset_cmd = y_vel_des_cmd * 0.02 / 2
    #     desired_ys_offset_yaw = y_vel_des_yaw * 0.02 / 2
    #     desired_ys_offset_yaw[:, 2:4] *= -1
    #
    #     desired_xs_offset_cmd = x_vel_des_cmd * 0.02 / 2
    #     desired_xs_offset_yaw = x_vel_des_yaw * 0.02 / 2
    #     desired_xs_offset_yaw[:,0:1] *= -1
    #     desired_xs_offset_yaw[:,2:3] *= -1
    #
    #     #x
    #     last_footsteps_in_body_frame[:,0,0]+=desired_xs_offset_cmd[:,0]+desired_xs_offset_yaw[:,0]
    #     last_footsteps_in_body_frame[:,1,0]+=desired_xs_offset_cmd[:,0]+desired_xs_offset_yaw[:,0]
    #     last_footsteps_in_body_frame[:,2,0]+=desired_xs_offset_cmd[:,0]+desired_xs_offset_yaw[:,0]
    #     last_footsteps_in_body_frame[:,3,0]+=desired_xs_offset_cmd[:,0]+desired_xs_offset_yaw[:,0]
    #     #y
    #     last_footsteps_in_body_frame[:,0,1]+=desired_ys_offset_cmd[:,0]+desired_ys_offset_yaw[:,0]
    #     last_footsteps_in_body_frame[:,1,1]+=desired_ys_offset_cmd[:,0]+desired_ys_offset_yaw[:,0]
    #     last_footsteps_in_body_frame[:,2,1]+=desired_ys_offset_cmd[:,0]+desired_ys_offset_yaw[:,0]
    #     last_footsteps_in_body_frame[:,3,1]+=desired_ys_offset_cmd[:,0]+desired_ys_offset_yaw[:,0]
    #
    #     err_raibert_heuristic = torch.abs(last_footsteps_in_body_frame[:,:,0:2] - footsteps_in_body_frame[:, :, 0:2])
    #     reward = torch.sum(torch.square(err_raibert_heuristic), dim=(1, 2))
    #     return reward
