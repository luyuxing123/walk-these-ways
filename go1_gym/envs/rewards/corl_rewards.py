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
    #     # Tracking of linear velocity commands (xy axes)
         lin_vel_error = torch.sum(torch.square(self.env.commands[:, :2] - self.env.base_lin_vel[:, :2]), dim=1)
         return torch.exp(-lin_vel_error / self.env.cfg.rewards.tracking_sigma)

    def _reward_tracking_ang_vel(self):
    #     # Tracking of angular velocity commands (yaw)
         ang_vel_error = torch.square(self.env.commands[:, 2] - self.env.base_ang_vel[:, 2])
         return torch.exp(-ang_vel_error / self.env.cfg.rewards.tracking_sigma_yaw)

    def _reward_lin_vel_z(self):
    #     # Penalize z axis base linear velocity
         return torch.square(self.env.base_lin_vel[:, 2])
    #
    def _reward_ang_vel_xy(self):
    #     # Penalize xy axes base angular velocity
         return torch.sum(torch.square(self.env.base_ang_vel[:, :2]), dim=1)
    #
    def _reward_orientation(self):
    #     # Penalize non flat base orientation
         return torch.sum(torch.square(self.env.projected_gravity[:, :2]), dim=1)
    #
    def _reward_torques(self):
    #     # Penalize torques
         return torch.sum(torch.square(self.env.torques), dim=1)
    #
    def _reward_dof_acc(self):
    #     # Penalize dof accelerations
         return torch.sum(torch.square((self.env.last_dof_vel - self.env.dof_vel) / self.env.dt), dim=1)
    #
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
    #
    def _reward_jump(self):
         #绝对身体高度
         body_height = self.env.base_pos[:, 2]
         jump_height_target = self.env.commands[:, 3] + self.env.cfg.rewards.base_height_target
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
         force = torch.norm(self.env.contact_forces[:, self.env.feet_indices, :],dim=-1)
         reward = torch.sum((force - self.env.cfg.rewards.max_contact_force).clip(min=0.), dim=1)
         return reward

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



    def _reward_orientation_control(self):
    #     # Penalize non flat base orientation
         roll_pitch_commands = self.env.commands[:, 10:12]
         quat_roll = quat_from_angle_axis(-roll_pitch_commands[:, 1],
                                          torch.tensor([1, 0, 0], device=self.env.device, dtype=torch.float))
         quat_pitch = quat_from_angle_axis(-roll_pitch_commands[:, 0],
                                           torch.tensor([0, 1, 0], device=self.env.device, dtype=torch.float))

         desired_base_quat = quat_mul(quat_roll, quat_pitch)
         desired_projected_gravity = quat_rotate_inverse(desired_base_quat, self.env.gravity_vec)

         return torch.sum(torch.square(self.env.projected_gravity[:, :2] - desired_projected_gravity[:, :2]), dim=1)
    #

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


    def _reward_feet_stumble(self):
        # Penalize feet hitting vertical surfaces
        #x，y方向合力大于2且z方向力小于1，惩罚脚触及垂直表面
        stumble = (torch.norm(self.env.contact_forces[:, self.env.feet_indices, :2], dim=2) > 2.)*\
                  (torch.abs(self.env.contact_forces[:, self.env.feet_indices, 2]) < 1.)
        return torch.sum(stumble, dim=1)

    def _reward_base_height(self):
        # Penalize base height away from target
        base_height = torch.mean(self.env.root_states[:, 2].unsqueeze(1) - self.env.measured_heights, dim=1)
        error = torch.abs(self.env.cfg.rewards.base_height_target - base_height)
        rew = 1 - torch.exp(-1 * error)
        return rew

    def _reward_trap_static(self):
        lin_trap_static = (torch.norm(self.env.base_lin_vel[:, :2], dim=-1) < self.env.cfg.commands.min_vel * 0.5)*(
            torch.norm(self.env.commands[:, :2], dim=-1) > self.env.cfg.commands.min_vel)
        ang_trap_static = (torch.abs(self.env.base_ang_vel[:, 2]) < self.env.cfg.commands.min_vel * 0.5)*(
            torch.abs(self.env.commands[:, 2]) > self.env.cfg.commands.min_vel)
        trap_mask = torch.logical_or(lin_trap_static, ang_trap_static)
        self.env.trap_static_time[trap_mask] += self.env.dt
        self.env.trap_static_time[~trap_mask] = 0.
        return self.env.trap_static_time.clip(max=5.)


    #wjz
    # def _reward_large_orientation(self):
    #     # Penalize large base orientation
    #     angle = torch.atan2(torch.norm(self.env.projected_gravity[:, :2], dim=1), -self.env.projected_gravity[:, 2])
    #     thresh = torch.ones_like(angle)*30*torch.pi/180.
    #     # thresh[zero_cmd_mask] = 0.1
    #     return (angle - thresh).clip(min=0., max=0.2)



    # def _reward_action_rate(self):
    #     # Penalize changes in actions
    #     diff_1 = torch.sum(torch.square(self.env.actions - self.env.last_actions), dim=1)
    #     diff_2 = torch.sum(torch.square(self.env.actions - 2 * self.env.last_actions + self.env.last_last_actions), dim=1)
    #     return diff_1 + diff_2
    #
    # def _reward_collision(self):
    #     # Penalize collisions on selected bodies
    #     penalize = torch.sum(1.*(torch.norm(self.env.contact_forces[:, self.env.penalised_contact_indices, :], dim=-1) > 0.1), dim=1)
    #     base_collision = torch.norm(self.env.contact_forces[:, self.env.penalised_contact_indices[0], :], dim=-1) > 0.1
    #     penalize[base_collision] += 2
    #     return penalize
    #
    # def _reward_termination(self):
    #     # Terminal reward / penalty
    #     return self.env.reset_buf * ~self.env.time_out_buf
    #
    # def _reward_dof_pos_limits(self):
    #     # Penalize dof positions too close to the limit
    #     out_of_limits = - \
    #         (self.env.dof_pos -
    #          self.env.dof_pos_limits[:, 0]).clip(max=0.)  # lower limit
    #     out_of_limits += (self.env.dof_pos -
    #                       self.env.dof_pos_limits[:, 1]).clip(min=0.)
    #     return torch.sum(out_of_limits, dim=1)

    def _reward_dof_vel_limits(self):
        # Penalize dof velocities too close to the limit
        dof_vel_limits = torch.clip(10*self.env.v_level.unsqueeze(-1).repeat(1,self.env.num_dof), min=10, max=22)
        error = torch.sum((torch.abs(self.env.dof_vel) - dof_vel_limits).clip(min=0., max=15.), dim=1)
        rew = 1 - torch.exp(-1 * error)
        return rew
        # return torch.sum(
        #     (torch.abs(self.env.dof_vel) - self.env.dof_vel_limits * self.env.cfg.rewards.soft_dof_vel_limit).clip(min=0., max=1.),
        #     dim=1)

    # def _reward_torque_limits(self):
        # penalize torques too close to the limit
        return torch.sum((torch.abs(self.env.torques) - self.env.torque_limits*self.env.cfg.rewards.soft_torque_limit).clip(min=0., max=1.), dim=1)

    # def _reward_tracking_lin_vel(self):
    #     # Tracking of linear velocity commands (xy axes)
    #     lin_vel_error = torch.sum(torch.square(self.env.commands[:, :2] - self.env.base_lin_vel[:, :2]), dim=1)
    #     return torch.exp(-lin_vel_error/self.env.cfg.rewards.tracking_sigma)
    #
    # def _reward_tracking_ang_vel(self):
    #     # Tracking of angular velocity commands (yaw)
    #     ang_vel_error = torch.square(self.env.commands[:, 2] - self.env.base_ang_vel[:, 2])
    #     return torch.exp(-ang_vel_error/self.env.cfg.rewards.tracking_sigma)

    def _reward_feet_air_time(self):
        # Reward long steps
        first_contact = (self.env.feet_air_time > 0.) * self.env.bool_foot_contact
        self.env.feet_air_time += self.env.dt
        # reward only on first contact with the ground
        des_feet_air_time = 0.5/self.env.v_level.unsqueeze(-1).repeat(1,self.env.feet_indices.shape[0])
        # des_feet_air_time = 0.2
        rew_airTime = torch.sum(torch.clamp((self.env.feet_air_time - des_feet_air_time), max=0.) * first_contact, dim=1)
        self.env.feet_air_time *= ~self.env.bool_foot_contact
        cmd_mask = torch.logical_or(torch.norm(self.env.commands[:, :2], dim=1) > self.env.cfg.commands.min_vel,
        torch.abs(self.env.commands[:, 2]) > self.env.cfg.commands.min_vel)
        rew_airTime[~cmd_mask] = -torch.sum(self.env.feet_air_time[~cmd_mask], dim=1) # reward stand still for zero command
        return rew_airTime

    def _reward_feet_slip(self):
        # penalize feet_slip
        penalize = torch.zeros(self.env.num_envs, self.env.feet_indices.shape[0], device=self.env.device, requires_grad=False)
        contact_vel = self.env.foot_vel_world[self.env.bool_foot_contact]
        # penalize x/y/z vel
        penalize[self.env.bool_foot_contact] = torch.norm(contact_vel, dim=-1)
        rew = 1 - torch.exp(-1 * torch.sum(penalize, dim=-1))
        return rew


    # def _reward_feet_contact_forces(self):
    #     # penalize high contact forces
    #     # print('contact feet forces:', self.env.contact_forces[:, self.env.feet_indices, :])
    #     return torch.sum((torch.norm(self.env.contact_forces[:, self.env.feet_indices, :], dim=-1) - self.env.cfg.rewards.max_contact_force).clip(min=0.), dim=1)
    #
    # def _reward_foot_stance_ori(self):
    #     penalize = torch.zeros(self.env.num_envs, self.env.feet_indices.shape[0], device=self.env.device, requires_grad=False)
    #     for i in range(self.env.feet_indices.shape[0]):
    #         _, pitch, _ = get_euler_xyz(self.env.foot_quat[:,i,:])
    #         # if i == 3:
    #         #     print(f"pitch_{i}: ",pitch)
    #         p_thresh = -15*torch.ones(self.env.num_envs, device=self.env.device, requires_grad=False)
    #         p_thresh[self.env.high_track_mode] = 10
    #         penalize[:,i] = torch.clip(pitch - p_thresh * torch.pi/180., min=0, max=1)
    #     penalize[~self.env.bool_foot_contact]=0
    #     return torch.sum(penalize, dim=-1)