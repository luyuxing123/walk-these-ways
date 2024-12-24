import os
import glob
import json
import logging

import torch
import numpy as np

from . import pose3d
from . import motion_util


def quat_slerp(q0, q1, t):
    qx, qy, qz, qw = 0, 1, 2, 3

    cos_half_theta = q0[..., qw] * q1[..., qw] \
        + q0[..., qx] * q1[..., qx] \
        + q0[..., qy] * q1[..., qy] \
        + q0[..., qz] * q1[..., qz]

    neg_mask = cos_half_theta < 0
    q1 = q1.clone()
    q1[neg_mask] = -q1[neg_mask]
    cos_half_theta = torch.abs(cos_half_theta)
    cos_half_theta = torch.unsqueeze(cos_half_theta, dim=-1)

    half_theta = torch.acos(cos_half_theta)
    sin_half_theta = torch.sqrt(1.0 - cos_half_theta * cos_half_theta)

    ratioA = torch.sin((1 - t) * half_theta) / sin_half_theta
    ratioB = torch.sin(t * half_theta) / sin_half_theta

    new_q_x = ratioA * q0[..., qx:qx+1] + ratioB * q1[..., qx:qx+1]
    new_q_y = ratioA * q0[..., qy:qy+1] + ratioB * q1[..., qy:qy+1]
    new_q_z = ratioA * q0[..., qz:qz+1] + ratioB * q1[..., qz:qz+1]
    new_q_w = ratioA * q0[..., qw:qw+1] + ratioB * q1[..., qw:qw+1]

    cat_dim = len(new_q_w.shape) - 1
    new_q = torch.cat([new_q_x, new_q_y, new_q_z, new_q_w], dim=cat_dim)

    new_q = torch.where(torch.abs(sin_half_theta) <
                        0.001, 0.5 * q0 + 0.5 * q1, new_q)
    new_q = torch.where(torch.abs(cos_half_theta) >= 1, q0, new_q)

    return new_q


NUM_LEGS = 4
NUM_JOINTS_LEG = 3


class AMPLoader:

    POS_SIZE = 3
    ROT_SIZE = 4
    LINEAR_VEL_SIZE = 3
    ANGULAR_VEL_SIZE = 3
    JOINT_POS_SIZE = NUM_LEGS * NUM_JOINTS_LEG
    JOINT_VEL_SIZE = NUM_LEGS * NUM_JOINTS_LEG
    TAR_TOE_POS_LOCAL_SIZE = NUM_LEGS * 3
    TAR_TOE_VEL_LOCAL_SIZE = NUM_LEGS * 3

    ROOT_POS_START_IDX = 0  # 基座位置
    ROOT_POS_END_IDX = ROOT_POS_START_IDX + POS_SIZE

    ROOT_ROT_START_IDX = ROOT_POS_END_IDX  # 基座姿态
    ROOT_ROT_END_IDX = ROOT_ROT_START_IDX + ROT_SIZE

    LINEAR_VEL_START_IDX = ROOT_ROT_END_IDX  # 基座线速度
    LINEAR_VEL_END_IDX = LINEAR_VEL_START_IDX + LINEAR_VEL_SIZE

    ANGULAR_VEL_START_IDX = LINEAR_VEL_END_IDX  # 基座角速度
    ANGULAR_VEL_END_IDX = ANGULAR_VEL_START_IDX + ANGULAR_VEL_SIZE

    JOINT_POSE_START_IDX = ANGULAR_VEL_END_IDX  # 关节角度
    JOINT_POSE_END_IDX = JOINT_POSE_START_IDX + JOINT_POS_SIZE

    JOINT_VEL_START_IDX = JOINT_POSE_END_IDX  # 关节速度
    JOINT_VEL_END_IDX = JOINT_VEL_START_IDX + JOINT_VEL_SIZE

    JOINT_TOR_START_IDX = JOINT_VEL_END_IDX  # 关节力矩
    JOINT_TOR_END_IDX = JOINT_TOR_START_IDX + JOINT_VEL_SIZE

    # TAR_TOE_POS_LOCAL_START_IDX = JOINT_VEL_END_IDX  # 足端位置
    # TAR_TOE_POS_LOCAL_END_IDX = TAR_TOE_POS_LOCAL_START_IDX + TAR_TOE_POS_LOCAL_SIZE

    # TAR_TOE_VEL_LOCAL_START_IDX = TAR_TOE_POS_LOCAL_END_IDX  # 足端速度
    # TAR_TOE_VEL_LOCAL_END_IDX = TAR_TOE_VEL_LOCAL_START_IDX + TAR_TOE_VEL_LOCAL_SIZE

    # AMP_START = LINEAR_VEL_START_IDX
    # AMP_END = JOINT_VEL_END_IDX
    AMP_START = 0
    AMP_END = 30

    def __init__(
            self,
            device,
            time_between_frames,
            data_dir='',
            preload_transitions=False,
            num_preload_transitions=1000000,
            motion_files=glob.glob('datasets/motion_files2/*'),
    ):
        """Expert dataset provides AMP observations from Dog mocap dataset.

        time_between_frames: Amount of time in seconds between transition.
        """
        self.device = device
        self.time_between_frames = time_between_frames  # 两个相邻状态的时间间隔

        # Values to store for each trajectory.
        self.trajectories = []  # 不包含机身位置与姿态(amp状态)
        self.trajectories_full = []
        self.trajectory_names = []  # 轨迹数据的名称(例如前进,转弯等)
        self.trajectory_idxs = []
        self.trajectory_lens = []  # Traj length in seconds.
        self.trajectory_weights = []
        self.trajectory_frame_durations = []
        self.trajectory_num_frames = []

        for i, motion_file in enumerate(motion_files):

            self.trajectory_names.append(motion_file.split('.')[0])
            with open(motion_file, "r") as f:
                motion_json = json.load(f)  # json格式数据
                motion_data = np.array(motion_json["Frames"])

                # Normalize and standardize quaternions.
                for f_i in range(motion_data.shape[0]):  # 帧数
                    # 对每一个运动帧标准化姿态四元数
                    root_rot = AMPLoader.get_root_rot(motion_data[f_i])
                    root_rot = pose3d.QuaternionNormalize(root_rot)
                    root_rot = motion_util.standardize_quaternion(root_rot)
                    motion_data[
                        f_i,
                        AMPLoader.POS_SIZE:
                            (AMPLoader.POS_SIZE +
                             AMPLoader.ROT_SIZE)] = root_rot  # 修改姿态值

                # Remove first 7 observation dimensions (root_pos and root_orn).
                self.trajectories.append(torch.tensor(
                    motion_data[
                        :,
                        AMPLoader.AMP_START:AMPLoader.AMP_END
                    ], dtype=torch.float, device=device))

                # 完整的运动轨迹
                self.trajectories_full.append(torch.tensor(
                    motion_data[:, :AMPLoader.AMP_END],
                    dtype=torch.float, device=device))

                self.trajectory_idxs.append(i)
                self.trajectory_weights.append(
                    float(motion_json["MotionWeight"]))  # 该轨迹的采样权重

                frame_duration = float(motion_json["FrameDuration"])
                self.trajectory_frame_durations.append(
                    frame_duration)  # 每条轨迹单帧持续时长
                traj_len = (motion_data.shape[0] - 1) * \
                    frame_duration  # 该轨迹运动总时长(s)
                self.trajectory_lens.append(traj_len)
                self.trajectory_num_frames.append(
                    float(motion_data.shape[0]))  # 该轨迹总运动帧数

            print(f"Loaded {traj_len}s. motion from {motion_file}.")

        # Trajectory weights are used to sample some trajectories more than others.
        self.trajectory_weights = np.array(
            self.trajectory_weights) / np.sum(self.trajectory_weights)
        # 专家轨迹时间间隔
        self.trajectory_frame_durations = np.array(
            self.trajectory_frame_durations)
        self.trajectory_lens = np.array(self.trajectory_lens)
        self.trajectory_num_frames = np.array(self.trajectory_num_frames)

        # Preload transitions.
        self.preload_transitions = preload_transitions
        if self.preload_transitions:
            print(f'Preloading {num_preload_transitions} transitions')
            # 按权重随机抽取轨迹序号
            traj_idxs = self.weighted_traj_idx_sample_batch(
                num_preload_transitions)
            # 随机生成各轨迹中的时间点
            times = self.traj_time_sample_batch(traj_idxs)

            # 取出对应时间点的状态转移对
            self.preloaded_s = self.get_full_frame_at_time_batch(
                traj_idxs, times)
            self.preloaded_s_next = self.get_full_frame_at_time_batch(
                traj_idxs, times + self.time_between_frames)
            print(f'Finished preloading')

        self.all_trajectories_full = torch.vstack(self.trajectories_full)

    def weighted_traj_idx_sample(self):
        """Get traj idx via weighted sampling."""
        return np.random.choice(
            self.trajectory_idxs, p=self.trajectory_weights)

    def weighted_traj_idx_sample_batch(self, size):
        """Batch sample traj idxs."""
        return np.random.choice(
            self.trajectory_idxs, size=size, p=self.trajectory_weights,
            replace=True)

    def traj_time_sample(self, traj_idx):
        """Sample random time for traj."""
        subst = self.time_between_frames + \
            self.trajectory_frame_durations[traj_idx]
        return max(
            0, (self.trajectory_lens[traj_idx] * np.random.uniform() - subst))

    def traj_time_sample_batch(self, traj_idxs):
        """Sample random time for multiple trajectories."""
        subst = self.time_between_frames + \
            self.trajectory_frame_durations[traj_idxs]
        time_samples = self.trajectory_lens[traj_idxs] * \
            np.random.uniform(size=len(traj_idxs)) - subst
        return np.maximum(np.zeros_like(time_samples), time_samples)

    def slerp(self, val0, val1, blend):
        return (1.0 - blend) * val0 + blend * val1

    def get_trajectory(self, traj_idx):
        """Returns trajectory of AMP observations."""
        return self.trajectories_full[traj_idx]

    def get_frame_at_time(self, traj_idx, time):
        """Returns frame for the given trajectory at the specified time."""
        p = float(time) / self.trajectory_lens[traj_idx]
        n = self.trajectories[traj_idx].shape[0]
        idx_low, idx_high = int(np.floor(p * n)), int(np.ceil(p * n))
        frame_start = self.trajectories[traj_idx][idx_low]
        frame_end = self.trajectories[traj_idx][idx_high]
        blend = p * n - idx_low
        return self.slerp(frame_start, frame_end, blend)

    def get_frame_at_time_batch(self, traj_idxs, times):
        """Returns frame for the given trajectory at the specified time."""
        p = times / self.trajectory_lens[traj_idxs]
        n = self.trajectory_num_frames[traj_idxs]
        idx_low, idx_high = np.floor(
            p * n).astype(np.int), np.ceil(p * n).astype(np.int)
        # 返回的是部分状态
        all_frame_starts = torch.zeros(len(
            traj_idxs),  AMPLoader.AMP_END - AMPLoader.AMP_START, device=self.device)
        all_frame_ends = torch.zeros(len(
            traj_idxs),  AMPLoader.AMP_END - AMPLoader.AMP_START, device=self.device)
        for traj_idx in set(traj_idxs):
            trajectory = self.trajectories[traj_idx]
            traj_mask = traj_idxs == traj_idx
            all_frame_starts[traj_mask] = trajectory[idx_low[traj_mask]]
            all_frame_ends[traj_mask] = trajectory[idx_high[traj_mask]]
        blend = torch.tensor(p * n - idx_low, device=self.device,
                             dtype=torch.float32).unsqueeze(-1)
        return self.slerp(all_frame_starts, all_frame_ends, blend)

    def get_full_frame_at_time(self, traj_idx, time):
        """Returns full frame for the given trajectory at the specified time."""
        p = float(time) / self.trajectory_lens[traj_idx]
        n = self.trajectories_full[traj_idx].shape[0]
        idx_low, idx_high = int(np.floor(p * n)), int(np.ceil(p * n))
        frame_start = self.trajectories_full[traj_idx][idx_low]
        frame_end = self.trajectories_full[traj_idx][idx_high]
        blend = p * n - idx_low
        return self.blend_frame_pose(frame_start, frame_end, blend)

    def get_full_frame_at_time_batch(self, traj_idxs, times):
        p = times / self.trajectory_lens[traj_idxs]  # 该时间点在总时长中的位置
        n = self.trajectory_num_frames[traj_idxs]
        idx_low, idx_high = np.floor(
            p * n).astype(int), np.ceil(p * n).astype(int)
        all_frame_pos_starts = torch.zeros(
            len(traj_idxs), AMPLoader.POS_SIZE, device=self.device)
        all_frame_pos_ends = torch.zeros(
            len(traj_idxs), AMPLoader.POS_SIZE, device=self.device)
        all_frame_rot_starts = torch.zeros(
            len(traj_idxs), AMPLoader.ROT_SIZE, device=self.device)
        all_frame_rot_ends = torch.zeros(
            len(traj_idxs), AMPLoader.ROT_SIZE, device=self.device)
        all_frame_amp_starts = torch.zeros(len(
            traj_idxs), AMPLoader.AMP_END - AMPLoader.AMP_START, device=self.device)
        all_frame_amp_ends = torch.zeros(len(
            traj_idxs),  AMPLoader.AMP_END - AMPLoader.AMP_START, device=self.device)

        for traj_idx in set(traj_idxs):
            trajectory = self.trajectories_full[traj_idx]
            traj_mask = traj_idxs == traj_idx  # 生成布尔数组
            all_frame_pos_starts[traj_mask] = AMPLoader.get_root_pos_batch(
                trajectory[idx_low[traj_mask]])
            all_frame_pos_ends[traj_mask] = AMPLoader.get_root_pos_batch(
                trajectory[idx_high[traj_mask]])
            all_frame_rot_starts[traj_mask] = AMPLoader.get_root_rot_batch(
                trajectory[idx_low[traj_mask]])
            all_frame_rot_ends[traj_mask] = AMPLoader.get_root_rot_batch(
                trajectory[idx_high[traj_mask]])
            all_frame_amp_starts[traj_mask] = trajectory[idx_low[traj_mask]
                                                         ][:, AMPLoader.AMP_START:AMPLoader.AMP_END]
            all_frame_amp_ends[traj_mask] = trajectory[idx_high[traj_mask]
                                                       ][:, AMPLoader.AMP_START:AMPLoader.AMP_END]
        blend = torch.tensor(p * n - idx_low, device=self.device,
                             dtype=torch.float32).unsqueeze(-1)
        # 状态插值
        pos_blend = self.slerp(all_frame_pos_starts, all_frame_pos_ends, blend)
        rot_blend = quat_slerp(all_frame_rot_starts, all_frame_rot_ends, blend)
        amp_blend = self.slerp(all_frame_amp_starts, all_frame_amp_ends, blend)
        return torch.cat([pos_blend, rot_blend, amp_blend], dim=-1)

    def get_frame(self):
        """Returns random frame."""
        traj_idx = self.weighted_traj_idx_sample()
        sampled_time = self.traj_time_sample(traj_idx)
        return self.get_frame_at_time(traj_idx, sampled_time)

    def get_full_frame(self):
        """Returns random full frame."""
        traj_idx = self.weighted_traj_idx_sample()
        sampled_time = self.traj_time_sample(traj_idx)
        return self.get_full_frame_at_time(traj_idx, sampled_time)

    def get_full_frame_batch(self, num_frames):
        if self.preload_transitions:
            idxs = np.random.choice(
                self.preloaded_s.shape[0], size=num_frames)
            return self.preloaded_s[idxs]
        else:
            traj_idxs = self.weighted_traj_idx_sample_batch(num_frames)
            times = self.traj_time_sample_batch(traj_idxs)
            return self.get_full_frame_at_time_batch(traj_idxs, times)

    def blend_frame_pose(self, frame0, frame1, blend):
        """Linearly interpolate between two frames, including orientation.

        Args:
            frame0: First frame to be blended corresponds to (blend = 0).
            frame1: Second frame to be blended corresponds to (blend = 1).
            blend: Float between [0, 1], specifying the interpolation between
            the two frames.
        Returns:
            An interpolation of the two frames.
        """

        root_pos0, root_pos1 = AMPLoader.get_root_pos(
            frame0), AMPLoader.get_root_pos(frame1)
        root_rot0, root_rot1 = AMPLoader.get_root_rot(
            frame0), AMPLoader.get_root_rot(frame1)
        amp0, amp1 = frame0[AMPLoader.AMP_START:
                            AMPLoader.AMP_END], frame1[AMPLoader.AMP_START:AMPLoader.AMP_END]

        blend_root_pos = self.slerp(root_pos0, root_pos1, blend)
        blend_root_rot = quat_slerp(root_rot0, root_rot1, blend)
        amp_blend = self.slerp(amp0, amp1, blend)

        return torch.cat([blend_root_pos, blend_root_rot, amp_blend])

    def feed_forward_generator(self, num_mini_batch, mini_batch_size):
        """Generates a batch of AMP transitions."""
        for _ in range(num_mini_batch):
            if self.preload_transitions:
                # 随机生成批量状态抽取序号
                idxs = np.random.choice(self.preloaded_s.shape[0], size=mini_batch_size)
                s = self.preloaded_s[idxs,AMPLoader.AMP_START:AMPLoader.AMP_END]
                s_next = self.preloaded_s_next[idxs,AMPLoader.AMP_START:AMPLoader.AMP_END]

                # 附加一个机身高度信息
                # s = torch.cat([s,self.preloaded_s[idxs, AMPLoader.ROOT_POS_START_IDX + 2:AMPLoader.ROOT_POS_START_IDX + 3]], dim=-1)
                # s_next = torch.cat([s_next,self.preloaded_s_next[idxs, AMPLoader.ROOT_POS_START_IDX + 2:AMPLoader.ROOT_POS_START_IDX + 3]], dim=-1)
            else:
                s, s_next = [], []
                traj_idxs = self.weighted_traj_idx_sample_batch(mini_batch_size)
                times = self.traj_time_sample_batch(traj_idxs)
                for traj_idx, frame_time in zip(traj_idxs, times):
                    s.append(self.get_frame_at_time(traj_idx, frame_time))
                    s_next.append(self.get_frame_at_time(traj_idx, frame_time + self.time_between_frames))

                s = torch.vstack(s)
                s_next = torch.vstack(s_next)
            yield s, s_next

    @property
    def observation_dim(self):
        """Size of AMP observations."""
        return self.trajectories[0].shape[1] #+ 1  # 多加一个机身高度

    @property
    def num_motions(self):
        return len(self.trajectory_names)

    def get_root_pos(pose):
        return pose[AMPLoader.ROOT_POS_START_IDX:AMPLoader.ROOT_POS_END_IDX]

    def get_root_pos_batch(poses):
        return poses[:, AMPLoader.ROOT_POS_START_IDX:AMPLoader.ROOT_POS_END_IDX]

    def get_root_rot(pose):
        return pose[AMPLoader.ROOT_ROT_START_IDX:AMPLoader.ROOT_ROT_END_IDX]

    def get_root_rot_batch(poses):
        return poses[:, AMPLoader.ROOT_ROT_START_IDX:AMPLoader.ROOT_ROT_END_IDX]

    def get_joint_pose(pose):
        return pose[AMPLoader.JOINT_POSE_START_IDX:AMPLoader.JOINT_POSE_END_IDX]

    def get_joint_pose_batch(poses):
        return poses[:, AMPLoader.JOINT_POSE_START_IDX:AMPLoader.JOINT_POSE_END_IDX]

    def get_tar_toe_pos_local(pose):
        return pose[AMPLoader.TAR_TOE_POS_LOCAL_START_IDX:AMPLoader.TAR_TOE_POS_LOCAL_END_IDX]

    def get_tar_toe_pos_local_batch(poses):
        return poses[:, AMPLoader.TAR_TOE_POS_LOCAL_START_IDX:AMPLoader.TAR_TOE_POS_LOCAL_END_IDX]

    def get_linear_vel(pose):
        return pose[AMPLoader.LINEAR_VEL_START_IDX:AMPLoader.LINEAR_VEL_END_IDX]

    def get_linear_vel_batch(poses):
        return poses[:, AMPLoader.LINEAR_VEL_START_IDX:AMPLoader.LINEAR_VEL_END_IDX]

    def get_angular_vel(pose):
        return pose[AMPLoader.ANGULAR_VEL_START_IDX:AMPLoader.ANGULAR_VEL_END_IDX]

    def get_angular_vel_batch(poses):
        return poses[:, AMPLoader.ANGULAR_VEL_START_IDX:AMPLoader.ANGULAR_VEL_END_IDX]

    def get_joint_vel(pose):
        return pose[AMPLoader.JOINT_VEL_START_IDX:AMPLoader.JOINT_VEL_END_IDX]

    def get_joint_vel_batch(poses):
        return poses[:, AMPLoader.JOINT_VEL_START_IDX:AMPLoader.JOINT_VEL_END_IDX]

    def get_tar_toe_vel_local(pose):
        return pose[AMPLoader.TAR_TOE_VEL_LOCAL_START_IDX:AMPLoader.TAR_TOE_VEL_LOCAL_END_IDX]

    def get_tar_toe_vel_local_batch(poses):
        return poses[:, AMPLoader.TAR_TOE_VEL_LOCAL_START_IDX:AMPLoader.TAR_TOE_VEL_LOCAL_END_IDX]
