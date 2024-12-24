import rospy
from motorstate import MotorState  # dof vel pos
from gazebo_msgs.msg import ModelStates        # base vel
from datetime import datetime
import json

#base lin:5000,base ang:1000,dof:2000

# 全局变量存储最新消息
fl_hip_motor_state = None
fl_thigh_motor_state = None
fl_calf_motor_state = None

fr_hip_motor_state = None
fr_thigh_motor_state = None
fr_calf_motor_state = None

rl_hip_motor_state = None
rl_thigh_motor_state = None
rl_calf_motor_state = None

rr_hip_motor_state = None
rr_thigh_motor_state = None
rr_calf_motor_state = None

base_lin = None
base_angular = None
collected_data = {
    "gaitParam": [0.5, 0.0, 0],
    "FrameDuration": 0.02,
    "MotionWeight": 1,
    "Frames": []
}

# 话题的回调函数
def fl_hip_motor_state_callback(data):
    global fl_hip_motor_state
    fl_hip_motor_state = data
def fl_thigh_motor_state_callback(data):
    global fl_thigh_motor_state
    fl_thigh_motor_state = data
def fl_calf_motor_state_callback(data):
    global fl_calf_motor_state
    fl_calf_motor_state = data

def fr_hip_motor_state_callback(data):
    global fr_hip_motor_state
    fr_hip_motor_state = data
def fr_thigh_motor_state_callback(data):
    global fr_thigh_motor_state
    fr_thigh_motor_state = data
def fr_calf_motor_state_callback(data):
    global fr_calf_motor_state
    fr_calf_motor_state = data

def rl_hip_motor_state_callback(data):
    global rl_hip_motor_state
    rl_hip_motor_state = data
def rl_thigh_motor_state_callback(data):
    global rl_thigh_motor_state
    rl_thigh_motor_state = data
def rl_calf_motor_state_callback(data):
    global rl_calf_motor_state
    rl_calf_motor_state = data

def rr_thigh_motor_state_callback(data):
    global rr_thigh_motor_state
    rr_thigh_motor_state = data
def rr_hip_motor_state_callback(data):
    global rr_hip_motor_state
    rr_hip_motor_state = data
def rr_calf_motor_state_callback(data):
    global rr_calf_motor_state
    rr_calf_motor_state = data

def model_states_callback(data):
    global base_lin
    global base_angular
    base_lin = data.twist[2].linear
    base_angular = data.twist[2].angular


# 定时器回调函数，每50Hz记录一次数据
def collect_trajectory_data(event):
    global fl_hip_motor_state,fl_thigh_motor_state,fl_calf_motor_state,\
        fr_hip_motor_state, fr_thigh_motor_state,fr_calf_motor_state,\
        rl_hip_motor_state,rl_thigh_motor_state,rl_calf_motor_state,\
        rr_hip_motor_state, rl_thigh_motor_state,rl_calf_motor_state,\
        base_lin, base_angular, collected_data

    # 检查所有数据是否已接收到
    if (fl_hip_motor_state and fl_calf_motor_state and fl_thigh_motor_state and
        fr_hip_motor_state and fr_calf_motor_state and fr_thigh_motor_state and
        rl_hip_motor_state and rl_calf_motor_state and rl_thigh_motor_state and
        rr_hip_motor_state and rr_calf_motor_state and rr_thigh_motor_state and
        base_lin and base_angular):
        # 根据需要提取和格式化数据
        frame_data = [
            base_lin.x,
            base_lin.y,
            base_lin.z,
            base_angular.x,
            base_angular.y,
            base_angular.z,
            fl_hip_motor_state.q,fl_thigh_motor_state.q,fl_calf_motor_state.q,
            fr_hip_motor_state.q,fr_thigh_motor_state.q,fr_calf_motor_state.q,
            rl_hip_motor_state.q,rl_thigh_motor_state.q,rl_calf_motor_state.q,
            rr_hip_motor_state.q,rr_thigh_motor_state.q,rr_calf_motor_state.q,
            fl_hip_motor_state.dq, fl_thigh_motor_state.dq, fl_calf_motor_state.dq,
            fr_hip_motor_state.dq, fr_thigh_motor_state.dq, fr_calf_motor_state.dq,
            rl_hip_motor_state.dq, rl_thigh_motor_state.dq, rl_calf_motor_state.dq,
            rr_hip_motor_state.dq, rr_thigh_motor_state.dq, rr_calf_motor_state.dq,

        ]
        # 将frame_data添加到Frames中
        collected_data["Frames"].append(frame_data)

def listener():
    # 初始化ROS节点
    rospy.init_node('robot_listener', anonymous=True)

    # 订阅各个topic
    rospy.Subscriber('/go1_gazebo/FL_hip_controller/state', MotorState, fl_hip_motor_state_callback)
    rospy.Subscriber('/go1_gazebo/FL_thigh_controller/state', MotorState, fl_thigh_motor_state_callback)
    rospy.Subscriber('/go1_gazebo/FL_calf_controller/state', MotorState, fl_calf_motor_state_callback)

    rospy.Subscriber('/go1_gazebo/FR_hip_controller/state', MotorState, fr_hip_motor_state_callback)
    rospy.Subscriber('/go1_gazebo/FR_thigh_controller/state', MotorState, fr_thigh_motor_state_callback)
    rospy.Subscriber('/go1_gazebo/FR_calf_controller/state', MotorState, fr_calf_motor_state_callback)

    rospy.Subscriber('/go1_gazebo/RL_hip_controller/state', MotorState, rl_hip_motor_state_callback)
    rospy.Subscriber('/go1_gazebo/RL_thigh_controller/state', MotorState, rl_thigh_motor_state_callback)
    rospy.Subscriber('/go1_gazebo/RL_calf_controller/state', MotorState, rl_calf_motor_state_callback)

    rospy.Subscriber('/go1_gazebo/RR_hip_controller/state', MotorState, rr_hip_motor_state_callback)
    rospy.Subscriber('/go1_gazebo/RR_thigh_controller/state', MotorState, rr_thigh_motor_state_callback)
    rospy.Subscriber('/go1_gazebo/RR_calf_controller/state', MotorState, rr_calf_motor_state_callback)

    rospy.Subscriber('/gazebo/model_states', ModelStates, model_states_callback)

    # 设置定时器，每隔0.02秒（50Hz）调用数据收集函数
    rospy.Timer(rospy.Duration(0.02), collect_trajectory_data)

    # 设置30秒后停止收集数据并保存
    rospy.sleep(30)
    save_data_to_json(collected_data, "../datasets/turnleft.json")

def save_data_to_json(data, filename):
    # 保存数据到json文件
    with open(filename, 'w') as json_file:
        json.dump(data, json_file, indent=4)
    rospy.loginfo(f"Data saved to {filename}")

if __name__ == '__main__':
    listener()
