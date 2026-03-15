# Piper 机器人数据采集与推理工具

基于 LeRobot 格式的 Piper 机械臂数据采集、回放和推理脚本。

## 文件说明

| 文件 | 功能 |
|------|------|
| `recorder_v1.py` | 数据采集脚本 - 录制机器人演示数据 |
| `replay_v1.py` | 数据回放脚本 - 回放采集的数据 |
| `inference_simple.py` | 推理脚本 - 简化版 (每帧推理, EMA平滑) |

## 目录结构

```
songling_code/
├── recorder_v1.py      # 数据采集
├── replay_v1.py        # 数据回放
├── inference_simple.py # 模型推理
├── data/               # 数据集目录
├── outputs/            # 训练输出
```

## 环境要求

- Python 3.10+
- ROS 
- LeRobot 库
- Piper 机械臂 ROS 驱动

---

## 数据采集 (recorder_v1.py)

### 环境启动

```bash
# 终端1: 启动 roscore
roscore

# 终端2: 启动机械臂驱动
bash /home/agilex/cobot_magic/Piper_ros_private-ros-noetic/can_config.sh
roslaunch piper start_ms_piper.launch mode:=0

# 终端3: 启动摄像头
roslaunch astra_camera multi_camera.launch
```

### 基本用法

```bash
# 左臂采集
python recorder_v1.py --repo-id my_data --arm left

# 右臂采集
python recorder_v1.py --repo-id my_data --arm right

# 双臂采集
python recorder_v1.py --repo-id my_data --arm both
```

---

## 数据回放 (replay_v1.py)

### 环境启动

```bash
# 终端1: 启动 roscore
roscore

# 终端2: 启动机械臂驱动 (mode:=1)
bash /home/agilex/cobot_magic/Piper_ros_private-ros-noetic/can_config.sh
roslaunch piper start_ms_piper.launch mode:=1
```

### 基本用法

```bash
# 回放指定 episode
python replay_v1.py --repo-id my_data --episode 0 --arm left

# 回放所有 episodes
python replay_v1.py --repo-id my_data --episode -1 --arm left

# 循环回放
python replay_v1.py --repo-id my_data --episode 0 --loop --arm left
```

---

## 模型推理 (inference_simple.py)

适用于 `n_action_steps=1` 的训练配置，每帧推理，只取第一个动作，使用 EMA 平滑。

### 关节限位 (URDF物理限位)

| 关节 | 类型 | 下限 | 上限 |
|------|------|------|------|
| joint1 | revolute | -2.618 | 2.618 |
| joint2 | revolute | 0 | 3.14 |
| joint3 | revolute | -2.967 | 0 |
| joint4 | revolute | -1.832 | 1.832 |
| joint5 | revolute | -1.22 | 1.22 |
| joint6 | revolute | -3.14 | 3.14 |
| joint7 (夹爪) | prismatic | 0 | 0.076 |

### 基本用法

```bash
# 左臂推理
python inference_simple.py --checkpoint outputs/train/piper_act/checkpoints/050000/pretrained_model --arm left

# 右臂推理
python inference_simple.py --checkpoint outputs/train/piper_act/checkpoints/050000/pretrained_model --arm right

# 双臂推理
python inference_simple.py --checkpoint outputs/train/piper_act/checkpoints/050000/pretrained_model --arm both
```

### 参数说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--checkpoint` | 必填 | 模型checkpoint路径 |
| `--arm` | `left` | 手臂选择: `left` / `right` / `both` |
| `--device` | `cuda` | 计算设备 |
| `--fps` | `30` | 控制频率 |
| `--duration` | `300.0` | 运行时长(秒) |
| `--smoothing` | `0.3` | EMA平滑系数 (0=完全平滑, 1=不平滑) |
| `--max-change` | `0.08` | 每帧最大动作变化量 (rad) |



## 相关链接

- [LeRobot 文档](https://github.com/huggingface/lerobot)
- [LeRobot 数据集格式](https://huggingface.co/docs/lerobot/dataset_format)