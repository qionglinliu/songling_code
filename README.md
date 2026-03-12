# Piper 机器人数据采集与回放工具

基于 LeRobot 格式的 Piper 机械臂数据采集和回放脚本。

## 文件说明

| 文件 | 功能 |
|------|------|
| `recorder.py` | 数据采集脚本 - 录制机器人演示数据 |
| `replay_v1.py` | 数据回放脚本 - 回放采集的数据 |

## 环境要求

- Python 3.10+
- ROS (已运行 roscore)
- LeRobot 库
- Piper 机械臂 ROS 驱动

## 数据采集 (recorder.py)

### 环境启动


```bash
开三个终端分别对应以下步骤
# 启动 roscore
roscore

# 启动机械臂驱动
bash /home/agilex/cobot_magic/Piper_ros_private-ros-noetic/can_config.sh
roslaunch piper start_ms_piper.launch mode:=0

# 摄像头
roslaunch astra_camera multi_camera.launch
```


### 基本用法

```bash
# 创建新数据集 (左臂模式)
python recorder.py --repo-id my_data --arm left

# 追加数据到现有数据集 (使用 "." 作为 repo-id)
python recorder.py --repo-id . --arm left

# 右臂模式
python recorder.py --repo-id my_data --arm right

# 双臂模式
python recorder.py --repo-id my_data --arm both
```

### 参数说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--repo-id` | 必填 | 数据集名称，使用 `.` 表示直接使用 `--root` 目录 |
| `--root` | `songling_code/data` | 数据保存路径 |
| `--arm` | `left` | 手臂选择: `left` / `right` / `both` |
| `--num-episodes` | `10` | 采集 episode 数量 |
| `--fps` | `30` | 采集帧率 |
| `--task` | `机器人操作演示` | 任务描述 |

### 操作流程

1. 启动 ROS 和机械臂驱动
2. 运行采集命令
3. 按回车键开始录制
4. 演示操作
5. 按回车键停止录制
6. 重复步骤 3-5 直到完成所有 episodes

## 数据回放 (replay_v1.py)

### 环境启动



```bash
# 启动 roscore
roscore

硬件上机械臂需要断掉连接。主从臂
# 启动机械臂驱动
bash /home/agilex/cobot_magic/Piper_ros_private-ros-noetic/can_config.sh
 roslaunch piper start_ms_piper.launch mode:=1

# 摄像头
roslaunch astra_camera multi_camera.launch
```


### 基本用法

```bash
# 回放 songling_code/data 目录下的数据集
python replay_v1.py --repo-id . --episode 0 --arm left

# 回放指定数据集
python replay_v1.py --repo-id my_data --episode 0 --arm left

# 回放所有 episodes
python replay_v1.py --repo-id . --episode -1 --arm left

# 慢速回放 (调试用)
python replay_v1.py --repo-id . --episode 0 --speed 0.5 --arm left

# 循环回放
python replay_v1.py --repo-id . --episode 0 --loop --arm left
```

### 参数说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--repo-id` | 必填 | 数据集名称，使用 `.` 表示直接使用 `--root` 目录 |
| `--root` | `songling_code/data` | 数据集路径 |
| `--arm` | `left` | 手臂选择: `left` / `right` / `both` |
| `--episode` | `0` | Episode 索引，`-1` 表示回放所有 |
| `--fps` | `30` | 回放帧率 |
| `--speed` | `1.0` | 回放速度 (0.5=半速, 2.0=双速) |
| `--no-interpolate` | - | 禁用插值平滑 |
| `--loop` | - | 循环回放 |

## 数据格式

采用 LeRobot 标准格式：

```
data/
├── meta/
│   ├── info.json        # 数据集信息
│   ├── stats.json       # 统计信息
│   └── episodes/        # Episode 元数据
├── data/
│   └── chunk-000/       # Parquet 数据文件
├── images/              # 图像数据
│   ├── observation.images.camera_f/
│   ├── observation.images.camera_l/
│   └── observation.images.camera_r/
└── videos/              # 视频数据
```

### 数据字段

| 字段 | 说明 |
|------|------|
| `observation.state` | Puppet 机械臂关节位置 (当前状态) |
| `action` | 下一帧 Puppet 关节位置 (预测目标) |
| `observation.images.camera_f` | 前置摄像头图像 |
| `observation.images.camera_l` | 左侧摄像头图像 (左臂/双臂模式) |
| `observation.images.camera_r` | 右侧摄像头图像 (右臂/双臂模式) |

## 常见问题

### Q: 如何追加数据到现有数据集？

使用 `--repo-id .` 参数：

```bash
python recorder.py --repo-id . --arm left
```

这会将数据追加到 `songling_code/data/` 目录下的现有数据集。

### Q: `--repo-id .` 和 `--repo-id my_data` 有什么区别？

| 命令 | 数据集路径 | 适用场景 |
|------|-----------|---------|
| `--repo-id .` | `songling_code/data/` | 数据直接在 data/ 目录下 |
| `--repo-id my_data` | `songling_code/data/my_data/` | 数据在 data/my_data/ 子目录下 |

### Q: 回放时机械臂不动？

确保：
1. ROS 节点正常运行
2. 机械臂驱动已启动
3. 检查 ROS 话题是否正确发布：
   ```bash
   rostopic list | grep master
   ```

### Q: 数据集已存在错误？

使用 `--repo-id .` 来追加数据，而不是创建新数据集。

## 相关链接

- [LeRobot 文档](https://github.com/huggingface/lerobot)
- [LeRobot 数据集格式](https://huggingface.co/docs/lerobot/dataset_format)