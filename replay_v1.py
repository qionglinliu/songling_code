#!/usr/bin/env python3
"""
Piper数据回放脚本 v1 - LeRobot格式
参考松灵 replay_data.py 设计，夹爪特殊处理

优化: 直接读取 parquet 文件，避免视频解码卡顿

使用方法:
  python replay_v1.py --repo-id 3-12-banana --episode 0 --arm left
  python replay_v1.py --repo-id . --episode 0 --arm left
"""

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
import rospy
from std_msgs.msg import Header
from sensor_msgs.msg import Image, JointState


class PiperReplayV1:
    """Piper数据回放器 - 直接读取parquet，避免视频解码"""
    
    def __init__(
        self,
        repo_id: str,
        root: str | Path,
        arm: str = "left",
        fps: int = 30,
    ):
        self.repo_id = repo_id
        self.root = Path(root)
        self.arm = arm
        self.fps = fps
        
        # 手臂配置
        self.is_left_arm = arm in ["left", "both"]
        self.is_right_arm = arm in ["right", "both"]
        self.is_dual_arm = arm == "both"
        
        # ROS
        self.joint_left_pub = None
        self.joint_right_pub = None
        
        # 数据
        self.parquet_data = None
        self.episodes_info = None
        self._actions_cache = {}  # 缓存处理好的 actions
        
        # 夹爪初始位置 (参考松灵)
        self.origin_left = [-0.0057, -0.031, -0.0122, -0.032, 0.0099, 0.0179, 0.2279]
        self.origin_right = [0.0616, 0.0021, 0.0475, -0.1013, 0.1097, 0.0872, 0.2279]
    
    def _init_ros(self):
        """初始化ROS发布者"""
        rospy.init_node('piper_replay_v1', anonymous=True)
        
        # 关节状态发布者
        if self.is_left_arm or self.is_dual_arm:
            self.joint_left_pub = rospy.Publisher(
                '/master/joint_left', JointState, queue_size=10
            )
        if self.is_right_arm or self.is_dual_arm:
            self.joint_right_pub = rospy.Publisher(
                '/master/joint_right', JointState, queue_size=10
            )
        
        arm_type = "双臂" if self.is_dual_arm else ("左臂" if self.is_left_arm else "右臂")
        rospy.loginfo(f"ROS发布者已初始化 - {arm_type}模式")
    
    def _load_dataset(self):
        """加载数据集 - 直接读取parquet文件，避免视频解码"""
        if self.repo_id == ".":
            dataset_path = self.root
        else:
            dataset_path = self.root / self.repo_id
            
        print(f"加载数据集: {dataset_path}")
        
        if not dataset_path.exists():
            raise FileNotFoundError(f"数据集不存在: {dataset_path}")
        
        # 读取 parquet 数据文件
        data_dir = dataset_path / "data" / "chunk-000"
        if not data_dir.exists():
            raise FileNotFoundError(f"数据目录不存在: {data_dir}")
        
        # 合并所有 parquet 文件
        parquet_files = sorted(data_dir.glob("*.parquet"))
        if not parquet_files:
            raise FileNotFoundError(f"未找到 parquet 文件: {data_dir}")
        
        dfs = []
        for pf in parquet_files:
            df = pd.read_parquet(pf)
            dfs.append(df)
        
        self.parquet_data = pd.concat(dfs, ignore_index=True)
        
        # 读取 episode 元数据
        episodes_dir = dataset_path / "meta" / "episodes" / "chunk-000"
        if episodes_dir.exists():
            ep_files = sorted(episodes_dir.glob("*.parquet"))
            ep_dfs = [pd.read_parquet(f) for f in ep_files]
            self.episodes_info = pd.concat(ep_dfs, ignore_index=True)
        else:
            # 从数据中推断 episode 边界
            self.episodes_info = None
        
        # 统计 episodes
        if 'episode_index' in self.parquet_data.columns:
            num_episodes = self.parquet_data['episode_index'].nunique()
        else:
            num_episodes = 1
        
        print(f"数据集信息:")
        print(f"  总帧数: {len(self.parquet_data)}")
        print(f"  Episodes数量: {num_episodes}")
        print(f"  FPS: {self.fps}")
    
    def _create_joint_msg(self, positions: list) -> JointState:
        """创建关节状态消息"""
        msg = JointState()
        msg.header = Header()
        msg.header.stamp = rospy.Time.now()
        msg.name = ['joint0', 'joint1', 'joint2', 'joint3', 'joint4', 'joint5', 'joint6']
        msg.position = positions
        msg.velocity = [0.0] * 7
        msg.effort = [0.0] * 7
        return msg
    
    def _get_episode_actions(self, episode_idx: int) -> np.ndarray:
        """获取指定episode的所有action数据 (带缓存)"""
        # 检查缓存
        if episode_idx in self._actions_cache:
            return self._actions_cache[episode_idx]
        
        if 'episode_index' in self.parquet_data.columns:
            episode_data = self.parquet_data[
                self.parquet_data['episode_index'] == episode_idx
            ]
        else:
            episode_data = self.parquet_data
        
        # 按 frame_index 排序
        if 'frame_index' in episode_data.columns:
            episode_data = episode_data.sort_values('frame_index')
        
        # 获取 action 列 (可能是 numpy 数组或分开的列)
        if 'action' in episode_data.columns:
            # action 列直接包含 numpy 数组
            actions = np.stack(episode_data['action'].values)
        else:
            # 尝试 action.0, action.1 格式
            action_cols = [col for col in episode_data.columns if col.startswith('action.')]
            if action_cols:
                actions = episode_data[action_cols].values
            else:
                raise ValueError("未找到 action 数据列")
        
        # 缓存结果
        self._actions_cache[episode_idx] = actions
        return actions
    
    def _precompute_interpolated(self, actions: np.ndarray, interpolate_steps: int) -> np.ndarray:
        """预先计算所有插值后的动作 (夹爪不插值!)"""
        # 初始位置
        if self.is_dual_arm:
            last_action = np.array(self.origin_left + self.origin_right, dtype=np.float32)
        elif self.is_left_arm:
            last_action = np.array(self.origin_left, dtype=np.float32)
        else:
            last_action = np.array(self.origin_right, dtype=np.float32)
        
        all_interpolated = []
        for action in actions:
            interpolated = np.linspace(last_action, action, interpolate_steps)
            
            # 夹爪不插值! 直接使用目标值 (参考松灵做法)
            if self.is_dual_arm:
                interpolated[:, 6] = action[6]    # 左臂夹爪
                interpolated[:, 13] = action[13]  # 右臂夹爪
            else:
                interpolated[:, 6] = action[6]    # 单臂夹爪
            
            all_interpolated.append(interpolated)
            last_action = action.copy()
        
        # 合并为一个大数组
        return np.concatenate(all_interpolated, axis=0)
    
    def replay_episode(self, episode_idx: int, speed: float = 1.0, interpolate: bool = True):
        """回放单个episode - 预计算插值，快速回放"""
        print(f"\n回放 Episode {episode_idx}...")
        
        # 获取 episode 数据
        actions = self._get_episode_actions(episode_idx)
        
        if len(actions) == 0:
            print(f"Episode {episode_idx} 为空，跳过")
            return
        
        print(f"  帧数: {len(actions)}")
        print(f"  插值: {'开启' if interpolate else '关闭'}")
        
        # 回放参数
        interpolate_steps = 5 if interpolate else 1
        
        # 预先计算所有插值后的动作
        print(f"  预计算插值...")
        all_actions = self._precompute_interpolated(actions, interpolate_steps)
        print(f"  总步数: {len(all_actions)}")
        
        # 预先创建所有消息 (避免循环中创建对象)
        if self.is_dual_arm:
            msgs = []
            for act in all_actions:
                msg_left = self._create_joint_msg(act[:7].tolist())
                msg_right = self._create_joint_msg(act[7:].tolist())
                msgs.append((msg_left, msg_right))
        else:
            msgs = [self._create_joint_msg(act.tolist()) for act in all_actions]
        
        # 计算发布频率
        target_fps = self.fps * interpolate_steps
        rate = rospy.Rate(target_fps)
        
        print(f"  开始回放 (目标FPS: {target_fps})...")
        
        for i, msg in enumerate(msgs):
            if rospy.is_shutdown():
                return
            
            # 更新时间戳
            if self.is_dual_arm:
                msg[0].header.stamp = rospy.Time.now()
                msg[1].header.stamp = rospy.Time.now()
                self.joint_left_pub.publish(msg[0])
                self.joint_right_pub.publish(msg[1])
            elif self.is_left_arm:
                msg.header.stamp = rospy.Time.now()
                self.joint_left_pub.publish(msg)
            else:
                msg.header.stamp = rospy.Time.now()
                self.joint_right_pub.publish(msg)
            
            rate.sleep()
        
        print(f"  Episode {episode_idx} 回放完成")
    
    def run(self, episode_idx: int = 0, speed: float = 1.0, interpolate: bool = True, loop: bool = False):
        """运行回放"""
        self._load_dataset()
        self._init_ros()
        
        print("\n等待ROS连接...")
        time.sleep(0.5)
        
        # 获取 episodes 数量
        if 'episode_index' in self.parquet_data.columns:
            num_episodes = self.parquet_data['episode_index'].nunique()
        else:
            num_episodes = 1
        
        arm_type = "双臂" if self.is_dual_arm else ("左臂" if self.is_left_arm else "右臂")
        
        print(f"\n" + "="*50)
        print(f"Piper数据回放 ({arm_type})")
        print("="*50)
        print(f"  FPS: 100 (插值后)")
        print(f"  插值步数: {20 if interpolate else 1}")
        print("="*50)
        
        while True:
            if episode_idx >= 0:
                if episode_idx >= num_episodes:
                    print(f"错误: episode索引 {episode_idx} 超出范围")
                    return
                self.replay_episode(episode_idx, speed, interpolate)
            else:
                # 回放所有 episodes
                for ep_idx in range(num_episodes):
                    if rospy.is_shutdown():
                        return
                    self.replay_episode(ep_idx, speed, interpolate)
                    if ep_idx < num_episodes - 1:
                        print("\n等待1秒后回放下一个episode...")
                        time.sleep(1.0)
            
            if not loop:
                break
            
            print("\n循环回放，等待3秒后重新开始...")
            time.sleep(3.0)
        
        print("\n回放完成!")


def main():
    parser = argparse.ArgumentParser(description="Piper数据回放 v1 (直接读取parquet)")
    parser.add_argument("--repo-id", type=str, required=True, help="数据集名称 (使用 '.' 表示直接使用 root 目录)")
    parser.add_argument("--root", type=str, default="/home/agilex/robot/songling_code/data", help="数据集路径")
    parser.add_argument("--arm", type=str, default="left", choices=["left", "right", "both"], help="手臂选择")
    parser.add_argument("--episode", type=int, default=0, help="Episode索引 (-1 表示回放所有)")
    parser.add_argument("--fps", type=int, default=30, help="原始帧率")
    parser.add_argument("--speed", type=float, default=1.0, help="回放速度 (暂不支持)")
    parser.add_argument("--no-interpolate", action="store_true", help="禁用插值")
    parser.add_argument("--loop", action="store_true", help="循环回放")
    
    args = parser.parse_args()
    
    replay = PiperReplayV1(
        repo_id=args.repo_id,
        root=Path(args.root).expanduser(),
        arm=args.arm,
        fps=args.fps,
    )
    
    replay.run(
        episode_idx=args.episode,
        speed=args.speed,
        interpolate=not args.no_interpolate,
        loop=args.loop,
    )


if __name__ == "__main__":
    main()