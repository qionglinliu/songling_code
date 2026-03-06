#!/usr/bin/env python3
"""
Piper数据回放脚本 - LeRobot格式
读取采集的数据并发布到ROS话题进行回放

使用方法:
  python piper_replay_v2.py --repo-id test_data --episode 0
"""

import argparse
import time
from pathlib import Path

import numpy as np
import rospy
from sensor_msgs.msg import JointState, Image
from cv_bridge import CvBridge

from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata


class PiperReplay:
    """Piper数据回放器"""
    
    def __init__(self, repo_id: str, root: Path):
        self.repo_id = repo_id
        self.root = root
        self.dataset = None
        self.bridge = CvBridge()
        
        # ROS发布器
        self.pub_puppet_joints = None
        self.pub_master_joints = None
        self.pub_camera_f = None
        self.pub_camera_l = None
        
    def load_dataset(self):
        """加载数据集"""
        dataset_path = self.root / self.repo_id
        print(f"加载数据集: {dataset_path}")
        
        # 检查数据集是否存在
        if not dataset_path.exists():
            raise FileNotFoundError(f"数据集不存在: {dataset_path}")
        
        # 检查必要文件
        required_files = ["meta/info.json", "meta/tasks.parquet"]
        for f in required_files:
            if not (dataset_path / f).exists():
                raise FileNotFoundError(f"缺少必要文件: {dataset_path / f}")
        
        # 使用本地模式加载，避免连接HuggingFace Hub
        # 设置环境变量禁用网络请求
        import os
        os.environ['HF_HUB_OFFLINE'] = '1'
        
        try:
            self.dataset = LeRobotDataset(
                repo_id=self.repo_id,
                root=self.root,
                local_files_only=True,
            )
        except Exception as e:
            print(f"加载数据集失败: {e}")
            print("\n尝试直接读取parquet文件...")
            self._load_dataset_manual()
            return
        
        print(f"数据集信息:")
        print(f"  Episodes数量: {self.dataset.num_episodes}")
        print(f"  总帧数: {self.dataset.num_frames}")
        print(f"  FPS: {self.dataset.fps}")
    
    def _load_dataset_manual(self):
        """手动加载数据集（备用方案）"""
        import json
        import pyarrow.parquet as pq
        
        dataset_path = self.root / self.repo_id
        
        # 读取info.json
        with open(dataset_path / "meta/info.json") as f:
            self.info = json.load(f)
        
        # 读取所有parquet数据文件
        self.frames = []
        data_dir = dataset_path / "data" / "chunk-000"
        if data_dir.exists():
            parquet_files = sorted(data_dir.glob("*.parquet"))
            for pf in parquet_files:
                table = pq.read_table(pf)
                for i in range(table.num_rows):
                    row = {col: table[col][i].as_py() for col in table.column_names}
                    self.frames.append(row)
        
        # 从info.json读取基本信息
        self.num_episodes = self.info.get("total_episodes", 0)
        self.num_frames = self.info.get("total_frames", len(self.frames))
        self.fps = self.info.get("fps", 30)
        
        # 从episode parquet文件读取episode长度，构建索引
        self.episode_data_index = {"from": [], "to": []}
        episodes_dir = dataset_path / "meta" / "episodes" / "chunk-000"
        if episodes_dir.exists():
            ep_parquet_files = sorted(episodes_dir.glob("*.parquet"))
            for epf in ep_parquet_files:
                ep_table = pq.read_table(epf)
                # 读取episode_index和length
                if 'episode_index' in ep_table.column_names and 'length' in ep_table.column_names:
                    episode_indices = ep_table['episode_index'].to_pylist()
                    lengths = ep_table['length'].to_pylist()
                    
                    # 根据长度构建索引
                    frame_idx = 0
                    for ep_idx, length in zip(episode_indices, lengths):
                        self.episode_data_index["from"].append(frame_idx)
                        frame_idx += length
                        self.episode_data_index["to"].append(frame_idx)
        
        # 如果没找到episode索引，使用简单分配
        if not self.episode_data_index["from"]:
            if self.num_episodes > 0:
                frames_per_ep = self.num_frames // self.num_episodes
                for i in range(self.num_episodes):
                    self.episode_data_index["from"].append(i * frames_per_ep)
                    self.episode_data_index["to"].append((i + 1) * frames_per_ep if i < self.num_episodes - 1 else self.num_frames)
            else:
                self.episode_data_index = {"from": [0], "to": [self.num_frames]}
                self.num_episodes = 1 if self.num_frames > 0 else 0
        
        self._manual_mode = True
        print(f"数据集信息 (手动模式):")
        print(f"  Episodes数量: {self.num_episodes}")
        print(f"  总帧数: {self.num_frames}")
        print(f"  FPS: {self.fps}")
        print(f"  Episode索引: {self.episode_data_index}")
        
    def _init_ros(self):
        """初始化ROS发布器"""
        rospy.init_node('piper_replay', anonymous=True)
        
        # 发布到回放话题 (带_replay后缀，避免与原始话题冲突)
        self.pub_puppet_joints = rospy.Publisher(
            '/puppet/joint_left', JointState, queue_size=10
        )
        self.pub_master_joints = rospy.Publisher(
            '/master/joint_left', JointState, queue_size=10
        )
        self.pub_camera_f = rospy.Publisher(
            '/camera_f/color/image_raw', Image, queue_size=10
        )
        self.pub_camera_l = rospy.Publisher(
            '/camera_l/color/image_raw', Image, queue_size=10
        )
        
        print("ROS发布器已初始化:")
        print(f"  /puppet/joint_left")
        print(f"  /master/joint_left")
        print(f"  /camera_f/color/image_raw")
        print(f"  /camera_l/color/image_raw")
        
    def _publish_joints(self, joints, is_master: bool = True, timestamp: float = 0):
        """发布关节数据"""
        msg = JointState()
        msg.header.stamp = rospy.Time.from_sec(timestamp)
        msg.name = [f'joint{i}' for i in range(7)]
        # 支持numpy数组和list两种格式
        if hasattr(joints, 'tolist'):
            msg.position = joints.tolist()
        else:
            msg.position = list(joints)
        msg.velocity = [0.0] * 7
        msg.effort = [0.0] * 7
        
        if is_master:
            self.pub_master_joints.publish(msg)
        else:
            self.pub_puppet_joints.publish(msg)
            
    def _publish_image(self, image: np.ndarray, camera_name: str, timestamp: float = 0):
        """发布图像数据"""
        msg = self.bridge.cv2_to_imgmsg(image, encoding="rgb8")
        msg.header.stamp = rospy.Time.from_sec(timestamp)
        
        if camera_name == 'camera_f':
            self.pub_camera_f.publish(msg)
        elif camera_name == 'camera_l':
            self.pub_camera_l.publish(msg)
    
    def _get_frame(self, frame_idx: int):
        """获取指定帧"""
        if hasattr(self, '_manual_mode') and self._manual_mode:
            return self.frames[frame_idx]
        else:
            return self.dataset[frame_idx]
    
    def _get_num_episodes(self):
        """获取episode数量"""
        if hasattr(self, '_manual_mode') and self._manual_mode:
            return self.num_episodes
        else:
            return self.dataset.num_episodes
    
    def _get_episode_range(self, episode_idx: int):
        """获取episode的帧范围"""
        if hasattr(self, '_manual_mode') and self._manual_mode:
            from_idx = self.episode_data_index["from"][episode_idx]
            to_idx = self.episode_data_index["to"][episode_idx]
        else:
            from_idx = self.dataset.episode_data_index["from"][episode_idx].item()
            to_idx = self.dataset.episode_data_index["to"][episode_idx].item()
        return from_idx, to_idx
    
    def replay_episode(self, episode_idx: int, speed: float = 1.0, loop: bool = False):
        """回放指定episode"""
        num_episodes = self._get_num_episodes()
        
        if episode_idx >= num_episodes:
            print(f"错误: episode索引 {episode_idx} 超出范围 (共 {num_episodes} 个)")
            return
            
        # 获取该episode的帧范围
        from_idx, to_idx = self._get_episode_range(episode_idx)
        num_frames = to_idx - from_idx
        
        print(f"\n回放 Episode {episode_idx}:")
        print(f"  帧范围: [{from_idx}, {to_idx})")
        print(f"  帧数: {num_frames}")
        print(f"  速度: {speed}x")
        print(f"  循环: {loop}")
        
        fps = self.fps if hasattr(self, '_manual_mode') and self._manual_mode else self.dataset.fps
        dt = 1.0 / fps / speed
        
        round_count = 0
        while True:
            print(f"\n--- 第 {round_count + 1} 次回放 ---")
            
            for frame_idx in range(from_idx, to_idx):
                if rospy.is_shutdown():
                    return
                    
                # 读取帧数据
                frame = self._get_frame(frame_idx)
                
                # 获取时间戳
                timestamp = frame.get('timestamp', 0.0)
                
                # 发布关节数据
                if 'observation.state' in frame:
                    state = frame['observation.state']
                    if hasattr(state, 'numpy'):
                        state = state.numpy()
                    self._publish_joints(
                        state,
                        is_master=False,
                        timestamp=timestamp
                    )
                    
                if 'action' in frame:
                    action = frame['action']
                    if hasattr(action, 'numpy'):
                        action = action.numpy()
                    self._publish_joints(
                        action,
                        is_master=True,
                        timestamp=timestamp
                    )
                
                # 发布图像
                if 'observation.images.camera_f' in frame:
                    img = frame['observation.images.camera_f']
                    if hasattr(img, 'numpy'):
                        img = img.numpy()
                    self._publish_image(img, 'camera_f', timestamp)
                    
                if 'observation.images.camera_l' in frame:
                    img = frame['observation.images.camera_l']
                    if hasattr(img, 'numpy'):
                        img = img.numpy()
                    self._publish_image(img, 'camera_l', timestamp)
                
                # 打印进度
                current_frame = frame_idx - from_idx
                if current_frame % 30 == 0:
                    print(f"  回放进度: {current_frame}/{num_frames} 帧")
                
                time.sleep(dt)
            
            round_count += 1
            
            if not loop:
                break
                
            print("回放完成，等待1秒后重新开始...")
            time.sleep(1.0)
        
        print("\n回放完成!")
        
    def list_episodes(self):
        """列出所有episodes的信息"""
        num_episodes = self._get_num_episodes()
        fps = self.fps if hasattr(self, '_manual_mode') and self._manual_mode else self.dataset.fps
        
        print(f"\n数据集: {self.root / self.repo_id}")
        print(f"Episodes数量: {num_episodes}")
        print("-" * 50)
        
        for ep_idx in range(num_episodes):
            from_idx, to_idx = self._get_episode_range(ep_idx)
            num_frames = to_idx - from_idx
            duration = num_frames / fps
            
            print(f"Episode {ep_idx}: {num_frames} 帧, {duration:.1f} 秒")
            
    def run(self, episode_idx: int = None, speed: float = 1.0, loop: bool = False, list_only: bool = False):
        """运行回放"""
        self.load_dataset()
        
        if list_only:
            self.list_episodes()
            return
            
        self._init_ros()
        
        # 等待订阅者连接
        print("\n等待ROS连接...")
        time.sleep(1.0)
        
        num_episodes = self._get_num_episodes()
        
        if episode_idx is not None:
            self.replay_episode(episode_idx, speed, loop)
        else:
            # 回放所有episodes
            for ep_idx in range(num_episodes):
                if rospy.is_shutdown():
                    break
                self.replay_episode(ep_idx, speed, loop=False)
                if ep_idx < num_episodes - 1:
                    print("\n等待2秒后回放下一个episode...")
                    time.sleep(2.0)


def main():
    parser = argparse.ArgumentParser(description="Piper数据回放")
    parser.add_argument("--repo-id", type=str, required=True, help="数据集名称")
    parser.add_argument("--root", type=str, default="~/my_data", help="数据集路径")
    parser.add_argument("--episode", type=int, default=None, help="指定回放的episode索引")
    parser.add_argument("--speed", type=float, default=1.0, help="回放速度倍率")
    parser.add_argument("--loop", action="store_true", help="循环回放")
    parser.add_argument("--list", action="store_true", help="只列出episodes信息")
    
    args = parser.parse_args()
    
    replay = PiperReplay(
        repo_id=args.repo_id,
        root=Path(args.root).expanduser(),
    )
    
    replay.run(
        episode_idx=args.episode,
        speed=args.speed,
        loop=args.loop,
        list_only=args.list,
    )


if __name__ == "__main__":
    main()