#!/usr/bin/env python3
"""
Piper数据回放脚本 - 直接从parquet文件回放动作到机器人
"""

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
import rospy
from sensor_msgs.msg import JointState
from std_msgs.msg import Header


class PiperReplay:
    """Piper数据回放器"""
    
    def __init__(
        self,
        dataset_path: str | Path,
        fps: int = 30,
    ):
        self.dataset_path = Path(dataset_path)
        self.fps = fps
        
        # 数据
        self.data = None
        
        # ROS发布者
        self.pub_left = None
        self.pub_right = None
    
    def _init_ros(self):
        """初始化ROS"""
        rospy.init_node('piper_replay', anonymous=True)
        
        self.pub_left = rospy.Publisher('/master/joint_left', JointState, queue_size=10)
        self.pub_right = rospy.Publisher('/master/joint_right', JointState, queue_size=10)
        
        rospy.loginfo("ROS初始化完成")
    
    def load_dataset(self):
        """加载数据集"""
        print(f"加载数据集: {self.dataset_path}")
        
        # 读取meta/info.json获取episode信息
        info_path = self.dataset_path / "meta" / "info.json"
        if not info_path.exists():
            raise FileNotFoundError(f"找不到info.json: {info_path}")
        
        import json
        with open(info_path) as f:
            info = json.load(f)
        
        total_episodes = info["total_episodes"]
        print(f"  总episodes: {total_episodes}")
        
        # 读取所有parquet文件
        data_dir = self.dataset_path / "data" / "chunk-000"
        parquet_files = sorted(data_dir.glob("*.parquet"))
        
        print(f"  读取 {len(parquet_files)} 个parquet文件...")
        
        dfs = []
        for pf in parquet_files:
            df = pd.read_parquet(pf)
            dfs.append(df)
        
        self.data = pd.concat(dfs, ignore_index=True)
        print(f"  总帧数: {len(self.data)}")
        print(f"  列: {list(self.data.columns)}")
    
    def _publish_action(self, action: np.ndarray):
        """发布动作到机器人"""
        left_action = action[:7]
        right_action = action[7:14]
        
        msg_left = JointState()
        msg_left.header = Header()
        msg_left.header.stamp = rospy.Time.now()
        msg_left.name = [f"left_j{i}" for i in range(7)]
        msg_left.position = left_action.tolist()
        self.pub_left.publish(msg_left)
        
        msg_right = JointState()
        msg_right.header = Header()
        msg_right.header.stamp = rospy.Time.now()
        msg_right.name = [f"right_j{i}" for i in range(7)]
        msg_right.position = right_action.tolist()
        self.pub_right.publish(msg_right)
    
    def replay_episode(self, episode_idx: int = 0):
        """回放指定的episode"""
        if self.data is None:
            print("错误: 数据未加载")
            return
        
        # 获取该episode的数据
        episode_data = self.data[self.data['episode_index'] == episode_idx].copy()
        episode_data = episode_data.sort_values('frame_index')
        
        if len(episode_data) == 0:
            print(f"错误: 找不到episode {episode_idx}")
            return
        
        num_frames = len(episode_data)
        print(f"\n回放 Episode {episode_idx}")
        print(f"  总帧数: {num_frames}")
        print(f"  预计时长: {num_frames / self.fps:.1f}s")
        print("\n按 Ctrl+C 停止")
        
        # 等待ROS连接
        time.sleep(0.5)
        
        dt = 1.0 / self.fps
        frame_count = 0
        
        try:
            for _, row in episode_data.iterrows():
                start_time = time.time()
                
                # 获取action数据 (14维)
                action = np.array(row['action'])
                
                # 发布
                self._publish_action(action)
                
                frame_count += 1
                if frame_count % 30 == 0:
                    progress = frame_count / num_frames * 100
                    print(f"  进度: {progress:.0f}% ({frame_count}/{num_frames} 帧)")
                
                # 控制频率
                elapsed = time.time() - start_time
                if elapsed < dt:
                    time.sleep(dt - elapsed)
                    
        except KeyboardInterrupt:
            print("\n用户停止回放")
        
        print(f"\n回放完成, 共 {frame_count} 帧")
    
    def list_episodes(self):
        """列出所有episode"""
        if self.data is None:
            return
        
        episodes = self.data['episode_index'].unique()
        episodes = sorted(episodes)
        print(f"\n可用episodes ({len(episodes)}个):")
        for ep in episodes[:10]:
            ep_data = self.data[self.data['episode_index'] == ep]
            print(f"  Episode {ep}: {len(ep_data)} 帧")
        if len(episodes) > 10:
            print(f"  ... 还有 {len(episodes) - 10} 个episodes")
    
    def run(self, episode_idx: int = 0):
        """运行回放"""
        self._init_ros()
        self.load_dataset()
        self.list_episodes()
        self.replay_episode(episode_idx)


def main():
    parser = argparse.ArgumentParser(description="Piper数据回放")
    parser.add_argument("--dataset", type=str, default="my_data_100", help="数据集路径")
    parser.add_argument("--episode", type=int, default=0, help="要回放的episode索引")
    parser.add_argument("--fps", type=int, default=30, help="回放帧率")
    
    args = parser.parse_args()
    
    dataset_path = Path(__file__).parent / args.dataset
    
    replay = PiperReplay(
        dataset_path=dataset_path,
        fps=args.fps,
    )
    
    replay.run(episode_idx=args.episode)


if __name__ == "__main__":
    main()