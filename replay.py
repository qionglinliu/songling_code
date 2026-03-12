#!/usr/bin/env python3
"""
Piper数据回放脚本 - LeRobot格式
参考松灵replay_data.py设计

功能:
  - 从LeRobot数据集读取数据
  - 发布关节状态到ROS话题
  - 发布图像到ROS话题
  - 支持左臂/右臂/双臂模式
  - 支持插值平滑运动

使用方法:
  # 回放指定episode
  python replay.py --repo-id my_data --episode 0
  
  # 回放所有episode
  python replay.py --repo-id my_data --episode -1
  
  # 指定手臂
  python replay.py --repo-id my_data --episode 0 --arm left
  
  # 慢速回放 (用于调试)
  python replay.py --repo-id my_data --episode 0 --speed 0.5
"""

import argparse
import time
from pathlib import Path

import numpy as np
import rospy
from cv_bridge import CvBridge
from std_msgs.msg import Header
from sensor_msgs.msg import Image, JointState

# LeRobot API
from lerobot.datasets.lerobot_dataset import LeRobotDataset


class PiperReplay:
    """Piper数据回放器 - 支持左臂/右臂/双臂模式"""
    
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
        
        # 状态维度
        if self.is_dual_arm:
            self.state_dim = 14  # 双臂: 7+7
        else:
            self.state_dim = 7   # 单臂
        
        # ROS
        self.bridge = CvBridge()
        self.joint_left_pub = None
        self.joint_right_pub = None
        self.img_front_pub = None
        self.img_left_pub = None
        self.img_right_pub = None
        
        # 数据集
        self.dataset = None
        
        # 夹爪初始位置 (参考松灵)
        self.origin_left = [-0.0057, -0.031, -0.0122, -0.032, 0.0099, 0.0179, 0.2279]
        self.origin_right = [0.0616, 0.0021, 0.0475, -0.1013, 0.1097, 0.0872, 0.2279]
    
    def _init_ros(self):
        """初始化ROS发布者"""
        rospy.init_node('piper_replay', anonymous=True)
        
        # 关节状态发布者
        if self.is_left_arm or self.is_dual_arm:
            self.joint_left_pub = rospy.Publisher(
                '/master/joint_left', JointState, queue_size=10
            )
        if self.is_right_arm or self.is_dual_arm:
            self.joint_right_pub = rospy.Publisher(
                '/master/joint_right', JointState, queue_size=10
            )
        
        # 图像发布者
        self.img_front_pub = rospy.Publisher(
            '/camera_f/color/image_raw', Image, queue_size=10
        )
        if self.is_left_arm or self.is_dual_arm:
            self.img_left_pub = rospy.Publisher(
                '/camera_l/color/image_raw', Image, queue_size=10
            )
        if self.is_right_arm or self.is_dual_arm:
            self.img_right_pub = rospy.Publisher(
                '/camera_r/color/image_raw', Image, queue_size=10
            )
        
        arm_type = "双臂" if self.is_dual_arm else ("左臂" if self.is_left_arm else "右臂")
        rospy.loginfo(f"ROS发布者已初始化 - {arm_type}模式")
    
    def _load_dataset(self):
        """加载LeRobot数据集"""
        dataset_path = self.root / self.repo_id
        if not dataset_path.exists():
            raise FileNotFoundError(f"数据集不存在: {dataset_path}")
        
        self.dataset = LeRobotDataset(
            repo_id=self.repo_id,
            root=self.root,
        )
        print(f"数据集已加载: {dataset_path}")
        print(f"  总帧数: {len(self.dataset)}")
        print(f"  Episodes: {self.dataset.num_episodes}")
    
    def _create_joint_msg(self, positions: np.ndarray) -> JointState:
        """创建关节状态消息"""
        msg = JointState()
        msg.header = Header()
        msg.header.stamp = rospy.Time.now()
        msg.name = [f'joint{i}' for i in range(len(positions))]
        msg.position = positions.tolist()
        return msg
    
    def _create_image_msg(self, image: np.ndarray) -> Image:
        """创建图像消息 - 不使用cv_bridge避免libffi冲突"""
        msg = Image()
        msg.header = Header()
        msg.header.stamp = rospy.Time.now()
        msg.height = image.shape[0]
        msg.width = image.shape[1]
        
        if len(image.shape) == 3:
            if image.shape[2] == 3:
                # 输入是RGB，转换为BGRleft
                image_bgr = image[:, :, ::-1].copy()
                msg.encoding = "bgr8"
                msg.step = msg.width * 3
            else:
                image_bgr = image
                msg.encoding = "rgb8"
                msg.step = msg.width * image.shape[2]
        else:
            image_bgr = image
            msg.encoding = "mono8"
            msg.step = msg.width
        
        msg.data = image_bgr.tobytes()
        msg.is_bigendian = 0
        return msg
    
    def _interpolate_actions(self, last_action: np.ndarray, current_action: np.ndarray, num_steps: int) -> np.ndarray:
        """插值生成平滑动作序列 (参考松灵做法)"""
        return np.linspace(last_action, current_action, num_steps)
    
    def _get_episode_data(self, episode_idx: int) -> list:
        """获取指定episode的所有数据"""
        episode_data = []
        
        # 使用meta.episodes获取episode信息
        ep = self.dataset.meta.episodes[episode_idx]
        from_idx = ep["dataset_from_index"]
        to_idx = ep["dataset_to_index"]
        
        for i in range(from_idx, to_idx):
            frame = self.dataset[i]
            episode_data.append(frame)
        
        return episode_data
    
    def replay_episode(self, episode_idx: int, speed: float = 1.0, interpolate: bool = True):
        """回放单个episode
        
        Args:
            episode_idx: episode索引
            speed: 回放速度 (1.0 = 正常速度)
            interpolate: 是否使用插值平滑
        """
        print(f"\n回放 Episode {episode_idx}...")
        
        # 获取episode数据
        episode_data = self._get_episode_data(episode_idx)
        
        if not episode_data:
            print(f"Episode {episode_idx} 为空，跳过")
            return
        
        print(f"  帧数: {len(episode_data)}")
        
        # 回放参数
        interpolate_steps = 20 if interpolate else 1
        
        # 初始位置
        if self.is_dual_arm:
            last_action = np.array(self.origin_left + self.origin_right, dtype=np.float32)
        elif self.is_left_arm:
            last_action = np.array(self.origin_left, dtype=np.float32)
        else:
            last_action = np.array(self.origin_right, dtype=np.float32)
        
        rate = rospy.Rate(self.fps * interpolate_steps)
        
        for frame_idx, frame in enumerate(episode_data):
            if rospy.is_shutdown():
                return
            
            # 获取当前帧数据
            action = frame['action'].numpy()
            
            # 插值
            if interpolate:
                interpolated = self._interpolate_actions(last_action, action, interpolate_steps)
            else:
                interpolated = [action]
            
            last_action = action.copy()
            
            for step, act in enumerate(interpolated):
                if rospy.is_shutdown():
                    return
                
                # 发布关节状态
                if self.is_dual_arm:
                    left_action = act[:7]
                    right_action = act[7:]
                    self.joint_left_pub.publish(self._create_joint_msg(left_action))
                    self.joint_right_pub.publish(self._create_joint_msg(right_action))
                elif self.is_left_arm:
                    self.joint_left_pub.publish(self._create_joint_msg(act))
                else:
                    self.joint_right_pub.publish(self._create_joint_msg(act))
                
                # 只在第一步发布图像 (避免重复)
                if step == 0:
                    # 发布图像
                    if 'observation.images.camera_f' in frame:
                        img_f = frame['observation.images.camera_f'].numpy()
                        self.img_front_pub.publish(self._create_image_msg(img_f))
                    
                    if self.is_left_arm or self.is_dual_arm:
                        if 'observation.images.camera_l' in frame:
                            img_l = frame['observation.images.camera_l'].numpy()
                            self.img_left_pub.publish(self._create_image_msg(img_l))
                    
                    if self.is_right_arm or self.is_dual_arm:
                        if 'observation.images.camera_r' in frame:
                            img_r = frame['observation.images.camera_r'].numpy()
                            self.img_right_pub.publish(self._create_image_msg(img_r))
                
                rate.sleep()
            
            if frame_idx % 30 == 0:
                print(f"  已回放 {frame_idx + 1}/{len(episode_data)} 帧")
        
        print(f"  Episode {episode_idx} 回放完成")
    
    def replay_all(self, speed: float = 1.0, interpolate: bool = True, loop: bool = False):
        """回放所有episodes"""
        num_episodes = self.dataset.num_episodes
        
        while True:
            for ep_idx in range(num_episodes):
                if rospy.is_shutdown():
                    return
                self.replay_episode(ep_idx, speed=speed, interpolate=interpolate)
                time.sleep(1.0)  # episode之间暂停
            
            if not loop:
                break
            print("\n循环回放...")
    
    def run(self, episode_idx: int = 0, speed: float = 1.0, interpolate: bool = True, loop: bool = False):
        """运行回放"""
        self._init_ros()
        time.sleep(0.5)
        self._load_dataset()
        
        arm_type = "双臂" if self.is_dual_arm else ("左臂" if self.is_left_arm else "右臂")
        
        print(f"\n" + "="*50)
        print(f"Piper数据回放 ({arm_type})")
        print("="*50)
        print(f"  数据集: {self.root / self.repo_id}")
        print(f"  FPS: {self.fps}")
        print(f"  速度: {speed}x")
        print(f"  插值: {'启用' if interpolate else '禁用'}")
        print(f"  循环: {'启用' if loop else '禁用'}")
        print("="*50)
        
        if episode_idx >= 0:
            # 回放指定episode
            if episode_idx >= self.dataset.num_episodes:
                print(f"错误: episode索引 {episode_idx} 超出范围 (0-{self.dataset.num_episodes-1})")
                return
            self.replay_episode(episode_idx, speed=speed, interpolate=interpolate)
        else:
            # 回放所有episodes
            self.replay_all(speed=speed, interpolate=interpolate, loop=loop)
        
        print("\n回放完成!")


def main():
    parser = argparse.ArgumentParser(description="Piper数据回放 (支持左臂/右臂/双臂)")
    parser.add_argument("--repo-id", type=str, required=True, help="数据集名称")
    parser.add_argument("--root", type=str, default="/home/agilex/robot/code/code/data", help="数据集路径")
    parser.add_argument("--arm", type=str, default="left", choices=["left", "right", "both"], help="手臂选择")
    parser.add_argument("--episode", type=int, default=0, help="Episode索引 (-1 表示回放所有)")
    parser.add_argument("--fps", type=int, default=30, help="回放帧率")
    parser.add_argument("--speed", type=float, default=1.0, help="回放速度 (0.5 = 半速, 2.0 = 双速)")
    parser.add_argument("--no-interpolate", action="store_true", help="禁用插值")
    parser.add_argument("--loop", action="store_true", help="循环回放")
    
    args = parser.parse_args()
    
    replay = PiperReplay(
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