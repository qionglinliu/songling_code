#!/usr/bin/env python3
"""
Piper数据采集脚本 - LeRobot格式
参考松灵ACT方案设计：
  - observation.state = puppet位置 (当前状态)
  - action = 下一帧的puppet位置 (预测目标)
  
松灵时间戳同步方式：
  - frame_time = min(所有数据的时间戳)
  - 弹出该时间戳对应的所有数据

支持的功能:
  - 手臂选择: 左臂/右臂/双臂 (--arm left/right/both)
  - 深度图支持: --use-depth
  - 摄像头自动选择

使用方法:
  # 创建新数据集 (左臂模式)
  python recorder.py --repo-id my_data --arm left
  
  # 追加到现有数据集 (使用 "." 作为 repo-id)
  python recorder.py --repo-id . --arm left
  
  # 右臂模式
  python recorder.py --repo-id my_data --arm right
  
  # 双臂模式
  python recorder.py --repo-id my_data --arm both
  
  # 启用深度图
  python recorder.py --repo-id my_data --arm both --use-depth
"""

import argparse
import time
import threading
import sys
import termios
import tty
from pathlib import Path
from collections import deque

import numpy as np
import rospy
from sensor_msgs.msg import JointState, Image
from cv_bridge import CvBridge

# LeRobot API
from lerobot.datasets.lerobot_dataset import LeRobotDataset


class PiperRecorder:
    """Piper数据采集器 - 支持左臂/右臂/双臂模式"""
    
    def __init__(
        self,
        repo_id: str,
        root: str | Path,
        arm: str = "left",
        fps: int = 30,
        use_depth: bool = False,
    ):
        self.repo_id = repo_id
        self.root = Path(root)
        self.arm = arm
        self.fps = fps
        self.use_depth = use_depth
        
        # 手臂配置
        self.is_left_arm = arm in ["left", "both"]
        self.is_right_arm = arm in ["right", "both"]
        self.is_dual_arm = arm == "both"
        
        # 摄像头配置
        if self.is_dual_arm:
            self.camera_names = ["camera_f", "camera_l", "camera_r"]
        elif self.is_left_arm:
            self.camera_names = ["camera_f", "camera_l"]
        else:
            self.camera_names = ["camera_f", "camera_r"]
        
        # 状态维度
        if self.is_dual_arm:
            self.state_dim = 14
        else:
            self.state_dim = 7
        
        # ROS数据缓存
        self.bridge = CvBridge()
        self.puppet_left_deque = deque(maxlen=2000)
        self.puppet_right_deque = deque(maxlen=2000)
        self.img_front_deque = deque(maxlen=2000)
        self.img_left_deque = deque(maxlen=2000)
        self.img_right_deque = deque(maxlen=2000)
        
        # 录制控制
        self.recording = False
        self.stop_current_episode = False
        self.lock = threading.Lock()
        
        # 数据集
        self.dataset = None
        self.features = self._create_features()
    
    def _create_features(self) -> dict:
        """创建特征定义 - LeRobot标准格式"""
        features = {
            "observation.state": {
                "dtype": "float32", 
                "shape": (self.state_dim,), 
                "names": [f"joint{i}" for i in range(self.state_dim)]
            },
            "action": {
                "dtype": "float32", 
                "shape": (self.state_dim,), 
                "names": [f"joint{i}" for i in range(self.state_dim)]
            },
        }
        
        for cam in self.camera_names:
            features[f"observation.images.{cam}"] = {
                "dtype": "video",
                "shape": (480, 640, 3),
                "names": ["height", "width", "channel"],
            }
        
        return features
    
    def _init_ros(self):
        """初始化ROS订阅"""
        rospy.init_node('piper_recorder', anonymous=True)
        
        if self.is_left_arm or self.is_dual_arm:
            rospy.Subscriber('/puppet/joint_left', JointState, 
                             self._cb_puppet_left, queue_size=1000, tcp_nodelay=True)
        if self.is_right_arm or self.is_dual_arm:
            rospy.Subscriber('/puppet/joint_right', JointState, 
                             self._cb_puppet_right, queue_size=1000, tcp_nodelay=True)
        
        rospy.Subscriber('/camera_f/color/image_raw', Image, 
                         self._cb_img_front, queue_size=1000, tcp_nodelay=True)
        if self.is_left_arm or self.is_dual_arm:
            rospy.Subscriber('/camera_l/color/image_raw', Image, 
                             self._cb_img_left, queue_size=1000, tcp_nodelay=True)
        if self.is_right_arm or self.is_dual_arm:
            rospy.Subscriber('/camera_r/color/image_raw', Image, 
                             self._cb_img_right, queue_size=1000, tcp_nodelay=True)
        
        arm_type = "双臂" if self.is_dual_arm else ("左臂" if self.is_left_arm else "右臂")
        rospy.loginfo(f"ROS订阅已初始化 - {arm_type}模式")
    
    def _cb_puppet_left(self, msg: JointState):
        self.puppet_left_deque.append(msg)
    
    def _cb_puppet_right(self, msg: JointState):
        self.puppet_right_deque.append(msg)
    
    def _cb_img_front(self, msg: Image):
        self.img_front_deque.append(msg)
    
    def _cb_img_left(self, msg: Image):
        self.img_left_deque.append(msg)
    
    def _cb_img_right(self, msg: Image):
        self.img_right_deque.append(msg)
    
    def _rosimg_to_numpy(self, msg: Image) -> np.ndarray:
        """将ROS Image消息转换为numpy数组"""
        if msg.encoding == "rgb8":
            img = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width, 3)
        elif msg.encoding == "bgr8":
            img = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width, 3)
            img = img[:, :, ::-1].copy()
        else:
            img = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width, 3)
        return img
    
    def _get_frame(self) -> dict | None:
        """获取一帧数据 - 松灵时间戳同步方式"""
        # 检查数据是否完整
        if len(self.img_front_deque) == 0:
            return None
        if self.is_left_arm or self.is_dual_arm:
            if len(self.img_left_deque) == 0:
                return None
        if self.is_right_arm or self.is_dual_arm:
            if len(self.img_right_deque) == 0:
                return None
        if self.is_left_arm or self.is_dual_arm:
            if len(self.puppet_left_deque) == 0:
                return None
        if self.is_right_arm or self.is_dual_arm:
            if len(self.puppet_right_deque) == 0:
                return None
        
        # 计算frame_time
        timestamps = [self.img_front_deque[-1].header.stamp.to_sec()]
        if self.is_left_arm or self.is_dual_arm:
            timestamps.append(self.img_left_deque[-1].header.stamp.to_sec())
        if self.is_right_arm or self.is_dual_arm:
            timestamps.append(self.img_right_deque[-1].header.stamp.to_sec())
        if self.is_left_arm or self.is_dual_arm:
            timestamps.append(self.puppet_left_deque[-1].header.stamp.to_sec())
        if self.is_right_arm or self.is_dual_arm:
            timestamps.append(self.puppet_right_deque[-1].header.stamp.to_sec())
        
        frame_time = min(timestamps)
        
        # 弹出frame_time对应的数据
        while self.img_front_deque[0].header.stamp.to_sec() < frame_time:
            self.img_front_deque.popleft()
        img_front = self._rosimg_to_numpy(self.img_front_deque.popleft())
        
        if self.is_left_arm or self.is_dual_arm:
            while self.img_left_deque[0].header.stamp.to_sec() < frame_time:
                self.img_left_deque.popleft()
            img_left = self._rosimg_to_numpy(self.img_left_deque.popleft())
        
        if self.is_right_arm or self.is_dual_arm:
            while self.img_right_deque[0].header.stamp.to_sec() < frame_time:
                self.img_right_deque.popleft()
            img_right = self._rosimg_to_numpy(self.img_right_deque.popleft())
        
        if self.is_left_arm or self.is_dual_arm:
            while self.puppet_left_deque[0].header.stamp.to_sec() < frame_time:
                self.puppet_left_deque.popleft()
            puppet_left = self.puppet_left_deque.popleft()
        
        if self.is_right_arm or self.is_dual_arm:
            while self.puppet_right_deque[0].header.stamp.to_sec() < frame_time:
                self.puppet_right_deque.popleft()
            puppet_right = self.puppet_right_deque.popleft()
        
        # 构建state
        if self.is_dual_arm:
            state = np.concatenate([
                np.array(puppet_left.position),
                np.array(puppet_right.position)
            ], dtype=np.float32)
        elif self.is_left_arm:
            state = np.array(puppet_left.position, dtype=np.float32)
        else:
            state = np.array(puppet_right.position, dtype=np.float32)
        
        images = {"camera_f": img_front}
        if self.is_left_arm or self.is_dual_arm:
            images["camera_l"] = img_left
        if self.is_right_arm or self.is_dual_arm:
            images["camera_r"] = img_right
        
        return {
            "observation.state": state,
            "images": images,
        }
    
    def _init_dataset(self):
        """初始化LeRobot数据集
        
        支持两种模式:
        1. --repo-id my_data: 数据集路径为 root/my_data/
        2. --repo-id . : 直接使用 root 作为数据集路径（用于追加到现有数据集）
        """
        # 如果 repo-id 是 "."，则直接使用 root 作为数据集路径
        if self.repo_id == ".":
            dataset_path = self.root
            # 使用 root 目录名作为实际 repo_id
            actual_repo_id = self.root.name
        else:
            dataset_path = self.root / self.repo_id
            actual_repo_id = self.repo_id
            
        if dataset_path.exists() and (dataset_path / "meta" / "info.json").exists():
            print(f"数据集已存在，追加数据: {dataset_path}")
            self.dataset = LeRobotDataset(
                repo_id=actual_repo_id,
                root=dataset_path,
            )
        else:
            self.dataset = LeRobotDataset.create(
                repo_id=actual_repo_id,
                root=dataset_path,
                fps=self.fps,
                features=self.features,
            )
        print(f"数据集路径: {dataset_path}")
    
    def record_episode(self, task: str = "演示任务") -> int:
        """录制一个episode"""
        print("\n" + "="*50)
        print("准备录制...")
        print("="*50)
        
        print("按回车键开始录制 (按q退出)...")
        try:
            old_settings = termios.tcgetattr(sys.stdin)
            try:
                tty.setraw(sys.stdin.fileno())
                ch = sys.stdin.read(1)
                if ch == '\x03' or ch == 'q':
                    print("用户取消录制")
                    return -1
            finally:
                termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_settings)
        except Exception:
            input()
        
        self.stop_current_episode = False
        
        # 等待数据
        print("等待ROS数据...")
        frame = None
        for i in range(100):
            frame = self._get_frame()
            if frame is not None:
                print(f"数据就绪! (等待 {i+1} 次)")
                break
            time.sleep(0.05)
        
        if frame is None:
            print("错误: 未收到完整ROS数据!")
            return 0
        
        print("开始录制... (按回车键停止录制)")
        frame_count = 0
        dt = 1.0 / self.fps
        raw_frames = []
        
        # 停止监听线程
        def wait_for_stop():
            try:
                old_settings = termios.tcgetattr(sys.stdin)
                try:
                    tty.setraw(sys.stdin.fileno())
                    while not self.stop_current_episode:
                        ch = sys.stdin.read(1)
                        if ch == '\x03' or ch == 'q' or ch == '\r' or ch == '\n':
                            self.stop_current_episode = True
                            break
                finally:
                    termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_settings)
            except Exception:
                pass
        
        stop_thread = threading.Thread(target=wait_for_stop, daemon=True)
        stop_thread.start()
        
        try:
            while not self.stop_current_episode:
                start_time = time.time()
                
                frame = self._get_frame()
                if frame is not None:
                    raw_frames.append(frame)
                    frame_count += 1
                    
                    if frame_count % 30 == 0:
                        print(f"  已录制 {frame_count} 帧...")
                
                elapsed = time.time() - start_time
                if elapsed < dt:
                    time.sleep(dt - elapsed)
                    
        except Exception as e:
            print(f"\n录制出错: {e}")
        
        print("\n停止录制...")
        
        # 保存episode
        if len(raw_frames) > 1:
            saved_count = 0
            for i in range(len(raw_frames) - 1):
                current_frame = raw_frames[i]
                next_frame = raw_frames[i + 1]
                
                frame_to_save = {
                    "observation.state": current_frame["observation.state"],
                    "action": next_frame["observation.state"],
                    "task": task,
                }
                
                for cam_name, img in current_frame.get("images", {}).items():
                    frame_to_save[f"observation.images.{cam_name}"] = img
                
                self.dataset.add_frame(frame_to_save)
                saved_count += 1
            
            self.dataset.save_episode()
            print(f"Episode已保存: {saved_count} 帧 (原始录制 {frame_count} 帧)")
            return saved_count
        elif len(raw_frames) == 1:
            print("只有1帧数据，无法生成action，跳过保存")
            return 0
        
        return frame_count
    
    def run(self, num_episodes: int = 10, task: str = "机器人操作演示"):
        """运行采集"""
        self._init_ros()
        time.sleep(1.0)
        self._init_dataset()
        
        arm_type = "双臂" if self.is_dual_arm else ("左臂" if self.is_left_arm else "右臂")
        cameras = ", ".join(self.camera_names)
        
        print(f"\n" + "="*50)
        print(f"Piper数据采集 ({arm_type})")
        print("="*50)
        print(f"  目标episodes: {num_episodes}")
        print(f"  保存路径: {self.root}")
        print(f"  FPS: {self.fps}")
        print(f"  摄像头: {cameras}")
        print(f"\n数据说明:")
        print(f"  observation.state = puppet位置 (当前状态)")
        print(f"  action = 下一帧puppet位置 (预测目标)")
        print(f"\n操作说明:")
        print(f"  - 按回车键开始/停止录制")
        print(f"  - 按 q 键退出整个采集")
        print("="*50)
        
        completed_episodes = 0
        for ep in range(num_episodes):
            print(f"\n[Episode {ep + 1}/{num_episodes}]")
            frames = self.record_episode(task)
            
            if frames == -1:
                print("用户取消采集")
                break
            elif frames == 0:
                print("录制失败，跳过此episode")
                continue
            
            completed_episodes += 1
        
        # 完成后调用finalize
        self.dataset.finalize()
        
        print("\n" + "="*50)
        print("数据采集完成!")
        print(f"  成功录制: {completed_episodes} 个episodes")
        print(f"  数据集路径: {self.root / self.repo_id}")
        print("="*50)


def main():
    parser = argparse.ArgumentParser(description="Piper数据采集 (支持左臂/右臂/双臂)")
    parser.add_argument("--repo-id", type=str, required=True, help="数据集名称")
    parser.add_argument("--root", type=str, default="/home/agilex/robot/songling_code/data", help="数据保存路径")
    parser.add_argument("--arm", type=str, default="left", choices=["left", "right", "both"], help="手臂选择")
    parser.add_argument("--num-episodes", type=int, default=10, help="采集episode数量")
    parser.add_argument("--fps", type=int, default=30, help="采集帧率")
    parser.add_argument("--task", type=str, default="机器人操作演示", help="任务描述")
    parser.add_argument("--use-depth", action="store_true", help="是否采集深度图")
    
    args = parser.parse_args()
    
    recorder = PiperRecorder(
        repo_id=args.repo_id,
        root=Path(args.root).expanduser(),
        arm=args.arm,
        fps=args.fps,
        use_depth=args.use_depth,
    )
    
    recorder.run(
        num_episodes=args.num_episodes,
        task=args.task,
    )


if __name__ == "__main__":
    main()