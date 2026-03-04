#!/usr/bin/env python3
"""
Piper双臂数据采集脚本 - LeRobot格式
不修改lerobot源码，独立运行
"""

import argparse
import time
import threading
from pathlib import Path

import numpy as np
import rospy
from sensor_msgs.msg import JointState, Image

# LeRobot API
from lerobot.datasets.lerobot_dataset import LeRobotDataset


class PiperRecorder:
    """Piper双臂数据采集器"""
    
    def __init__(
        self,
        repo_id: str,
        root: str | Path,
        fps: int = 30,
        num_cameras: int = 3,
    ):
        self.repo_id = repo_id
        self.root = Path(root)
        self.fps = fps
        self.num_cameras = num_cameras
        
        # ROS数据缓存
        self.latest_joints_left = None
        self.latest_joints_right = None
        self.latest_images = {}
        self.lock = threading.Lock()
        
        # 数据集
        self.dataset = None
        self.features = self._create_features()
    
    def _rosimg_to_numpy(self, msg: Image) -> np.ndarray:
        """手动将ROS Image消息转换为numpy数组（避免cv_bridge依赖问题）"""
        if msg.encoding == "rgb8":
            img = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width, 3)
        elif msg.encoding == "bgr8":
            img = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width, 3)
            img = img[:, :, ::-1].copy()  # BGR转RGB
        elif msg.encoding == "mono8":
            img = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width, 1)
            img = np.repeat(img, 3, axis=2)  # 灰度转RGB
        else:
            # 默认按RGB8处理
            img = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width, 3)
        return img
    
    def _create_features(self) -> dict:
        """创建特征定义"""
        features = {
            # 关节状态 - 左臂 (7个关节)
            "joint_left": {"dtype": "float32", "shape": (7,), "names": ["j0", "j1", "j2", "j3", "j4", "j5", "j6"]},
            # 关节状态 - 右臂 (7个关节)
            "joint_right": {"dtype": "float32", "shape": (7,), "names": ["j0", "j1", "j2", "j3", "j4", "j5", "j6"]},
        }
        
        # 摄像头
        camera_names = ["camera_f", "camera_l", "camera_r"]
        for cam in camera_names[:self.num_cameras]:
            features[f"observation.images.{cam}"] = {
                "dtype": "video",
                "shape": (480, 640, 3),
                "names": ["height", "width", "channel"],
            }
        
        return features
    
    def _init_ros(self):
        """初始化ROS订阅"""
        rospy.init_node('piper_recorder', anonymous=True)
        
        # 关节订阅
        rospy.Subscriber('/puppet/joint_left', JointState, self._cb_joint_left)
        rospy.Subscriber('/puppet/joint_right', JointState, self._cb_joint_right)
        
        # 摄像头订阅
        rospy.Subscriber('/camera_f/color/image_raw', Image, self._cb_camera_f)
        rospy.Subscriber('/camera_l/color/image_raw', Image, self._cb_camera_l)
        rospy.Subscriber('/camera_r/color/image_raw', Image, self._cb_camera_r)
        
        rospy.loginfo("ROS订阅已初始化")
    
    def _cb_joint_left(self, msg: JointState):
        with self.lock:
            self.latest_joints_left = np.array(msg.position, dtype=np.float32)
    
    def _cb_joint_right(self, msg: JointState):
        with self.lock:
            self.latest_joints_right = np.array(msg.position, dtype=np.float32)
    
    def _cb_camera_f(self, msg: Image):
        with self.lock:
            self.latest_images['camera_f'] = self._rosimg_to_numpy(msg)
    
    def _cb_camera_l(self, msg: Image):
        with self.lock:
            self.latest_images['camera_l'] = self._rosimg_to_numpy(msg)
    
    def _cb_camera_r(self, msg: Image):
        with self.lock:
            self.latest_images['camera_r'] = self._rosimg_to_numpy(msg)
    
    def _get_frame(self) -> dict | None:
        """获取一帧数据"""
        with self.lock:
            if (self.latest_joints_left is None or 
                self.latest_joints_right is None or
                len(self.latest_images) < self.num_cameras):
                return None
            
            frame = {
                "joint_left": self.latest_joints_left.copy(),
                "joint_right": self.latest_joints_right.copy(),
            }
            
            for cam in ['camera_f', 'camera_l', 'camera_r'][:self.num_cameras]:
                if cam in self.latest_images:
                    frame[f"observation.images.{cam}"] = self.latest_images[cam].copy()
            
            return frame
    
    def _init_dataset(self):
        """初始化LeRobot数据集"""
        self.dataset = LeRobotDataset.create(
            repo_id=self.repo_id,
            root=self.root,
            fps=self.fps,
            features=self.features,
        )
        print(f"数据集已创建: {self.root / self.repo_id}")
    
    def record_episode(self, task: str = "演示任务") -> int:
        """录制一个episode"""
        print("\n" + "="*50)
        print("准备录制...")
        print("按 Ctrl+C 停止录制")
        print("="*50)
        
        # 等待数据
        print("等待ROS数据...")
        for _ in range(100):  # 最多等待5秒
            frame = self._get_frame()
            if frame is not None:
                break
            time.sleep(0.05)
        
        if frame is None:
            print("错误: 未收到ROS数据!")
            return 0
        
        print("开始录制...")
        frame_count = 0
        dt = 1.0 / self.fps
        
        try:
            while not rospy.is_shutdown():
                start_time = time.time()
                
                frame = self._get_frame()
                if frame is not None:
                    frame["task"] = task
                    self.dataset.add_frame(frame)
                    frame_count += 1
                    
                    if frame_count % 30 == 0:
                        print(f"  已录制 {frame_count} 帧...")
                
                elapsed = time.time() - start_time
                if elapsed < dt:
                    time.sleep(dt - elapsed)
                    
        except KeyboardInterrupt:
            print("\n停止录制...")
        
        # 保存episode
        if frame_count > 0:
            self.dataset.save_episode()
            print(f"Episode已保存: {frame_count} 帧")
        
        return frame_count
    
    def run(self, num_episodes: int = 10, task: str = "机器人操作演示"):
        """运行采集"""
        self._init_ros()
        time.sleep(1.0)  # 等待ROS初始化
        self._init_dataset()
        
        print(f"\n开始采集数据")
        print(f"  目标episodes: {num_episodes}")
        print(f"  保存路径: {self.root / self.repo_id}")
        print(f"  FPS: {self.fps}")
        
        for ep in range(num_episodes):
            print(f"\n[Episode {ep + 1}/{num_episodes}]")
            frames = self.record_episode(task)
            
            if frames == 0:
                print("录制失败，跳过此episode")
                continue
            
            if ep < num_episodes - 1:
                print("\n等待3秒后继续下一个episode...")
                time.sleep(3)
        
        print("\n" + "="*50)
        print("数据采集完成!")
        print(f"数据集路径: {self.root / self.repo_id}")
        print("="*50)


def main():
    parser = argparse.ArgumentParser(description="Piper双臂数据采集")
    parser.add_argument("--repo-id", type=str, required=True, help="数据集名称")
    parser.add_argument("--root", type=str, default="~/my_data", help="数据保存路径")
    parser.add_argument("--num-episodes", type=int, default=10, help="采集episode数量")
    parser.add_argument("--fps", type=int, default=30, help="采集帧率")
    parser.add_argument("--num-cameras", type=int, default=3, help="摄像头数量")
    parser.add_argument("--task", type=str, default="机器人操作演示", help="任务描述")
    
    args = parser.parse_args()
    
    recorder = PiperRecorder(
        repo_id=args.repo_id,
        root=Path(args.root).expanduser(),
        fps=args.fps,
        num_cameras=args.num_cameras,
    )
    
    recorder.run(
        num_episodes=args.num_episodes,
        task=args.task,
    )


if __name__ == "__main__":
    main()