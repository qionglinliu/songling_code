#!/usr/bin/env python3
"""
Piper单臂数据采集脚本 - LeRobot格式
参照松灵ACT方案设计：
  - observation.state = 从臂当前位置 (puppet)
  - action = 主臂目标位置 (master)
  
使用方法:
  python piper_recorder_v2.py --repo-id my_data --num-episodes 10
"""

import argparse
import time
import threading
import sys
import termios
import tty
from pathlib import Path

import numpy as np
import rospy
from sensor_msgs.msg import JointState, Image

# LeRobot API
from lerobot.datasets.lerobot_dataset import LeRobotDataset


class PiperRecorder:
    """Piper单臂数据采集器 - 左臂"""
    
    def __init__(
        self,
        repo_id: str,
        root: str | Path,
        fps: int = 30,
    ):
        self.repo_id = repo_id
        self.root = Path(root)
        self.fps = fps
        
        # ROS数据缓存
        self.master_joints = None   # 主臂位置 → action
        self.puppet_joints = None   # 从臂位置 → observation.state
        self.latest_images = {}
        self.lock = threading.Lock()
        
        # 录制控制
        self.recording = False
        self.stop_current_episode = False
        
        # 摄像头配置 - 顶部相机 + 左臂相机
        self.camera_names = ["camera_f", "camera_l"]
        
        # 数据集
        self.dataset = None
        self.features = self._create_features()
    
    def _wait_for_key(self, prompt: str = "按回车键继续...") -> bool:
        """等待用户按键，返回True表示继续，False表示退出"""
        print(prompt)
        try:
            old_settings = termios.tcgetattr(sys.stdin)
            try:
                tty.setraw(sys.stdin.fileno())
                ch = sys.stdin.read(1)
                if ch == '\x03' or ch == 'q':
                    return False
                return True
            finally:
                termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_settings)
        except Exception:
            try:
                input()
                return True
            except KeyboardInterrupt:
                return False
    
    def _rosimg_to_numpy(self, msg: Image) -> np.ndarray:
        """将ROS Image消息转换为numpy数组"""
        if msg.encoding == "rgb8":
            img = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width, 3)
        elif msg.encoding == "bgr8":
            img = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width, 3)
            img = img[:, :, ::-1].copy()  # BGR转RGB
        elif msg.encoding == "mono8":
            img = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width, 1)
            img = np.repeat(img, 3, axis=2)  # 灰度转RGB
        else:
            img = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width, 3)
        return img
    
    def _create_features(self) -> dict:
        """创建特征定义 - LeRobot标准格式"""
        features = {
            # 机器人状态 (observation.state) - 从臂当前位置
            # 这是模型的输入，告诉模型机器人当前在哪里
            "observation.state": {
                "dtype": "float32", 
                "shape": (7,), 
                "names": ["joint0", "joint1", "joint2", "joint3", "joint4", "joint5", "joint6"]
            },
            # 动作 (action) - 主臂目标位置
            # 这是模型的输出，模型要预测人想让机器人去哪里
            "action": {
                "dtype": "float32", 
                "shape": (7,), 
                "names": ["joint0", "joint1", "joint2", "joint3", "joint4", "joint5", "joint6"]
            },
        }
        
        # 摄像头
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
        
        # 主臂关节订阅 → action
        rospy.Subscriber('/master/joint_left', JointState, self._cb_master_joint)
        
        # 从臂关节订阅 → observation.state
        rospy.Subscriber('/puppet/joint_left', JointState, self._cb_puppet_joint)
        
        # 摄像头订阅
        rospy.Subscriber('/camera_f/color/image_raw', Image, self._cb_camera_f)
        rospy.Subscriber('/camera_l/color/image_raw', Image, self._cb_camera_l)
        
        rospy.loginfo("ROS订阅已初始化")
        rospy.loginfo("  主臂话题: /master/joint_left → action")
        rospy.loginfo("  从臂话题: /puppet/joint_left → observation.state")
        rospy.loginfo("  相机话题: /camera_f, /camera_l")
    
    def _cb_master_joint(self, msg: JointState):
        """主臂回调 - 作为action"""
        with self.lock:
            self.master_joints = np.array(msg.position, dtype=np.float32)
    
    def _cb_puppet_joint(self, msg: JointState):
        """从臂回调 - 作为observation.state"""
        with self.lock:
            self.puppet_joints = np.array(msg.position, dtype=np.float32)
    
    def _cb_camera_f(self, msg: Image):
        with self.lock:
            self.latest_images['camera_f'] = self._rosimg_to_numpy(msg)
    
    def _cb_camera_l(self, msg: Image):
        with self.lock:
            self.latest_images['camera_l'] = self._rosimg_to_numpy(msg)
    
    def _get_frame(self) -> dict | None:
        """获取一帧数据"""
        with self.lock:
            # 检查数据是否完整
            if (self.master_joints is None or 
                self.puppet_joints is None or
                len(self.latest_images) < len(self.camera_names)):
                return None
            
            frame = {
                # observation.state: 从臂当前状态 (模型输入)
                "observation.state": self.puppet_joints.copy(),
                # action: 主臂目标位置 (模型输出目标)
                "action": self.master_joints.copy(),
            }
            
            # 添加图像
            for cam in self.camera_names:
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
        print("="*50)
        
        # 等待用户按键确认开始
        if not self._wait_for_key("按回车键开始录制 (按q退出)..."):
            print("用户取消录制")
            return -1
        
        # 重置停止标志
        self.stop_current_episode = False
        
        # 等待数据
        print("等待ROS数据...")
        for i in range(100):
            frame = self._get_frame()
            if frame is not None:
                print(f"数据就绪! (等待 {i+1} 次)")
                break
            time.sleep(0.05)
        
        if frame is None:
            print("错误: 未收到完整ROS数据!")
            print(f"  主臂数据: {self.master_joints is not None}")
            print(f"  从臂数据: {self.puppet_joints is not None}")
            print(f"  图像数据: {list(self.latest_images.keys())}")
            return 0
        
        print("开始录制... (按回车键停止录制)")
        frame_count = 0
        dt = 1.0 / self.fps
        
        # 启动监听停止信号的线程
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
                    frame["task"] = task
                    self.dataset.add_frame(frame)
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
        if frame_count > 0:
            self.dataset.save_episode()
            print(f"Episode已保存: {frame_count} 帧")
        
        return frame_count
    
    def run(self, num_episodes: int = 10, task: str = "机器人操作演示"):
        """运行采集"""
        self._init_ros()
        time.sleep(1.0)
        self._init_dataset()
        
        print(f"\n" + "="*50)
        print("Piper单臂数据采集 (左臂)")
        print("="*50)
        print(f"  目标episodes: {num_episodes}")
        print(f"  保存路径: {self.root / self.repo_id}")
        print(f"  FPS: {self.fps}")
        print(f"  摄像头: {self.camera_names}")
        print(f"\n数据说明:")
        print(f"  observation.state = 从臂位置 (puppet)")
        print(f"  action = 主臂位置 (master)")
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
        
        print("\n" + "="*50)
        print("数据采集完成!")
        print(f"  成功录制: {completed_episodes} 个episodes")
        print(f"  数据集路径: {self.root / self.repo_id}")
        print("="*50)


def main():
    parser = argparse.ArgumentParser(description="Piper单臂数据采集 (左臂)")
    parser.add_argument("--repo-id", type=str, required=True, help="数据集名称")
    parser.add_argument("--root", type=str, default="~/my_data", help="数据保存路径")
    parser.add_argument("--num-episodes", type=int, default=10, help="采集episode数量")
    parser.add_argument("--fps", type=int, default=30, help="采集帧率")
    parser.add_argument("--task", type=str, default="机器人操作演示", help="任务描述")
    
    args = parser.parse_args()
    
    recorder = PiperRecorder(
        repo_id=args.repo_id,
        root=Path(args.root).expanduser(),
        fps=args.fps,
    )
    
    recorder.run(
        num_episodes=args.num_episodes,
        task=args.task,
    )


if __name__ == "__main__":
    main()