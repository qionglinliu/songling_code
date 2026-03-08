#!/usr/bin/env python3
"""
Piper单臂推理脚本 - 使用训练好的LeRobot ACT模型控制机器人
与piper_recorder_v2.py对齐:
  - 左臂模式
  - observation.state = puppet当前位置
  - action = 发布到master目标位置
  - 摄像头: camera_f, camera_l
"""

import argparse
import time
import threading
from pathlib import Path

import numpy as np
import torch
import rospy
from PIL import Image as PILImage
from sensor_msgs.msg import JointState, Image
from std_msgs.msg import Header, Bool


class PiperInferenceV2:
    """Piper单臂推理控制器 - 左臂"""
    
    def __init__(self, checkpoint_path: str, device: str = "cuda", fps: int = 30):
        self.checkpoint_path = Path(checkpoint_path)
        self.device = device
        self.fps = fps
        
        # ROS数据缓存
        self.puppet_joints = None   # 从臂位置 → observation.state
        self.latest_images = {}
        self.lock = threading.Lock()
        
        # 摄像头配置 - 与recorder_v2一致
        self.camera_names = ["camera_f", "camera_l"]
        
        # 模型
        self.policy = None
        self.preprocessor = None
        self.postprocessor = None
        
        # 运行状态
        self.running = False
        self.image_size = (480, 640)
        
        # 归一化参数 (从stats.json加载)
        self.state_mean = None
        self.state_std = None
        self.action_mean = None
        self.action_std = None
        
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
    
    def _init_ros(self):
        """初始化ROS - 左臂模式"""
        rospy.init_node('piper_inference_v2', anonymous=True)
        
        # 订阅从臂位置 (observation.state)
        rospy.Subscriber('/puppet/joint_left', JointState, self._cb_puppet_joint)
        
        # 订阅摄像头
        rospy.Subscriber('/camera_f/color/image_raw', Image, self._cb_camera_f)
        rospy.Subscriber('/camera_l/color/image_raw', Image, self._cb_camera_l)
        
        # 发布到主臂 (action)
        self.pub_master = rospy.Publisher('/master/joint_left', JointState, queue_size=10)
        
        # 发布使能信号
        self.pub_enable = rospy.Publisher('/enable_flag', Bool, queue_size=1, latch=True)
        
        # 关节限位 (根据训练数据范围设置)
        self.joint_limits = {
            'min': np.array([-0.5, -0.5, -1.2, -0.6, -0.3, -0.5, 0.0], dtype=np.float32),
            'max': np.array([0.5, 1.8, 0.2, 0.2, 0.6, 0.6, 0.12], dtype=np.float32),
        }
        
        rospy.loginfo("ROS初始化完成 - 左臂模式")
        rospy.loginfo("  订阅 puppet/joint_left → observation.state")
        rospy.loginfo("  发布 master/joint_left ← action")
        rospy.loginfo(f"  摄像头: {self.camera_names}")
    
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
    
    def load_model(self):
        """加载模型和归一化参数"""
        print(f"加载模型: {self.checkpoint_path}")
        
        # 检查路径是否存在
        if not self.checkpoint_path.exists():
            raise FileNotFoundError(f"模型路径不存在: {self.checkpoint_path}")
        
        # 尝试使用LeRobot原生API加载
        try:
            from lerobot.policies.act.modeling_act import ACTPolicy
            from lerobot.processor.pipeline import DataProcessorPipeline
            
            # 加载策略模型
            self.policy = ACTPolicy.from_pretrained(str(self.checkpoint_path))
            self.policy.to(self.device)
            self.policy.eval()
            
            # 加载 preprocessor (用于归一化输入)
            preprocessor_path = self.checkpoint_path / "policy_preprocessor.json"
            if preprocessor_path.exists():
                self.preprocessor = DataProcessorPipeline.from_pretrained(
                    str(self.checkpoint_path),
                    config_filename="policy_preprocessor.json",
                )
                # DataProcessorPipeline 没有 .to() 方法，processor在CPU上运行
                print(f"  Preprocessor loaded: {len(self.preprocessor.steps)} steps")
            else:
                print("  Warning: policy_preprocessor.json not found")
            
            # 加载 postprocessor (用于反归一化输出)
            postprocessor_path = self.checkpoint_path / "policy_postprocessor.json"
            if postprocessor_path.exists():
                self.postprocessor = DataProcessorPipeline.from_pretrained(
                    str(self.checkpoint_path),
                    config_filename="policy_postprocessor.json",
                )
                # DataProcessorPipeline 没有 .to() 方法，processor在CPU上运行
                print(f"  Postprocessor loaded: {len(self.postprocessor.steps)} steps")
            else:
                print("  Warning: policy_postprocessor.json not found")
                
        except Exception as e:
            print(f"加载LeRobot模型失败: {e}")
            raise
        
        # 加载归一化统计参数 (备用)
        stats_path = self.checkpoint_path.parent.parent / "meta" / "stats.json"
        if stats_path.exists():
            import json
            with open(stats_path, 'r') as f:
                stats = json.load(f)
            self.state_mean = np.array(stats["observation.state"]["mean"], dtype=np.float32)
            self.state_std = np.array(stats["observation.state"]["std"], dtype=np.float32)
            self.action_mean = np.array(stats["action"]["mean"], dtype=np.float32)
            self.action_std = np.array(stats["action"]["std"], dtype=np.float32)
            print(f"  Stats loaded from: {stats_path}")
        else:
            print(f"  Warning: stats.json not found at {stats_path}")
        
        print("模型加载完成!")
    
    def _normalize_state(self, state: torch.Tensor) -> torch.Tensor:
        """归一化状态 (如果preprocessor不存在则手动归一化)"""
        if self.preprocessor is not None:
            return state  # 由preprocessor处理
        if self.state_mean is not None:
            return (state - torch.from_numpy(self.state_mean).to(state.device)) / (torch.from_numpy(self.state_std).to(state.device) + 1e-8)
        return state
    
    def _denormalize_action(self, action: torch.Tensor) -> torch.Tensor:
        """反归一化动作 (如果postprocessor不存在则手动反归一化)"""
        if self.postprocessor is not None:
            return action  # 由postprocessor处理
        if self.action_mean is not None:
            return action * torch.from_numpy(self.action_std).to(action.device) + torch.from_numpy(self.action_mean).to(action.device)
        return action
    
    def _get_observation(self) -> dict | None:
        """获取观测数据"""
        with self.lock:
            if self.puppet_joints is None:
                return None
            if len(self.latest_images) < len(self.camera_names):
                return None
            
            # observation.state: 从臂当前状态 (7维)
            state_tensor = torch.from_numpy(self.puppet_joints.copy()).float().unsqueeze(0).to(self.device)
            
            obs = {"observation.state": state_tensor}
            
            # 添加图像
            for cam in self.camera_names:
                if cam in self.latest_images:
                    img = self.latest_images[cam]
                    # 调整大小
                    pil_img = PILImage.fromarray(img)
                    pil_img = pil_img.resize((self.image_size[1], self.image_size[0]), PILImage.BILINEAR)
                    img = np.array(pil_img)
                    # 转换为tensor: HWC -> CHW, uint8 -> float32 [0,1]
                    img_tensor = torch.from_numpy(img.copy()).permute(2, 0, 1).float() / 255.0
                    img_tensor = img_tensor.unsqueeze(0).to(self.device)
                    obs[f"observation.images.{cam}"] = img_tensor
            
            return obs
    
    def _clamp_action(self, action: np.ndarray) -> np.ndarray:
        """限幅动作到安全范围"""
        return np.clip(action, self.joint_limits['min'], self.joint_limits['max'])
    
    def _publish_action(self, action: np.ndarray):
        """发布动作到主臂 (7维)"""
        # 限幅
        action = self._clamp_action(action)
        
        msg = JointState()
        msg.header = Header()
        msg.header.stamp = rospy.Time.now()
        msg.name = [f"left_j{i}" for i in range(7)]
        msg.position = action.tolist()
        self.pub_master.publish(msg)
    
    def _publish_enable(self, enable: bool = True):
        """发布使能信号"""
        msg = Bool()
        msg.data = enable
        self.pub_enable.publish(msg)
        rospy.loginfo(f"发布使能信号: {enable}")
    
    def run(self, duration: float = 60.0):
        """运行推理"""
        self._init_ros()
        self.load_model()
        
        print(f"\n" + "="*50)
        print("开始推理控制")
        print("="*50)
        print(f"  持续时间: {duration} 秒")
        print(f"  控制频率: {self.fps} Hz")
        print(f"  控制模式: 左臂 (7维)")
        print(f"  摄像头: {self.camera_names}")
        print("="*50)
        print("等待ROS数据...")
        
        # 等待数据
        obs = None
        for i in range(100):
            obs = self._get_observation()
            if obs is not None:
                print(f"数据就绪! (等待 {i+1} 次)")
                break
            rospy.sleep(0.1)
        
        if obs is None:
            print("错误: 未收到ROS数据!")
            print(f"  从臂关节数据: {self.puppet_joints is not None}")
            print(f"  图像数据: {list(self.latest_images.keys())}")
            return
        
        self.running = True
        dt = 1.0 / self.fps
        start_time = time.time()
        step_count = 0
        
        # 动作缓存 (ACT输出chunk)
        action_queue = []
        
        # 发布使能信号
        self._publish_enable(True)
        time.sleep(0.5)  # 等待使能生效
        
        print("\n开始执行...")
        
        try:
            while self.running and (time.time() - start_time) < duration:
                loop_start = time.time()
                
                obs = self._get_observation()
                if obs is None:
                    time.sleep(0.01)
                    continue
                
                with torch.no_grad():
                    # 如果动作队列空了，重新推理
                    if len(action_queue) == 0:
                        # 1. 预处理 (归一化)
                        if self.preprocessor is not None:
                            preprocessed = self.preprocessor(obs)
                        else:
                            preprocessed = obs.copy()
                            preprocessed["observation.state"] = self._normalize_state(preprocessed["observation.state"])
                        
                        # 2. 策略推理
                        action_chunk = self.policy.select_action(preprocessed)
                        
                        # 3. 后处理 (反归一化)
                        if self.postprocessor is not None:
                            postprocessed = self.postprocessor({"action": action_chunk})
                            action_chunk = postprocessed["action"]
                        else:
                            action_chunk = self._denormalize_action(action_chunk)
                        
                        # chunk shape: [batch, chunk_size, action_dim] 或 [batch, action_dim]
                        action_np = action_chunk.squeeze(0).cpu().numpy()
                        
                        # 如果是chunk，展开到队列
                        if len(action_np.shape) == 2:
                            action_queue.extend(action_np.tolist())
                        else:
                            action_queue.append(action_np.tolist())
                    
                    # 取出第一个动作执行
                    if len(action_queue) > 0:
                        action = np.array(action_queue.pop(0), dtype=np.float32)
                        
                        if step_count % 30 == 0:
                            print(f"Step {step_count}: action = {action.round(4)}")
                        
                        self._publish_action(action)
                        step_count += 1
                
                elapsed = time.time() - loop_start
                if elapsed < dt:
                    time.sleep(dt - elapsed)
                    
        except KeyboardInterrupt:
            print("\n用户中断")
        
        self.running = False
        print(f"\n推理完成, 总步数: {step_count}")


def main():
    parser = argparse.ArgumentParser(description="Piper单臂推理脚本 (左臂)")
    parser.add_argument(
        "--checkpoint", 
        default="/home/agilex/robot/code/code/outputs/train/piper_act/checkpoints/035000/pretrained_model",
        help="模型路径"
    )
    parser.add_argument("--device", default="cuda", help="设备")
    parser.add_argument("--fps", type=int, default=30, help="控制频率")
    parser.add_argument("--duration", type=float, default=60.0, help="运行时长(秒)")
    
    args = parser.parse_args()
    
    # 支持绝对路径和相对路径
    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.is_absolute():
        checkpoint_path = Path(__file__).parent / args.checkpoint
    
    inference = PiperInferenceV2(
        checkpoint_path=str(checkpoint_path),
        device=args.device,
        fps=args.fps,
    )
    inference.run(duration=args.duration)


if __name__ == "__main__":
    main()