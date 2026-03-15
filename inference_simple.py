#!/usr/bin/env python3
"""
Piper推理脚本 - 简化版 (基于 piper_inference_v2.py)
适用于 n_action_steps=1 的训练配置

特点:
  - 简单直接，EMA平滑
  - 支持左臂/右臂/双臂
  - 30-40Hz 控制频率

使用方法:
  # 左臂
  python inference_simple.py --checkpoint outputs_3_12_banana/train/piper_act/checkpoints/050000/pretrained_model --arm left
  
  # 右臂
  python inference_simple.py --checkpoint outputs_3_12_mango/train/piper_act/checkpoints/010000/pretrained_model --arm right
  
  # 双臂
  python inference_simple.py --checkpoint outputs_dual/train/piper_act/checkpoints/050000/pretrained_model --arm both
"""

import argparse
import time
import threading
from pathlib import Path

import numpy as np
import torch
import cv2
import rospy
from sensor_msgs.msg import JointState, Image
from std_msgs.msg import Header, Bool


class PiperInferenceSimple:
    """Piper推理控制器 - 简化版 (EMA平滑)"""
    
    def __init__(
        self,
        checkpoint_path: str,
        arm: str = "left",
        device: str = "cuda",
        fps: int = 30,
    ):
        self.checkpoint_path = Path(checkpoint_path)
        self.arm = arm
        self.device = device
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
        
        # 摄像头配置
        if self.is_dual_arm:
            self.camera_names = ["camera_f", "camera_l", "camera_r"]
        elif self.is_left_arm:
            self.camera_names = ["camera_f", "camera_l"]
        else:
            self.camera_names = ["camera_f", "camera_r"]
        
        # ROS数据缓存
        self.puppet_left_joints = None
        self.puppet_right_joints = None
        self.latest_images = {}
        self.lock = threading.Lock()
        
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
        
        # 动作平滑参数
        self.smoothing_alpha = 0.3  # EMA平滑系数 (0=完全平滑, 1=不平滑)
        self.max_action_change = 0.08  # 每帧最大动作变化量 (rad)
        self.last_action = None  # 上一次动作
        
        # 帧率监控
        self.frame_times = []
        
        # 关节限位 (根据数据集stats.json调整)
        self.joint_limits = {
            'min': np.array([-2.618, 0, -2.967, -1.832, -1.22, -3.14, 0.0], dtype=np.float32),
            'max': np.array([2.618, 3.14, 0, 1.832, 1.22, 3.14, 0.076], dtype=np.float32),
        }
    
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
        """初始化ROS"""
        rospy.init_node('piper_inference_simple', anonymous=True)
        
        # 订阅从臂位置 (observation.state)
        if self.is_left_arm or self.is_dual_arm:
            rospy.Subscriber('/puppet/joint_left', JointState, self._cb_puppet_left)
        if self.is_right_arm or self.is_dual_arm:
            rospy.Subscriber('/puppet/joint_right', JointState, self._cb_puppet_right)
        
        # 订阅摄像头
        rospy.Subscriber('/camera_f/color/image_raw', Image, self._cb_camera_f)
        if self.is_left_arm or self.is_dual_arm:
            rospy.Subscriber('/camera_l/color/image_raw', Image, self._cb_camera_l)
        if self.is_right_arm or self.is_dual_arm:
            rospy.Subscriber('/camera_r/color/image_raw', Image, self._cb_camera_r)
        
        # 发布到主臂 (action)
        if self.is_left_arm or self.is_dual_arm:
            self.pub_master_left = rospy.Publisher('/master/joint_left', JointState, queue_size=10)
        if self.is_right_arm or self.is_dual_arm:
            self.pub_master_right = rospy.Publisher('/master/joint_right', JointState, queue_size=10)
        
        # 发布使能信号
        self.pub_enable = rospy.Publisher('/enable_flag', Bool, queue_size=1, latch=True)
        
        arm_type = "双臂" if self.is_dual_arm else ("左臂" if self.is_left_arm else "右臂")
        rospy.loginfo(f"ROS初始化完成 - {arm_type}模式")
    
    def _cb_puppet_left(self, msg: JointState):
        with self.lock:
            self.puppet_left_joints = np.array(msg.position, dtype=np.float32)
    
    def _cb_puppet_right(self, msg: JointState):
        with self.lock:
            self.puppet_right_joints = np.array(msg.position, dtype=np.float32)
    
    def _cb_camera_f(self, msg: Image):
        with self.lock:
            self.latest_images['camera_f'] = self._rosimg_to_numpy(msg)
    
    def _cb_camera_l(self, msg: Image):
        with self.lock:
            self.latest_images['camera_l'] = self._rosimg_to_numpy(msg)
    
    def _cb_camera_r(self, msg: Image):
        with self.lock:
            self.latest_images['camera_r'] = self._rosimg_to_numpy(msg)
    
    def load_model(self):
        """加载模型"""
        print(f"加载模型: {self.checkpoint_path}")
        
        if not self.checkpoint_path.exists():
            raise FileNotFoundError(f"模型路径不存在: {self.checkpoint_path}")
        
        import os
        os.environ['LEROBOT_VIDEO_BACKEND'] = 'pyav'
        
        try:
            from lerobot.policies.act.modeling_act import ACTPolicy
            from lerobot.processor.pipeline import DataProcessorPipeline
            
            self.policy = ACTPolicy.from_pretrained(str(self.checkpoint_path))
            self.policy.to(self.device)
            self.policy.eval()
            
            preprocessor_path = self.checkpoint_path / "policy_preprocessor.json"
            if preprocessor_path.exists():
                self.preprocessor = DataProcessorPipeline.from_pretrained(
                    str(self.checkpoint_path),
                    config_filename="policy_preprocessor.json",
                )
                print(f"  Preprocessor loaded")
            
            postprocessor_path = self.checkpoint_path / "policy_postprocessor.json"
            if postprocessor_path.exists():
                self.postprocessor = DataProcessorPipeline.from_pretrained(
                    str(self.checkpoint_path),
                    config_filename="policy_postprocessor.json",
                )
                print(f"  Postprocessor loaded")
                
        except Exception as e:
            print(f"加载模型失败: {e}")
            raise
        
        # 加载归一化统计参数
        stats_path = self.checkpoint_path.parent.parent.parent.parent / "meta" / "stats.json"
        if stats_path.exists():
            import json
            with open(stats_path, 'r') as f:
                stats = json.load(f)
            self.state_mean = np.array(stats["observation.state"]["mean"], dtype=np.float32)
            self.state_std = np.array(stats["observation.state"]["std"], dtype=np.float32)
            self.action_mean = np.array(stats["action"]["mean"], dtype=np.float32)
            self.action_std = np.array(stats["action"]["std"], dtype=np.float32)
        
        print("模型加载完成!")
    
    def _normalize_state(self, state: torch.Tensor) -> torch.Tensor:
        if self.preprocessor is not None:
            return state
        if self.state_mean is not None:
            return (state - torch.from_numpy(self.state_mean).to(state.device)) / (torch.from_numpy(self.state_std).to(state.device) + 1e-8)
        return state
    
    def _denormalize_action(self, action: torch.Tensor) -> torch.Tensor:
        if self.postprocessor is not None:
            return action
        if self.action_mean is not None:
            return action * torch.from_numpy(self.action_std).to(action.device) + torch.from_numpy(self.action_mean).to(action.device)
        return action
    
    def _get_observation(self) -> dict | None:
        """获取观测数据"""
        with self.lock:
            if self.is_dual_arm:
                if self.puppet_left_joints is None or self.puppet_right_joints is None:
                    return None
                state = np.concatenate([self.puppet_left_joints, self.puppet_right_joints])
            elif self.is_left_arm:
                if self.puppet_left_joints is None:
                    return None
                state = self.puppet_left_joints.copy()
            else:
                if self.puppet_right_joints is None:
                    return None
                state = self.puppet_right_joints.copy()
            
            for cam in self.camera_names:
                if cam not in self.latest_images:
                    return None
            
            state_tensor = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
            obs = {"observation.state": state_tensor}
            
            for cam in self.camera_names:
                img = self.latest_images[cam]
                img = cv2.resize(img, (self.image_size[1], self.image_size[0]), interpolation=cv2.INTER_LINEAR)
                img_tensor = torch.from_numpy(img.copy()).permute(2, 0, 1).float() / 255.0
                img_tensor = img_tensor.unsqueeze(0).to(self.device)
                obs[f"observation.images.{cam}"] = img_tensor
            
            return obs
    
    def _clamp_action(self, action: np.ndarray) -> np.ndarray:
        if self.is_dual_arm:
            left = np.clip(action[:7], self.joint_limits['min'], self.joint_limits['max'])
            right = np.clip(action[7:], self.joint_limits['min'], self.joint_limits['max'])
            return np.concatenate([left, right])
        return np.clip(action, self.joint_limits['min'], self.joint_limits['max'])
    
    def _smooth_action(self, action: np.ndarray) -> np.ndarray:
        """EMA平滑 + 速度限制"""
        if self.last_action is None:
            self.last_action = action.copy()
            return action
        
        # EMA平滑
        smoothed = self.smoothing_alpha * action + (1 - self.smoothing_alpha) * self.last_action
        
        # 速度限制
        delta = smoothed - self.last_action
        delta = np.clip(delta, -self.max_action_change, self.max_action_change)
        smoothed = self.last_action + delta
        
        self.last_action = smoothed.copy()
        return smoothed
    
    def _publish_action(self, action: np.ndarray):
        action = self._clamp_action(action)
        
        if self.is_dual_arm:
            left, right = action[:7], action[7:]
            
            msg_l = JointState()
            msg_l.header = Header()
            msg_l.header.stamp = rospy.Time.now()
            msg_l.name = [f"left_j{i}" for i in range(7)]
            msg_l.position = left.tolist()
            self.pub_master_left.publish(msg_l)
            
            msg_r = JointState()
            msg_r.header = Header()
            msg_r.header.stamp = rospy.Time.now()
            msg_r.name = [f"right_j{i}" for i in range(7)]
            msg_r.position = right.tolist()
            self.pub_master_right.publish(msg_r)
            
        elif self.is_left_arm:
            msg = JointState()
            msg.header = Header()
            msg.header.stamp = rospy.Time.now()
            msg.name = [f"left_j{i}" for i in range(7)]
            msg.position = action.tolist()
            self.pub_master_left.publish(msg)
        else:
            msg = JointState()
            msg.header = Header()
            msg.header.stamp = rospy.Time.now()
            msg.name = [f"right_j{i}" for i in range(7)]
            msg.position = action.tolist()
            self.pub_master_right.publish(msg)
    
    def _publish_enable(self, enable: bool = True):
        msg = Bool()
        msg.data = enable
        self.pub_enable.publish(msg)
        rospy.loginfo(f"发布使能信号: {enable}")
    
    def _get_avg_fps(self) -> float:
        if len(self.frame_times) < 2:
            return 0.0
        avg_dt = sum(self.frame_times) / len(self.frame_times)
        return 1.0 / avg_dt if avg_dt > 0 else 0.0
    
    def run(self, duration: float = 60.0):
        """运行推理 - 每帧推理，只取第一个动作"""
        self._init_ros()
        self.load_model()
        
        arm_type = "双臂" if self.is_dual_arm else ("左臂" if self.is_left_arm else "右臂")
        
        print(f"\n" + "="*50)
        print(f"Piper推理控制 - 简化版 ({arm_type})")
        print("="*50)
        print(f"  持续时间: {duration} 秒")
        print(f"  控制频率: {self.fps} Hz")
        print(f"  平滑系数: {self.smoothing_alpha}")
        print(f"  模式: 每帧推理 (无队列)")
        print("="*50)
        print("等待ROS数据...")
        
        # 等待数据
        obs = None
        for i in range(100):
            obs = self._get_observation()
            if obs is not None:
                print(f"数据就绪!")
                break
            rospy.sleep(0.1)
        
        if obs is None:
            print("错误: 未收到ROS数据!")
            return
        
        self.running = True
        dt = 1.0 / self.fps
        start_time = time.time()
        step_count = 0
        
        # 发布使能
        self._publish_enable(True)
        time.sleep(0.5)
        
        print("\n开始执行...")
        
        try:
            while self.running and (time.time() - start_time) < duration:
                loop_start = time.time()
                
                obs = self._get_observation()
                if obs is None:
                    time.sleep(0.001)
                    continue
                
                # 每帧都推理，只取第一个动作
                with torch.no_grad():
                    # 预处理
                    if self.preprocessor is not None:
                        preprocessed = self.preprocessor(obs)
                    else:
                        preprocessed = obs.copy()
                        preprocessed["observation.state"] = self._normalize_state(preprocessed["observation.state"])
                    
                    # 推理
                    action_chunk = self.policy.select_action(preprocessed)
                    
                    # 后处理
                    if self.postprocessor is not None:
                        postprocessed = self.postprocessor({"action": action_chunk})
                        action_chunk = postprocessed["action"]
                    else:
                        action_chunk = self._denormalize_action(action_chunk)
                    
                    action_np = action_chunk.squeeze(0).cpu().numpy()
                    
                    # 只取第一个动作（当前步的动作）
                    if len(action_np.shape) == 2:
                        action = action_np[0].astype(np.float32)
                    else:
                        action = action_np.astype(np.float32)
                
                # 平滑并执行
                action = self._smooth_action(action)
                
                if step_count % 30 == 0:
                    avg_fps = self._get_avg_fps()
                    gripper = action[6] if len(action) > 6 else 0
                    print(f"Step {step_count}: joints = {action[:6].round(4)}, gripper = {gripper:.4f}, FPS = {avg_fps:.1f}")
                
                self._publish_action(action)
                step_count += 1
                
                elapsed = time.time() - loop_start
                self.frame_times.append(elapsed)
                if len(self.frame_times) > 100:
                    self.frame_times.pop(0)
                
                if elapsed < dt:
                    time.sleep(dt - elapsed)
                    
        except KeyboardInterrupt:
            print("\n用户中断")
        
        self.running = False
        avg_fps = self._get_avg_fps()
        print(f"\n推理完成, 总步数: {step_count}, 平均帧率: {avg_fps:.1f} Hz")


def main():
    parser = argparse.ArgumentParser(description="Piper推理脚本 - 简化版")
    parser.add_argument("--checkpoint", required=True, help="模型路径")
    parser.add_argument("--arm", type=str, default="left", choices=["left", "right", "both"], help="手臂选择")
    parser.add_argument("--device", default="cuda", help="设备")
    parser.add_argument("--fps", type=int, default=30, help="控制频率")
    parser.add_argument("--duration", type=float, default=300.0, help="运行时长(秒)")
    parser.add_argument("--smoothing", type=float, default=0.3, help="平滑系数 (0=完全平滑, 1=不平滑)")
    parser.add_argument("--max-change", type=float, default=0.08, help="每帧最大动作变化量 (rad)")
    
    args = parser.parse_args()
    
    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.is_absolute():
        checkpoint_path = Path(__file__).parent / args.checkpoint
    
    inference = PiperInferenceSimple(
        checkpoint_path=str(checkpoint_path),
        arm=args.arm,
        device=args.device,
        fps=args.fps,
    )
    inference.smoothing_alpha = args.smoothing
    inference.max_action_change = args.max_change
    
    inference.run(duration=args.duration)


if __name__ == "__main__":
    main()