#!/usr/bin/env python3
"""
Piper双臂推理脚本 - 使用训练好的ACT模型控制真实机器人
"""

import argparse
import time
import threading
from pathlib import Path

import numpy as np
import torch
import rospy
from sensor_msgs.msg import JointState, Image
from std_msgs.msg import Header


class PiperInference:
    """Piper双臂推理控制器"""
    
    def __init__(
        self,
        checkpoint_path: str,
        device: str = "cuda",
        fps: int = 30,
    ):
        self.checkpoint_path = Path(checkpoint_path)
        self.device = device
        self.fps = fps
        
        # ROS数据缓存
        self.latest_joints_left = None
        self.latest_joints_right = None
        self.latest_images = {}
        self.lock = threading.Lock()
        
        # 摄像头名称
        self.camera_names = ["camera_f", "camera_r"]
        
        # 模型
        self.policy = None
        self.running = False
        
        # 数据集统计信息
        self.dataset_stats = None
        
    def _rosimg_to_numpy(self, msg: Image) -> np.ndarray:
        """将ROS Image消息转换为numpy数组"""
        if msg.encoding == "rgb8":
            img = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width, 3)
        elif msg.encoding == "bgr8":
            img = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width, 3)
            img = img[:, :, ::-1].copy()
        elif msg.encoding == "mono8":
            img = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width, 1)
            img = np.repeat(img, 3, axis=2)
        else:
            img = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width, 3)
        return img
    
    def _init_ros(self):
        """初始化ROS"""
        rospy.init_node('piper_inference', anonymous=True)
        
        rospy.Subscriber('/puppet/joint_left', JointState, self._cb_joint_left)
        rospy.Subscriber('/puppet/joint_right', JointState, self._cb_joint_right)
        rospy.Subscriber('/camera_f/color/image_raw', Image, self._cb_camera_f)
        rospy.Subscriber('/camera_r/color/image_raw', Image, self._cb_camera_r)
        
        self.pub_left = rospy.Publisher('/master/joint_left', JointState, queue_size=10)
        self.pub_right = rospy.Publisher('/master/joint_right', JointState, queue_size=10)
        
        rospy.loginfo("ROS初始化完成")
    
    def _cb_joint_left(self, msg: JointState):
        with self.lock:
            self.latest_joints_left = np.array(msg.position, dtype=np.float32)
    
    def _cb_joint_right(self, msg: JointState):
        with self.lock:
            self.latest_joints_right = np.array(msg.position, dtype=np.float32)
    
    def _cb_camera_f(self, msg: Image):
        with self.lock:
            self.latest_images['camera_f'] = self._rosimg_to_numpy(msg)
    
    def _cb_camera_r(self, msg: Image):
        with self.lock:
            self.latest_images['camera_r'] = self._rosimg_to_numpy(msg)
    
    def load_model(self):
        """加载训练好的模型"""
        print(f"加载模型: {self.checkpoint_path}")
        
        import json
        from lerobot.policies.act.modeling_act import ACTPolicy
        from lerobot.policies.act.configuration_act import ACTConfig
        from safetensors.torch import load_file
        from lerobot.configs.types import FeatureType, PolicyFeature, NormalizationMode
        
        # 加载数据集统计信息
        stats_path = Path(__file__).parent / "my_data_i" / "meta" / "stats.json"
        print(f"查找数据集统计信息: {stats_path}")
        if stats_path.exists():
            with open(stats_path) as f:
                self.dataset_stats = json.load(f)
            print("已加载数据集统计信息")
            action_mean = self.dataset_stats["action"]["mean"]
            print(f"左臂均值(用于固定): {np.array(action_mean[:7]).round(3)}")
        else:
            self.dataset_stats = None
            print(f"警告: 未找到数据集统计信息: {stats_path}")
        
        # 加载配置文件
        config_path = self.checkpoint_path / "config.json"
        with open(config_path) as f:
            config_dict = json.load(f)
        
        config_dict.pop("type", None)
        
        if "input_features" in config_dict and config_dict["input_features"]:
            input_features = {}
            for key, val in config_dict["input_features"].items():
                input_features[key] = PolicyFeature(
                    type=FeatureType[val["type"]],
                    shape=tuple(val["shape"])
                )
            config_dict["input_features"] = input_features
        
        if "output_features" in config_dict and config_dict["output_features"]:
            output_features = {}
            for key, val in config_dict["output_features"].items():
                output_features[key] = PolicyFeature(
                    type=FeatureType[val["type"]],
                    shape=tuple(val["shape"])
                )
            config_dict["output_features"] = output_features
        
        if "normalization_mapping" in config_dict and config_dict["normalization_mapping"]:
            norm_map = {}
            for key, val in config_dict["normalization_mapping"].items():
                norm_map[key] = NormalizationMode[val]
            config_dict["normalization_mapping"] = norm_map
        
        config = ACTConfig(**config_dict)
        self.policy = ACTPolicy(config)
        
        model_path = self.checkpoint_path / "model.safetensors"
        state_dict = load_file(str(model_path))
        self.policy.load_state_dict(state_dict)
        
        preprocessor_path = self.checkpoint_path / "policy_preprocessor_step_3_normalizer_processor.safetensors"
        postprocessor_path = self.checkpoint_path / "policy_postprocessor_step_0_unnormalizer_processor.safetensors"
        
        if preprocessor_path.exists():
            preprocessor_weights = load_file(str(preprocessor_path))
            if hasattr(self.policy, 'preprocessor') and 'normalizer' in self.policy.preprocessor:
                self.policy.preprocessor['normalizer'].load_state_dict(preprocessor_weights)
        
        if postprocessor_path.exists():
            postprocessor_weights = load_file(str(postprocessor_path))
            if hasattr(self.policy, 'postprocessor') and 'unnormalizer' in self.policy.postprocessor:
                self.policy.postprocessor['unnormalizer'].load_state_dict(postprocessor_weights)
        
        self.policy.to(self.device)
        self.policy.eval()
        print("模型加载完成!")
    
    def _get_observation(self) -> dict | None:
        """获取当前观测"""
        with self.lock:
            if (self.latest_joints_left is None or 
                self.latest_joints_right is None or
                len(self.latest_images) < len(self.camera_names)):
                return None
            
            joints_combined = np.concatenate([self.latest_joints_left, self.latest_joints_right])
            state_tensor = torch.from_numpy(joints_combined.copy()).float().unsqueeze(0).to(self.device)
            
            if hasattr(self.policy, 'preprocessor') and 'normalizer' in self.policy.preprocessor:
                state_tensor = self.policy.preprocessor['normalizer'](
                    state_tensor, key="observation.state"
                )
            
            obs = {"observation.state": state_tensor}
            
            for cam in self.camera_names:
                if cam in self.latest_images:
                    img = self.latest_images[cam]
                    img_tensor = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
                    img_tensor = img_tensor.unsqueeze(0).to(self.device)
                    obs[f"observation.images.{cam}"] = img_tensor
            
            return obs
    
    def _fix_left_arm_action(self, action: np.ndarray) -> np.ndarray:
        """修复左臂动作 - 使用数据集均值"""
        if self.dataset_stats is not None:
            action_mean = np.array(self.dataset_stats["action"]["mean"])
            action[:7] = action_mean[:7]
        return action
    
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
    
    def run(self, duration: float = 60.0):
        """运行推理控制"""
        self._init_ros()
        self.load_model()
        
        print(f"\n开始推理控制 (持续 {duration} 秒)")
        print("按 Ctrl+C 停止")
        
        print("等待ROS数据...")
        obs = None
        for _ in range(100):
            obs = self._get_observation()
            if obs is not None:
                break
            time.sleep(0.05)
        
        if obs is None:
            print("错误: 未收到ROS数据!")
            return
        
        self.running = True
        dt = 1.0 / self.fps
        start_time = time.time()
        step_count = 0
        
        try:
            while self.running and (time.time() - start_time) < duration:
                loop_start = time.time()
                
                obs = self._get_observation()
                if obs is None:
                    time.sleep(0.01)
                    continue
                
                with torch.no_grad():
                    batch = {
                        "observation.state": obs["observation.state"],
                        "observation.images.camera_f": obs["observation.images.camera_f"],
                        "observation.images.camera_r": obs["observation.images.camera_r"],
                    }
                    
                    action = self.policy.select_action(batch)
                    
                    if hasattr(self.policy, 'postprocessor') and 'unnormalizer' in self.policy.postprocessor:
                        action = self.policy.postprocessor['unnormalizer'](action)
                    
                    action = action.squeeze(0).cpu().numpy()
                
                action = self._fix_left_arm_action(action)
                
                if step_count % 30 == 0:
                    print(f"  左臂动作: {action[:7].round(3)}")
                    print(f"  右臂动作: {action[7:14].round(3)}")
                
                self._publish_action(action)
                
                step_count += 1
                if step_count % 30 == 0:
                    elapsed = time.time() - start_time
                    print(f"  已运行 {elapsed:.1f}s, 步数: {step_count}")
                
                elapsed = time.time() - loop_start
                if elapsed < dt:
                    time.sleep(dt - elapsed)
                    
        except KeyboardInterrupt:
            print("\n用户停止推理")
        
        self.running = False
        print(f"推理完成, 总步数: {step_count}")


def main():
    parser = argparse.ArgumentParser(description="Piper双臂推理控制")
    parser.add_argument(
        "--checkpoint", 
        type=str, 
        default="outputs/train/piper_act/checkpoints/016000/pretrained_model",
        help="模型checkpoint路径"
    )
    parser.add_argument("--device", type=str, default="cuda", help="推理设备")
    parser.add_argument("--fps", type=int, default=30, help="控制频率")
    parser.add_argument("--duration", type=float, default=60.0, help="运行时长(秒)")
    
    args = parser.parse_args()
    
    checkpoint_path = Path(__file__).parent / args.checkpoint
    
    inference = PiperInference(
        checkpoint_path=checkpoint_path,
        device=args.device,
        fps=args.fps,
    )
    
    inference.run(duration=args.duration)


if __name__ == "__main__":
    main()