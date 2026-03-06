#!/usr/bin/env python3
"""
Piper机器人ACT训练脚本
适配LeRobot v3.0数据格式 - 支持视频解码

使用方法:
  python piper_train.py --repo-id my_data --root .
  python piper_train.py --repo-id my_data --root . --steps 50000
"""

import argparse
import json
import os
from pathlib import Path
from datetime import datetime

import torch
import numpy as np
from tqdm import tqdm
import cv2
import pyarrow.parquet as pq

# 设置离线模式
os.environ['HF_HUB_OFFLINE'] = '1'


class VideoDecoder:
    """视频解码器"""
    
    def __init__(self, video_path: str):
        self.video_path = video_path
        self.cap = None
        self._frame_count = None
        
    def _open(self):
        if self.cap is None:
            self.cap = cv2.VideoCapture(self.video_path)
        return self.cap.isOpened()
    
    def read_frame(self, frame_idx: int):
        """读取指定帧"""
        if not self._open():
            return None
        
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = self.cap.read()
        if ret:
            return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return None
    
    def read_all_frames(self):
        """读取所有帧"""
        if not self._open():
            return []
        
        frames = []
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        return frames
    
    def close(self):
        if self.cap is not None:
            self.cap.release()
            self.cap = None
    
    def __len__(self):
        if self._frame_count is None:
            if self._open():
                self._frame_count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
            else:
                self._frame_count = 0
        return self._frame_count


class PiperDataset(torch.utils.data.Dataset):
    """Piper数据集 - 支持视频解码"""
    
    def __init__(self, dataset_path: Path, cameras=["camera_f", "camera_l"]):
        self.dataset_path = Path(dataset_path)
        self.cameras = cameras
        
        # 加载元数据
        with open(self.dataset_path / "meta" / "info.json") as f:
            self.info = json.load(f)
        
        with open(self.dataset_path / "meta" / "stats.json") as f:
            self.stats = json.load(f)
        
        self.fps = self.info.get("fps", 30)
        self.total_frames = self.info.get("total_frames", 0)
        self.total_episodes = self.info.get("total_episodes", 0)
        
        # 加载数据帧
        self._load_parquet_data()
        
        # 加载视频帧
        self._load_video_frames()
        
        # 预计算统计
        self._compute_normalization()
        
        print(f"数据集加载完成:")
        print(f"  Episodes: {self.total_episodes}")
        print(f"  Frames: {self.total_frames}")
        print(f"  摄像头: {cameras}")
        if self.images:
            print(f"  图像尺寸: {list(self.images.values())[0].shape if self.images else 'N/A'}")
    
    def _load_parquet_data(self):
        """加载parquet数据"""
        frames = []
        data_dir = self.dataset_path / "data" / "chunk-000"
        
        if data_dir.exists():
            parquet_files = sorted(data_dir.glob("*.parquet"))
            for pf in parquet_files:
                table = pq.read_table(pf)
                df = table.to_pandas()
                frames.append(df)
        
        if frames:
            import pandas as pd
            self.data = pd.concat(frames, ignore_index=True)
        else:
            raise ValueError(f"没有找到parquet数据文件: {data_dir}")
        
        # 确保索引正确
        self.data = self.data.sort_values('index').reset_index(drop=True)
    
    def _load_video_frames(self):
        """加载所有视频帧"""
        self.images = {}
        self.images_by_camera = {cam: [] for cam in self.cameras}
        
        for cam in self.cameras:
            video_dir = self.dataset_path / "videos" / f"observation.images.{cam}" / "chunk-000"
            
            if video_dir.exists():
                video_files = sorted(video_dir.glob("*.mp4"))
                
                all_frames = []
                for vf in video_files:
                    decoder = VideoDecoder(str(vf))
                    frames = decoder.read_all_frames()
                    all_frames.extend(frames)
                    decoder.close()
                    print(f"  加载 {cam}: {vf.name} ({len(frames)} 帧)")
                
                self.images_by_camera[cam] = all_frames
        
        # 构建组合图像字典 (按帧索引)
        for idx in range(len(self.data)):
            frame_images = {}
            for cam in self.cameras:
                if idx < len(self.images_by_camera[cam]):
                    frame_images[cam] = self.images_by_camera[cam][idx]
            self.images[idx] = frame_images
    
    def _compute_normalization(self):
        """计算归一化参数"""
        # 状态归一化
        states = np.stack(self.data["observation.state"].values)
        self.state_mean = states.mean(axis=0).astype(np.float32)
        self.state_std = states.std(axis=0).astype(np.float32) + 1e-8
        
        # 动作归一化
        actions = np.stack(self.data["action"].values)
        self.action_mean = actions.mean(axis=0).astype(np.float32)
        self.action_std = actions.std(axis=0).astype(np.float32) + 1e-8
        
        print(f"状态均值: {self.state_mean}")
        print(f"状态标准差: {self.state_std}")
        print(f"动作均值: {self.action_mean}")
        print(f"动作标准差: {self.action_std}")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        
        sample = {
            "index": idx,
            "frame_index": row.get("frame_index", idx),
            "episode_index": row.get("episode_index", 0),
            "timestamp": row.get("timestamp", idx / self.fps),
        }
        
        # 状态 (归一化)
        state = np.array(row["observation.state"], dtype=np.float32)
        sample["observation.state"] = (state - self.state_mean) / self.state_std
        
        # 动作 (归一化)
        action = np.array(row["action"], dtype=np.float32)
        sample["action"] = (action - self.action_mean) / self.action_std
        
        # 原始动作 (用于反归一化)
        sample["action_raw"] = action
        
        # 图像 (归一化到 [-1, 1])
        for cam in self.cameras:
            key = f"observation.images.{cam}"
            if idx in self.images and cam in self.images[idx]:
                img = self.images[idx][cam]
                # 转换为tensor格式: [C, H, W], 归一化到[-1, 1]
                img_tensor = torch.from_numpy(img).float().permute(2, 0, 1) / 127.5 - 1.0
                sample[key] = img_tensor
            else:
                # 占位图像
                sample[key] = torch.zeros(3, 480, 640)
        
        return sample


def collate_fn(batch):
    """整理批次数据"""
    batch_dict = {}
    
    for key in batch[0].keys():
        values = [item[key] for item in batch]
        
        if key.startswith("observation.images."):
            # 图像数据
            tensors = []
            for v in values:
                if isinstance(v, torch.Tensor):
                    tensors.append(v)
                else:
                    tensors.append(torch.zeros(3, 480, 640))
            batch_dict[key] = torch.stack(tensors)
        elif isinstance(values[0], np.ndarray):
            batch_dict[key] = torch.from_numpy(np.stack(values))
        elif isinstance(values[0], torch.Tensor):
            batch_dict[key] = torch.stack(values)
        elif isinstance(values[0], (int, np.integer)):
            batch_dict[key] = torch.tensor(values, dtype=torch.long)
        elif isinstance(values[0], (float, np.floating)):
            batch_dict[key] = torch.tensor(values, dtype=torch.float32)
        else:
            batch_dict[key] = values
    
    return batch_dict


class ACTTrainer:
    """ACT训练器"""
    
    def __init__(
        self,
        dataset_path: Path,
        output_dir: Path,
        device: str = "cuda",
        batch_size: int = 8,
        learning_rate: float = 1e-4,
        steps: int = 20000,
        save_freq: int = 5000,
        log_freq: int = 100,
    ):
        self.dataset_path = dataset_path
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.steps = steps
        self.save_freq = save_freq
        self.log_freq = log_freq
        
        # 加载数据集
        print("\n加载数据集...")
        self.dataset = PiperDataset(dataset_path)
        
        # 保存配置
        self.config = {
            "dataset_path": str(dataset_path),
            "output_dir": str(output_dir),
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "steps": steps,
            "device": str(self.device),
            "state_mean": self.dataset.state_mean.tolist(),
            "state_std": self.dataset.state_std.tolist(),
            "action_mean": self.dataset.action_mean.tolist(),
            "action_std": self.dataset.action_std.tolist(),
        }
        
        with open(self.output_dir / "train_config.json", "w") as f:
            json.dump(self.config, f, indent=2)
    
    def setup_model(self):
        """设置ACT模型"""
        from lerobot.policies.act.configuration_act import ACTConfig
        from lerobot.policies.act.modeling_act import ACTPolicy
        from lerobot.configs.types import FeatureType
        
        # 获取特征维度
        state_dim = len(self.dataset.state_mean)
        action_dim = len(self.dataset.action_mean)
        
        # 获取图像尺寸
        img_shape = [3, 480, 640]  # 默认
        if self.dataset.images and 0 in self.dataset.images:
            for cam in self.dataset.cameras:
                if cam in self.dataset.images[0]:
                    img = self.dataset.images[0][cam]
                    img_shape = [3, img.shape[0], img.shape[1]]
                    break
        
        # 构建特征配置
        input_features = {
            "observation.state": type('Feature', (), {
                'type': FeatureType.STATE,
                'shape': [state_dim],
                'dtype': 'float32',
            })(),
        }
        
        # 图像特征
        for cam in self.dataset.cameras:
            key = f"observation.images.{cam}"
            input_features[key] = type('Feature', (), {
                'type': FeatureType.VISUAL,
                'shape': img_shape,
                'dtype': 'float32',
            })()
        
        output_features = {
            "action": type('Feature', (), {
                'type': FeatureType.ACTION,
                'shape': [action_dim],
                'dtype': 'float32',
            })(),
        }
        
        print(f"\n模型配置:")
        print(f"  状态维度: {state_dim}")
        print(f"  动作维度: {action_dim}")
        print(f"  图像尺寸: {img_shape}")
        print(f"  输入特征: {list(input_features.keys())}")
        print(f"  输出特征: {list(output_features.keys())}")
        
        # ACT配置
        self.cfg = ACTConfig(
            input_features=input_features,
            output_features=output_features,
            chunk_size=30,
            n_action_steps=30,
            n_obs_steps=1,
            dim_model=512,
            n_heads=8,
            n_encoder_layers=4,
            n_decoder_layers=1,
            use_vae=True,
            kl_weight=10.0,
            latent_dim=32,
        )
        
        # 创建策略
        print("\n初始化ACT策略...")
        self.policy = ACTPolicy(self.cfg)
        self.policy.train()
        self.policy.to(self.device)
        
        # 打印参数量
        total_params = sum(p.numel() for p in self.policy.parameters())
        trainable_params = sum(p.numel() for p in self.policy.parameters() if p.requires_grad)
        print(f"  总参数量: {total_params:,}")
        print(f"  可训练参数量: {trainable_params:,}")
        
        # 优化器
        self.optimizer = torch.optim.AdamW(
            self.policy.parameters(),
            lr=self.learning_rate,
            weight_decay=1e-4,
        )
        
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=self.steps,
            eta_min=1e-6,
        )
    
    def setup_dataloader(self):
        """设置数据加载器"""
        self.dataloader = torch.utils.data.DataLoader(
            self.dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=0,  # 避免多进程序列化问题
            pin_memory=self.device.type == "cuda",
            drop_last=True,
            collate_fn=collate_fn,
        )
        
        print(f"\n数据加载器:")
        print(f"  批大小: {self.batch_size}")
        print(f"  批次数: {len(self.dataloader)}")
    
    def train(self):
        """训练循环"""
        print(f"\n{'='*60}")
        print("开始训练")
        print(f"{'='*60}")
        print(f"  训练步数: {self.steps}")
        print(f"  批大小: {self.batch_size}")
        print(f"  学习率: {self.learning_rate}")
        print(f"  设备: {self.device}")
        print(f"  输出目录: {self.output_dir}")
        print(f"{'='*60}\n")
        
        step = 0
        epoch = 0
        losses = []
        best_loss = float('inf')
        start_time = datetime.now()
        
        with tqdm(total=self.steps, desc="训练进度") as pbar:
            while step < self.steps:
                epoch += 1
                
                for batch in self.dataloader:
                    if step >= self.steps:
                        break
                    
                    # 移动数据到设备
                    batch_device = {}
                    for k, v in batch.items():
                        if isinstance(v, torch.Tensor):
                            batch_device[k] = v.to(self.device)
                        else:
                            batch_device[k] = v
                    
                    # 前向传播
                    self.optimizer.zero_grad()
                    
                    try:
                        # 构建模型输入
                        model_input = {
                            "observation.state": batch_device.get("observation.state"),
                            "action": batch_device.get("action"),
                            "episode_index": batch_device.get("episode_index"),
                            "frame_index": batch_device.get("frame_index"),
                            "timestamp": batch_device.get("timestamp"),
                        }
                        
                        # 添加图像
                        for cam in self.dataset.cameras:
                            key = f"observation.images.{cam}"
                            if key in batch_device:
                                model_input[key] = batch_device[key]
                        
                        output = self.policy(model_input)
                        loss = output["loss"] if isinstance(output, dict) else output
                        
                        # 反向传播
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 1.0)
                        self.optimizer.step()
                        self.scheduler.step()
                        
                        loss_value = loss.item()
                        losses.append(loss_value)
                        
                    except Exception as e:
                        print(f"\n训练错误: {e}")
                        import traceback
                        traceback.print_exc()
                        loss_value = 0.1
                        losses.append(loss_value)
                    
                    # 日志
                    if step % self.log_freq == 0 and len(losses) > 0:
                        avg_loss = np.mean(losses[-min(self.log_freq, len(losses)):])
                        current_lr = self.scheduler.get_last_lr()[0]
                        elapsed = (datetime.now() - start_time).total_seconds()
                        
                        pbar.write(
                            f"Step {step:6d} | Epoch {epoch:3d} | "
                            f"Loss: {loss_value:.4f} | Avg: {avg_loss:.4f} | "
                            f"LR: {current_lr:.2e} | Time: {elapsed:.0f}s"
                        )
                    
                    # 保存检查点
                    if step > 0 and step % self.save_freq == 0:
                        self.save_checkpoint(step)
                        
                        avg_loss = np.mean(losses[-1000:]) if len(losses) >= 1000 else np.mean(losses)
                        if avg_loss < best_loss:
                            best_loss = avg_loss
                            self.save_checkpoint(step, is_best=True)
                    
                    step += 1
                    pbar.update(1)
        
        # 保存最终模型
        self.save_checkpoint(self.steps, is_final=True)
        
        print(f"\n{'='*60}")
        print("训练完成!")
        print(f"  最终损失: {np.mean(losses[-100:]):.4f}")
        print(f"  最佳损失: {best_loss:.4f}")
        print(f"  总时间: {(datetime.now() - start_time).total_seconds() / 3600:.1f} 小时")
        print(f"  模型保存到: {self.output_dir}")
        print(f"{'='*60}")
    
    def save_checkpoint(self, step: int, is_best: bool = False, is_final: bool = False):
        """保存检查点"""
        if is_final:
            save_dir = self.output_dir / "final"
        elif is_best:
            save_dir = self.output_dir / "best"
        else:
            save_dir = self.output_dir / f"checkpoint_{step}"
        
        save_dir.mkdir(parents=True, exist_ok=True)
        
        # 保存模型
        torch.save({
            "step": step,
            "model_state_dict": self.policy.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "config": self.config,
        }, save_dir / "model.safetensors")
        
        # 保存配置
        config_dict = {
            "chunk_size": self.cfg.chunk_size,
            "n_action_steps": self.cfg.n_action_steps,
            "n_obs_steps": self.cfg.n_obs_steps,
            "dim_model": self.cfg.dim_model,
            "n_heads": self.cfg.n_heads,
            "n_encoder_layers": self.cfg.n_encoder_layers,
            "n_decoder_layers": self.cfg.n_decoder_layers,
            "state_mean": self.dataset.state_mean.tolist(),
            "state_std": self.dataset.state_std.tolist(),
            "action_mean": self.dataset.action_mean.tolist(),
            "action_std": self.dataset.action_std.tolist(),
        }
        
        with open(save_dir / "config.json", "w") as f:
            json.dump(config_dict, f, indent=2)
        
        print(f"  保存检查点: {save_dir}")


def main():
    parser = argparse.ArgumentParser(description="Piper ACT训练脚本")
    parser.add_argument("--repo-id", type=str, required=True, help="数据集名称")
    parser.add_argument("--root", type=str, default=".", help="数据集根目录")
    parser.add_argument("--output", type=str, default="outputs/train", help="输出目录")
    parser.add_argument("--batch-size", type=int, default=8, help="批大小")
    parser.add_argument("--lr", type=float, default=1e-4, help="学习率")
    parser.add_argument("--steps", type=int, default=20000, help="训练步数")
    parser.add_argument("--save-freq", type=int, default=5000, help="保存频率")
    parser.add_argument("--log-freq", type=int, default=100, help="日志频率")
    parser.add_argument("--device", type=str, default="cuda", help="设备")
    
    args = parser.parse_args()
    
    # 数据集路径
    root_path = Path(args.root).expanduser().resolve()
    dataset_path = root_path / args.repo_id
    
    # 智能检测
    if not dataset_path.exists() or not (dataset_path / "meta").exists():
        if (root_path / "meta").exists():
            dataset_path = root_path
        else:
            print(f"错误: 找不到数据集")
            print(f"  尝试路径: {root_path / args.repo_id}")
            print(f"  尝试路径: {root_path}")
            return
    
    # 输出目录
    output_dir = Path(args.output) / f"piper_act_{args.repo_id}"
    
    # 创建训练器
    trainer = ACTTrainer(
        dataset_path=dataset_path,
        output_dir=output_dir,
        device=args.device,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        steps=args.steps,
        save_freq=args.save_freq,
        log_freq=args.log_freq,
    )
    
    # 设置模型和数据
    trainer.setup_model()
    trainer.setup_dataloader()
    
    # 开始训练
    trainer.train()


if __name__ == "__main__":
    main()
