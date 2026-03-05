#!/usr/bin/env python3
"""
Piper双臂机器人训练脚本
使用ACT策略训练
"""

import torch
from pathlib import Path
from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
from lerobot.datasets.utils import dataset_to_policy_features
from lerobot.policies.act.configuration_act import ACTConfig
from lerobot.policies.act.modeling_act import ACTPolicy
from lerobot.policies.utils import make_pre_post_processors
from lerobot.configs.types import FeatureType


def main():
    # 配置
    # 数据集路径: root/repo_id 结构
    # 由于录制时用的是 --root my_data，数据在 ~/robot/code/my_data/ 下
    # 所以 root 应该是 ~/robot/code，repo_id 应该是 my_data
    dataset_root = Path.home() / "robot/code"
    dataset_repo_id = "my_data"  
    output_dir = Path("outputs/train/piper_act")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    
    # 训练参数
    training_steps = 20000
    batch_size = 8
    learning_rate = 1e-4
    log_freq = 100
    save_freq = 2000
    
    # 加载数据集元数据
    print("加载数据集元数据...")
    dataset_metadata = LeRobotDatasetMetadata(dataset_repo_id, root=dataset_root)
    print(f"  Episodes: {dataset_metadata.total_episodes}")
    print(f"  Frames: {dataset_metadata.total_frames}")
    print(f"  FPS: {dataset_metadata.fps}")
    
    # 获取特征
    features = dataset_to_policy_features(dataset_metadata.features)
    
    # 对于Piper双臂数据集，joint_left和joint_right是动作（action）
    # 需要手动标记为ACTION类型
    from lerobot.configs.types import FeatureType
    
    for key in ['joint_left', 'joint_right']:
        if key in features:
            features[key].type = FeatureType.ACTION
    
    # 分离输入和输出特征
    output_features = {key: ft for key, ft in features.items() if ft.type is FeatureType.ACTION}
    input_features = {key: ft for key, ft in features.items() if key not in output_features}
    
    print(f"\n输入特征: {list(input_features.keys())}")
    print(f"输出特征 (动作): {list(output_features.keys())}")
    
    # 配置ACT策略
    cfg = ACTConfig(
        input_features=input_features,
        output_features=output_features,
        chunk_size=100,  # 动作序列长度
        n_obs_steps=2,   # 观察历史帧数
        dim_model=512,
        n_heads=8,
        n_encoder_layers=4,
        n_decoder_layers=7,
    )
    
    # 创建策略
    print("\n初始化ACT策略...")
    policy = ACTPolicy(cfg)
    policy.train()
    policy.to(device)
    
    # 创建预处理器和后处理器
    preprocessor, postprocessor = make_pre_post_processors(cfg, dataset_stats=dataset_metadata.stats)
    
    # 设置delta_timestamps
    fps = dataset_metadata.fps
    delta_timestamps = {
        # 观察图像 - 当前帧和前一帧
        "observation.images.camera_f": [i / fps for i in cfg.observation_delta_indices],
        "observation.images.camera_r": [i / fps for i in cfg.observation_delta_indices],
        # 关节状态作为action
        "joint_left": [i / fps for i in cfg.action_delta_indices],
        "joint_right": [i / fps for i in cfg.action_delta_indices],
    }
    
    # 加载数据集
    print("加载数据集...")
    dataset = LeRobotDataset(
        dataset_repo_id,
        root=dataset_root,
        delta_timestamps=delta_timestamps,
    )
    print(f"  数据集大小: {len(dataset)}")
    
    # 创建数据加载器
    dataloader = torch.utils.data.DataLoader(
        dataset,
        num_workers=4,
        batch_size=batch_size,
        shuffle=True,
        pin_memory=device.type != "cpu",
        drop_last=True,
    )
    
    # 创建优化器
    optimizer = torch.optim.Adam(policy.parameters(), lr=learning_rate)
    
    # 训练循环
    print(f"\n开始训练...")
    print(f"  训练步数: {training_steps}")
    print(f"  批大小: {batch_size}")
    print(f"  学习率: {learning_rate}")
    print(f"  输出目录: {output_dir}")
    print()
    
    step = 0
    done = False
    epoch = 0
    best_loss = float('inf')
    
    while not done:
        epoch += 1
        epoch_loss = 0
        epoch_steps = 0
        
        for batch in dataloader:
            batch = preprocessor(batch)
            loss, loss_dict = policy.forward(batch)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            
            epoch_loss += loss.item()
            epoch_steps += 1
            
            if step % log_freq == 0:
                avg_loss = epoch_loss / epoch_steps if epoch_steps > 0 else 0
                print(f"Step {step:6d} | Epoch {epoch:3d} | Loss: {loss.item():.4f} | Avg Loss: {avg_loss:.4f}")
            
            step += 1
            
            # 保存检查点
            if step % save_freq == 0:
                checkpoint_dir = output_dir / f"checkpoint_{step}"
                policy.save_pretrained(checkpoint_dir)
                preprocessor.save_pretrained(checkpoint_dir)
                postprocessor.save_pretrained(checkpoint_dir)
                print(f"  保存检查点: {checkpoint_dir}")
            
            if step >= training_steps:
                done = True
                break
    
    # 保存最终模型
    final_dir = output_dir / "final"
    policy.save_pretrained(final_dir)
    preprocessor.save_pretrained(final_dir)
    postprocessor.save_pretrained(final_dir)
    print(f"\n训练完成!")
    print(f"最终模型保存到: {final_dir}")


if __name__ == "__main__":
    main()