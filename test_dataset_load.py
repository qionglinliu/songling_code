#!/usr/bin/env python3
"""测试数据集加载问题"""

import sys
import traceback

print("Step 1: 导入基础库...")
import torch
from pathlib import Path

print("Step 2: 导入 lerobot 库...")
from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata

dataset_root = Path.home() / "robot/code"
dataset_repo_id = "my_data"

print(f"Step 3: 加载数据集元数据 (root={dataset_root}, repo_id={dataset_repo_id})...")
try:
    dataset_metadata = LeRobotDatasetMetadata(dataset_repo_id, root=dataset_root)
    print(f"  ✓ Episodes: {dataset_metadata.total_episodes}")
    print(f"  ✓ Frames: {dataset_metadata.total_frames}")
    print(f"  ✓ FPS: {dataset_metadata.fps}")
    print(f"  ✓ Video keys: {dataset_metadata.video_keys}")
    print(f"  ✓ Features: {list(dataset_metadata.features.keys())}")
except Exception as e:
    print(f"  ✗ 加载元数据失败: {e}")
    traceback.print_exc()
    sys.exit(1)

print("\nStep 4: 测试数据集加载 (无视频解码)...")
try:
    # 不使用 delta_timestamps 先测试基本加载
    dataset = LeRobotDataset(
        dataset_repo_id,
        root=dataset_root,
        video_backend="pyav",  # 使用 pyav 后端
    )
    print(f"  ✓ 数据集加载成功!")
    print(f"  ✓ 数据集大小: {len(dataset)}")
except Exception as e:
    print(f"  ✗ 加载数据集失败: {e}")
    traceback.print_exc()
    sys.exit(1)

print("\nStep 5: 测试读取单个样本...")
try:
    sample = dataset[0]
    print(f"  ✓ 读取样本成功!")
    print(f"  ✓ 样本键: {list(sample.keys())}")
    for key, val in sample.items():
        if isinstance(val, torch.Tensor):
            print(f"    - {key}: shape={val.shape}, dtype={val.dtype}")
        else:
            print(f"    - {key}: {type(val).__name__}")
except Exception as e:
    print(f"  ✗ 读取样本失败: {e}")
    traceback.print_exc()
    sys.exit(1)

print("\n✓ 所有测试通过!")