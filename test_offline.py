#!/usr/bin/env python3
"""测试离线模式下的数据集加载"""

import os
# 设置离线模式
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

import sys
sys.path.insert(0, "/home/agilex/robot/lerobot/src")

print("Step 1: 导入 LeRobotDataset...")
from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
from pathlib import Path

print("Step 2: 测试 LeRobotDatasetMetadata...")
# root 应该是数据集的直接目录（包含 meta/, data/, videos/ 等）
root = Path("/home/agilex/robot/code/my_data_i")
repo_id = "my_data_i"

try:
    meta = LeRobotDatasetMetadata(repo_id, root=root)
    print(f"  ✓ Episodes: {meta.total_episodes}")
    print(f"  ✓ Frames: {meta.total_frames}")
except Exception as e:
    print(f"  ✗ 失败: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\nStep 3: 测试 LeRobotDataset...")
try:
    dataset = LeRobotDataset(
        repo_id,
        root=root,
        video_backend="pyav",
    )
    print(f"  ✓ 数据集大小: {len(dataset)}")
except Exception as e:
    print(f"  ✗ 失败: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n✓ 所有测试通过!")