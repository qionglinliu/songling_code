#!/usr/bin/env python3
"""快速诊断脚本"""
import sys
import os

# 设置离线模式，防止尝试从 Hub 下载
os.environ['HF_HUB_OFFLINE'] = '1'

print("1. 导入库...")
from pathlib import Path
from lerobot.datasets.utils import load_info, load_episodes, load_stats, load_tasks

dataset_root = Path.home() / "robot/code/my_data"

print("2. 直接加载 info.json...")
try:
    info = load_info(dataset_root)
    print(f"   codebase_version: {info.get('codebase_version')}")
    print(f"   total_episodes: {info.get('total_episodes')}")
    print("   ✓ OK")
except Exception as e:
    print(f"   ✗ 失败: {e}")
    sys.exit(1)

print("3. 直接加载 episodes...")
try:
    episodes = load_episodes(dataset_root)
    print(f"   episodes 数量: {len(episodes)}")
    print("   ✓ OK")
except Exception as e:
    print(f"   ✗ 失败: {e}")
    sys.exit(1)

print("4. 直接加载 stats...")
try:
    stats = load_stats(dataset_root)
    print(f"   特征数: {len(stats)}")
    print("   ✓ OK")
except Exception as e:
    print(f"   ✗ 失败: {e}")
    sys.exit(1)

print("5. 直接加载 tasks...")
try:
    tasks = load_tasks(dataset_root)
    print(f"   tasks: {tasks}")
    print("   ✓ OK")
except Exception as e:
    print(f"   ✗ 失败: {e}")
    sys.exit(1)

print("\n所有基本加载测试通过!")
print("\n现在测试 LeRobotDatasetMetadata...")

from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata
try:
    meta = LeRobotDatasetMetadata("my_data", root=dataset_root.parent, force_cache_sync=False)
    print(f"   ✓ 元数据加载成功!")
    print(f"   episodes: {meta.total_episodes}")
except Exception as e:
    print(f"   ✗ 失败: {e}")
    import traceback
    traceback.print_exc()