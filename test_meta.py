#!/usr/bin/env python3
"""测试元数据加载"""

import sys
print("Step 1: 导入 lerobot...")
sys.path.insert(0, "/home/agilex/robot/lerobot/src")
from lerobot.datasets.utils import load_info, load_episodes, load_tasks, load_stats
from pathlib import Path

print("Step 2: 测试加载元数据...")
root = Path("/home/agilex/robot/code/my_data_i")
print(f"  数据集根目录: {root}")

print("\n  加载 info.json...")
info = load_info(root)
print(f"    Episodes: {info['total_episodes']}")
print(f"    Frames: {info['total_frames']}")

print("\n  加载 episodes...")
episodes = load_episodes(root)
print(f"    Episodes 数量: {len(episodes)}")

print("\n  加载 tasks...")
tasks = load_tasks(root)
print(f"    Tasks: {tasks}")

print("\n  加载 stats...")
stats = load_stats(root)
print(f"    Stats keys: {list(stats.keys())}")

print("\n✓ 元数据加载成功!")