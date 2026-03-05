#!/usr/bin/env python3
"""简化的数据集测试"""

import sys
print("Step 1: 导入 Path...")
from pathlib import Path
print("  OK")

print("Step 2: 检查数据集目录...")
dataset_path = Path("/home/agilex/robot/code/my_data_i")
print(f"  数据集路径: {dataset_path}")
print(f"  存在: {dataset_path.exists()}")
print(f"  meta/info.json 存在: {(dataset_path / 'meta/info.json').exists()}")
print(f"  data/chunk-000/file-000.parquet 存在: {(dataset_path / 'data/chunk-000/file-000.parquet').exists()}")
print("  OK")

print("Step 3: 读取 info.json...")
import json
with open(dataset_path / "meta/info.json") as f:
    info = json.load(f)
print(f"  Episodes: {info['total_episodes']}")
print(f"  Frames: {info['total_frames']}")
print(f"  FPS: {info['fps']}")
print("  OK")

print("Step 4: 读取 parquet...")
import pyarrow.parquet as pq
table = pq.read_table(dataset_path / "data/chunk-000/file-000.parquet")
print(f"  行数: {table.num_rows}")
print(f"  列: {table.column_names}")
print("  OK")

print("\n✓ 所有基础测试通过!")