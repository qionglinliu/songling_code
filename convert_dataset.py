#!/usr/bin/env python3
"""
将现有数据集的joint_left和joint_right合并为action特征
"""

import json
import shutil
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq


def convert_dataset(dataset_path: Path):
    """转换数据集，添加action特征"""
    
    meta_path = dataset_path / "meta"
    data_path = dataset_path / "data"
    
    # 1. 更新info.json
    info_file = meta_path / "info.json"
    with open(info_file, 'r') as f:
        info = json.load(f)
    
    # 添加action特征 (14维: joint_left 7 + joint_right 7)
    info["features"]["action"] = {
        "dtype": "float32",
        "shape": [14],
        "names": ["left_j0", "left_j1", "left_j2", "left_j3", "left_j4", "left_j5", "left_j6",
                  "right_j0", "right_j1", "right_j2", "right_j3", "right_j4", "right_j5", "right_j6"]
    }
    
    with open(info_file, 'w') as f:
        json.dump(info, f, indent=4)
    
    print(f"更新 info.json: 添加 action 特征")
    
    # 2. 更新stats.json
    stats_file = meta_path / "stats.json"
    with open(stats_file, 'r') as f:
        stats = json.load(f)
    
    # 合并joint_left和joint_right的统计信息
    left_stats = stats["joint_left"]
    right_stats = stats["joint_right"]
    
    stats["action"] = {
        "min": left_stats["min"] + right_stats["min"],
        "max": left_stats["max"] + right_stats["max"],
        "mean": left_stats["mean"] + right_stats["mean"],
        "std": left_stats["std"] + right_stats["std"],
        "count": left_stats["count"],
        "q01": left_stats["q01"] + right_stats["q01"],
        "q99": left_stats["q99"] + right_stats["q99"],
    }
    
    with open(stats_file, 'w') as f:
        json.dump(stats, f, indent=4)
    
    print(f"更新 stats.json: 添加 action 统计信息")
    
    # 3. 更新parquet文件
    for chunk_dir in sorted(data_path.iterdir()):
        if not chunk_dir.is_dir():
            continue
        print(f"处理 {chunk_dir.name}...")
        
        for parquet_file in sorted(chunk_dir.glob("*.parquet")):
            print(f"  处理 {parquet_file.name}...")
            
            # 读取parquet
            table = pq.read_table(parquet_file)
            df = table.to_pandas()
            
            # 创建action列 (合并joint_left和joint_right)
            df['action'] = df.apply(
                lambda row: np.concatenate([row['joint_left'], row['joint_right']]).astype(np.float32),
                axis=1
            )
            
            # 保存回parquet
            df.to_parquet(parquet_file)
    
    print("\n转换完成!")
    print(f"action 特征维度: 14 (joint_left 7 + joint_right 7)")


if __name__ == "__main__":
    dataset_path = Path.home() / "robot/code/my_data"
    convert_dataset(dataset_path)