#!/usr/bin/env python3
"""
修复数据集：添加 observation.state 特征
ACT 模型需要 observation.state，但数据集只有 joint_left 和 joint_right
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path


def fix_dataset():
    dataset_path = Path.home() / "robot/code/my_data"
    meta_path = dataset_path / "meta"
    data_path = dataset_path / "data"
    
    # 1. 更新 info.json - 添加 observation.state 特征
    info_path = meta_path / "info.json"
    print(f"更新 info.json...")
    with open(info_path, 'r') as f:
        info = json.load(f)
    
    # 添加 observation.state 特征 (14维: joint_left 7 + joint_right 7)
    info["features"]["observation.state"] = {
        "dtype": "float32",
        "shape": [14],
        "names": ["left_j0", "left_j1", "left_j2", "left_j3", "left_j4", "left_j5", "left_j6",
                  "right_j0", "right_j1", "right_j2", "right_j3", "right_j4", "right_j5", "right_j6"]
    }
    
    with open(info_path, 'w') as f:
        json.dump(info, f, indent=4)
    print(f"  ✓ 添加 observation.state 特征定义")
    
    # 2. 更新 stats.json - 添加 observation.state 统计信息
    stats_path = meta_path / "stats.json"
    print(f"更新 stats.json...")
    with open(stats_path, 'r') as f:
        stats = json.load(f)
    
    # 使用 action 的统计信息（因为 action 已经是 joint_left + joint_right）
    if "action" in stats:
        stats["observation.state"] = stats["action"].copy()
    else:
        # 如果没有 action，从 joint_left 和 joint_right 计算
        left_stats = stats["joint_left"]
        right_stats = stats["joint_right"]
        stats["observation.state"] = {
            "min": left_stats["min"] + right_stats["min"],
            "max": left_stats["max"] + right_stats["max"],
            "mean": left_stats["mean"] + right_stats["mean"],
            "std": left_stats["std"] + right_stats["std"],
            "count": left_stats["count"],
            "q01": left_stats["q01"] + right_stats["q01"],
            "q99": left_stats["q99"] + right_stats["q99"],
        }
    
    with open(stats_path, 'w') as f:
        json.dump(stats, f, indent=4)
    print(f"  ✓ 添加 observation.state 统计信息")
    
    # 3. 更新 parquet 文件 - 添加 observation.state 列
    print(f"\n更新 parquet 文件...")
    for chunk_dir in sorted(data_path.iterdir()):
        if not chunk_dir.is_dir():
            continue
        print(f"  处理 {chunk_dir.name}...")
        
        for parquet_file in sorted(chunk_dir.glob("*.parquet")):
            print(f"    处理 {parquet_file.name}...")
            
            # 读取 parquet
            df = pd.read_parquet(parquet_file)
            
            # 如果已经有 observation.state 列，跳过
            if 'observation.state' in df.columns:
                print(f"      已存在 observation.state，跳过")
                continue
            
            # 创建 observation.state 列 (合并 joint_left 和 joint_right)
            df['observation.state'] = df.apply(
                lambda row: np.concatenate([row['joint_left'], row['joint_right']]).astype(np.float32),
                axis=1
            )
            
            # 保存回 parquet
            df.to_parquet(parquet_file)
            print(f"      ✓ 已添加 observation.state 列")
    
    print("\n✓ 修复完成!")
    print(f"  observation.state 特征维度: 14 (joint_left 7 + joint_right 7)")


if __name__ == "__main__":
    fix_dataset()