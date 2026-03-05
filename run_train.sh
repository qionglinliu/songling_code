#!/bin/bash
# Piper双臂机器人训练脚本

# 激活conda环境
source ~/miniconda3/etc/profile.d/conda.sh
conda activate lerobot

# 设置PYTHONPATH
export PYTHONPATH=/home/agilex/robot/lerobot/src:$PYTHONPATH

# 设置离线模式，避免访问 HuggingFace Hub
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

# 运行训练
cd /home/agilex/robot/code

# 使用ACT策略训练
python -m lerobot.scripts.lerobot_train \
    --dataset.repo_id my_data_i \
    --dataset.root /home/agilex/robot/code/my_data_i \
    --dataset.video_backend pyav \
    --policy.type act \
    --policy.push_to_hub false \
    --output_dir outputs/train/piper_act \
    --save_freq 2000 \
    --steps 20000 \
    --batch_size 8 \
    --num_workers 4 \
    --log_freq 100
