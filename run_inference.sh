#!/bin/bash
# Piper双臂推理脚本

# 激活conda环境
source ~/miniconda3/etc/profile.d/conda.sh
conda activate lerobot

# 设置PYTHONPATH
export PYTHONPATH=/home/agilex/robot/lerobot/src:$PYTHONPATH

# 运行推理
cd /home/agilex/robot/code

# 默认使用最新的 checkpoint (last -> 016000)
python piper_inference.py \
    --checkpoint outputs/train/piper_act/checkpoints/016000/pretrained_model \
    --device cuda \
    --fps 30 \
    --duration 60

# 如果要使用其他 checkpoint，可以修改上面的路径，例如：
# --checkpoint outputs/train/piper_act/checkpoints/014000/pretrained_model
