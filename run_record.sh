#!/bin/bash
# Piper数据采集启动脚本

# 激活conda环境
source ~/miniconda3/etc/profile.d/conda.sh
conda activate lerobot

# 设置PYTHONPATH
export PYTHONPATH=/home/agilex/robot/lerobot/src:$PYTHONPATH

# 运行采集
cd /home/agilex/robot/code

python piper_recorder.py \
    --repo-id piper_dataset \
    --root my_data_100 \
    --num-episodes 100 \
    --fps 30 \
    --num-cameras 2 \
    --task "pick_place"
