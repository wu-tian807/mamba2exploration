#!/bin/bash
# Mamba-2 中文模型交互聊天启动脚本

eval "$(conda shell.bash hook)"
conda activate mamba2
cd /home/wutian/projects/mamba2-lab

echo "=========================================="
echo "  Mamba-2 中文故事模型 (83M) 交互模式"
echo "  输入任意中文开头，模型会续写故事"
echo "  输入 quit 退出"
echo "=========================================="
echo ""

python generate.py --interactive --checkpoint checkpoints/final.pt --temperature 0.8 --max_new_tokens 200
