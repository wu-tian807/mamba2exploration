# Mamba-2 中文动态学习实验室

基于 Mamba-2 (Structured State Space Duality) 架构的中文语言模型训练实验，专注于研究 SSM 的**动态学习**机制。

## 实验配置

| 参数 | 值 |
|------|------|
| 架构 | Mamba-2 (SSD) |
| 参数量 | 83.2M |
| 词表大小 | 32,000 (自训练 BPE) |
| 模型维度 | d_model=768, n_layer=16 |
| SSM 隐状态 | d_state=64 |
| 训练数据 | TinyStoryInChinese (~219K 中文故事, 165MB, 2450万 tokens) |
| GPU | NVIDIA RTX 4090D |
| 训练用时 | ~35 分钟 (10000 步, batch=8, seq=2048) |
| 最终 Loss | 3.75 (从 163.7 下降 97.7%) |
| 训练速度 | ~81K tokens/s |

## 参数分配

```
Embedding: 24.6M (29.5%) — 词表 "字典"
Mamba 层:  58.6M (70.5%) — 动态学习核心
其他:       0.01M         — LayerNorm
```

## 项目结构

```
├── model.py           # Mamba-2 LM 架构定义
├── data.py            # 数据管道 (分词器训练 + DataLoader)
├── train.py           # 训练主循环 + 动态学习观测
├── generate.py        # 推理/文本生成
├── observe.py         # 参数快照 + 激活对比 + 可视化
├── prepare_data.py    # 一键数据准备脚本
├── requirements.txt   # 依赖
├── checkpoints/       # 模型权重
│   ├── final.pt       # 最终模型
│   └── step_*.pt      # 中间 checkpoints
├── data/              # 语料和分词器
│   ├── corpus.txt     # 中文语料
│   └── zh_tokenizer.json  # 32K BPE 分词器
└── logs/              # 训练日志和观测数据
    ├── loss_log.json          # 逐步 loss 记录
    ├── loss_curve.png         # loss 曲线
    ├── training_report.md     # 训练报告
    ├── generation_results.json # 推理测试结果
    └── observations/          # 动态学习观测
        ├── param_snapshots/   # 各层参数快照 (JSON)
        ├── param_evolution/   # 参数演化图 (PNG)
        ├── activation_diffs/  # 激活差异数据 (JSON)
        └── activation_plots/  # 激活对比图 (PNG)
```

## 环境搭建

```bash
conda create -n mamba2 python=3.11 -y
conda activate mamba2

# CUDA Toolkit (编译 CUDA 内核需要)
conda install -c nvidia/label/cuda-12.4.0 cuda-toolkit -y
conda install -c conda-forge gxx_linux-64=13.3 gcc_linux-64=13.3 -y

# PyTorch
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124

# Mamba (从源码编译, 确保 ABI 兼容)
# causal-conv1d v1.4.0 + mamba-ssm v2.2.4
git clone --branch v1.4.0 --depth 1 https://github.com/Dao-AILab/causal-conv1d.git
cd causal-conv1d && CUDA_HOME=$CONDA_PREFIX TORCH_CUDA_ARCH_LIST="8.9" \
  CAUSAL_CONV1D_FORCE_BUILD=TRUE pip install . --no-build-isolation
cd .. && git clone --branch v2.2.4 --depth 1 https://github.com/state-spaces/mamba.git
cd mamba && CUDA_HOME=$CONDA_PREFIX TORCH_CUDA_ARCH_LIST="8.9" \
  MAMBA_FORCE_BUILD=TRUE MAX_JOBS=2 pip install . --no-build-isolation

# 其他依赖
pip install datasets matplotlib tokenizers tqdm
```

## 快速开始

```bash
# 1. 准备数据 + 训练分词器
python prepare_data.py --action all

# 2. 开始训练 (大语料)
python train.py --corpus data/corpus_large.txt --seq_len 2048 --batch_size 8 --max_steps 10000

# 3. 生成文本
python generate.py --prompt "从前有一个小女孩"

# 4. 交互模式
bash chat.sh
# 或 python generate.py --interactive --checkpoint checkpoints/final.pt

# 5. 动态学习观测
python generate.py --prompt "从前" --observe
```

## 动态学习观测亮点

训练过程中，`observe.py` 自动采集:

1. **参数快照**: 每 100 步记录各层 Mamba 的 dt_bias / A_log / D 等关键参数统计
2. **全层快照**: 每 500 步对全部 16 层做完整参数快照
3. **激活差异对比**: 给两段不同文本，观测各层输出的余弦相似度变化
4. **参数演化图**: 可视化训练过程中各参数的 mean/std/abs_mean 变化轨迹

## License

Apache 2.0
