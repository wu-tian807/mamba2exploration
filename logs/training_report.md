# Mamba-2 中文模型训练报告

**生成时间**: 2026-03-24 12:45:14

## 模型配置

| 参数 | 值 |
|------|----|
| vocab_size | 32000 |
| d_model | 768 |
| n_layer | 16 |
| d_state | 64 |
| d_conv | 4 |
| expand | 2 |

## 训练超参

| 参数 | 值 |
|------|----|
| corpus | /home/wutian/projects/mamba2-lab/data/corpus.txt |
| seq_len | 1024 |
| batch_size | 4 |
| max_steps | 5000 |
| max_lr | 0.0006 |
| min_lr | 6e-05 |
| warmup_steps | 100 |
| max_grad_norm | 1.0 |
| precision | bf16 |
| log_interval | 10 |
| save_interval | 500 |
| observe_interval | 100 |
| num_workers | 4 |
| resume | None |

## 训练结果

- **初始 Loss**: 155.3030
- **最终 Loss**: 3.7484
- **最低 Loss**: 3.1247
- **Loss 下降**: 151.5546 (97.6%)
- **总训练步数**: 4999

## 动态学习观测

共采集了 **50** 次参数快照。

各参数演化图见 `logs/observations/param_evolution/` 目录。
各层激活差异对比见 `logs/observations/activation_plots/` 目录。

## 文件清单

```
checkpoints/     # 模型权重
logs/            # 训练日志和可视化
  loss_log.json  # 逐步 loss 记录
  loss_curve.png # loss 曲线图
  observations/  # 动态学习观测数据
    param_snapshots/   # 参数快照 (JSON)
    param_evolution/   # 参数演化图 (PNG)
    activation_diffs/  # 激活差异数据 (JSON)
    activation_plots/  # 激活差异图 (PNG)
data/            # 语料和分词器
```
