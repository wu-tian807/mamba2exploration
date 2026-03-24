"""
动态学习观测工具 —— 窥探 Mamba-2 的 "大脑活动"

所有观测数据强制落盘:
- JSON 格式的参数快照 (可供后续脚本分析)
- PNG 格式的可视化图表
- 每次运行带时间戳, 绝不覆盖历史数据

观测能力:
1. 各层 Mamba 内部参数 (dt/A/B/C/D) 的分布和变化
2. 不同输入文本触发的隐状态差异对比
3. 训练过程中参数的演化轨迹
"""

import os
import json
import time
from datetime import datetime
from pathlib import Path

import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = ["WenQuanYi Micro Hei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

LOGS_DIR = Path(__file__).parent / "logs"
OBSERVE_DIR = LOGS_DIR / "observations"
OBSERVE_DIR.mkdir(parents=True, exist_ok=True)


def _run_tag() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _save_json(data: dict, path: Path):
    """安全写入 JSON, 自动处理 numpy/torch 类型。"""
    def convert(obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating, torch.Tensor)) and obj.ndim == 0:
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, torch.Tensor):
            return obj.tolist()
        if isinstance(obj, Path):
            return str(obj)
        return obj

    class Encoder(json.JSONEncoder):
        def default(self, o):
            result = convert(o)
            if result is not o:
                return result
            return super().default(o)

    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, cls=Encoder, ensure_ascii=False, indent=2)


def snapshot_mamba_params(model, layer_idx: int = 0, step: int = -1, persist: bool = True) -> dict:
    """
    抓取指定 Mamba 层的关键参数快照。

    Args:
        model: Mamba2LM 模型
        layer_idx: 要观测的层编号
        step: 当前训练步数 (用于文件命名)
        persist: 是否落盘

    Returns:
        各参数的统计信息 dict
    """
    mamba_block = model.layers[layer_idx]["mamba"]
    snapshot = {}
    snapshot_meta = {"step": step, "layer_idx": layer_idx, "timestamp": _run_tag()}

    for name, param in mamba_block.named_parameters():
        data = param.detach().cpu().float()
        snapshot[name] = {
            "mean": data.mean().item(),
            "std": data.std().item(),
            "min": data.min().item(),
            "max": data.max().item(),
            "abs_mean": data.abs().mean().item(),
            "shape": list(data.shape),
            "numel": data.numel(),
        }

    result = {"meta": snapshot_meta, "params": snapshot}

    if persist:
        tag = f"step_{step:06d}" if step >= 0 else _run_tag()
        out_dir = OBSERVE_DIR / "param_snapshots"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"layer{layer_idx}_{tag}.json"
        _save_json(result, out_path)

    return result


def snapshot_all_layers(model, step: int = -1, persist: bool = True) -> list[dict]:
    """对所有层做参数快照。"""
    n_layers = len(model.layers)
    results = []
    for i in range(n_layers):
        snap = snapshot_mamba_params(model, layer_idx=i, step=step, persist=persist)
        results.append(snap)
    return results


def compare_layer_activations(
    model,
    input_ids_a: torch.Tensor,
    input_ids_b: torch.Tensor,
    text_a: str = "",
    text_b: str = "",
    step: int = -1,
    persist: bool = True,
) -> list[dict]:
    """
    对比两段不同输入通过同一个 Mamba 层时, 中间激活值的差异。
    """
    model.eval()
    activations = {"a": [], "b": []}

    hooks = []

    def make_hook(key):
        def hook_fn(module, input, output):
            if isinstance(output, torch.Tensor):
                activations[key].append(output.detach().cpu())
            elif isinstance(output, tuple):
                activations[key].append(output[0].detach().cpu())
        return hook_fn

    for layer in model.layers:
        hooks.append(layer["mamba"].register_forward_hook(make_hook("a")))
    with torch.no_grad():
        model(input_ids_a)
    for h in hooks:
        h.remove()
    hooks.clear()

    for layer in model.layers:
        hooks.append(layer["mamba"].register_forward_hook(make_hook("b")))
    with torch.no_grad():
        model(input_ids_b)
    for h in hooks:
        h.remove()

    diffs = []
    for i, (act_a, act_b) in enumerate(zip(activations["a"], activations["b"])):
        min_len = min(act_a.shape[1], act_b.shape[1])
        a_trim = act_a[:, :min_len, :]
        b_trim = act_b[:, :min_len, :]
        cos_sim = torch.nn.functional.cosine_similarity(
            a_trim.reshape(1, -1), b_trim.reshape(1, -1)
        ).item()
        l2_diff = (a_trim - b_trim).norm().item()
        diffs.append({
            "layer": i,
            "cosine_similarity": cos_sim,
            "l2_distance": l2_diff,
            "act_a_norm": a_trim.norm().item(),
            "act_b_norm": b_trim.norm().item(),
        })

    if persist:
        tag = f"step_{step:06d}" if step >= 0 else _run_tag()
        out_dir = OBSERVE_DIR / "activation_diffs"
        out_dir.mkdir(parents=True, exist_ok=True)
        record = {
            "meta": {"step": step, "timestamp": _run_tag(), "text_a": text_a, "text_b": text_b},
            "diffs": diffs,
        }
        _save_json(record, out_dir / f"diff_{tag}.json")

    return diffs


def plot_param_evolution(history: list[dict], param_name: str, save_dir: str = None) -> str:
    """
    绘制某个参数在训练过程中的演化曲线。

    Returns:
        保存的图片路径
    """
    if save_dir is None:
        save_dir = str(OBSERVE_DIR / "param_evolution")
    os.makedirs(save_dir, exist_ok=True)

    steps = []
    means = []
    stds = []
    abs_means = []

    for h in history:
        params = h.get("params", h)
        if param_name not in params:
            continue
        steps.append(h.get("meta", {}).get("step", len(steps)))
        info = params[param_name]
        means.append(info["mean"])
        stds.append(info["std"])
        abs_means.append(info.get("abs_mean", abs(info["mean"])))

    if len(steps) < 2:
        return ""

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    axes[0].plot(steps, means, "b-o", markersize=2, linewidth=1)
    axes[0].set_title(f"{param_name} — 均值变化")
    axes[0].set_xlabel("训练步数")
    axes[0].set_ylabel("Mean")
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(steps, stds, "r-o", markersize=2, linewidth=1)
    axes[1].set_title(f"{param_name} — 标准差变化")
    axes[1].set_xlabel("训练步数")
    axes[1].set_ylabel("Std")
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(steps, abs_means, "g-o", markersize=2, linewidth=1)
    axes[2].set_title(f"{param_name} — 绝对均值变化")
    axes[2].set_xlabel("训练步数")
    axes[2].set_ylabel("|Mean|")
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    safe_name = param_name.replace(".", "_").replace("/", "_")
    save_path = os.path.join(save_dir, f"evolution_{safe_name}.png")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  参数演化图: {save_path}")
    return save_path


def plot_activation_diff(
    diffs: list[dict],
    text_a: str,
    text_b: str,
    step: int = -1,
    save_dir: str = None,
) -> str:
    """绘制两段文本在各层的激活差异对比图。"""
    if save_dir is None:
        save_dir = str(OBSERVE_DIR / "activation_plots")
    os.makedirs(save_dir, exist_ok=True)

    layers = [d["layer"] for d in diffs]
    cosines = [d["cosine_similarity"] for d in diffs]
    l2s = [d["l2_distance"] for d in diffs]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.bar(layers, cosines, color="steelblue", alpha=0.8)
    ax1.set_title("各层余弦相似度")
    ax1.set_xlabel("层序号")
    ax1.set_ylabel("Cosine Similarity")
    ax1.set_ylim(0, 1.1)
    ax1.grid(True, alpha=0.3)

    ax2.bar(layers, l2s, color="coral", alpha=0.8)
    ax2.set_title("各层 L2 距离")
    ax2.set_xlabel("层序号")
    ax2.set_ylabel("L2 Distance")
    ax2.grid(True, alpha=0.3)

    label_a = text_a[:20] + "..." if len(text_a) > 20 else text_a
    label_b = text_b[:20] + "..." if len(text_b) > 20 else text_b
    fig.suptitle(f"激活差异: 「{label_a}」 vs 「{label_b}」", fontsize=12)
    plt.tight_layout()

    tag = f"step_{step:06d}" if step >= 0 else _run_tag()
    save_path = os.path.join(save_dir, f"activation_diff_{tag}.png")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  激活对比图: {save_path}")
    return save_path


def plot_training_loss(loss_log: list[dict], save_dir: str = None) -> str:
    """绘制并保存训练 Loss 曲线。"""
    if save_dir is None:
        save_dir = str(LOGS_DIR)
    os.makedirs(save_dir, exist_ok=True)

    steps_plot = [x["step"] for x in loss_log]
    losses_plot = [x["loss"] for x in loss_log]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 5))

    ax1.plot(steps_plot, losses_plot, alpha=0.3, color="blue", label="raw loss")
    window = min(50, len(losses_plot) // 5 + 1)
    if window > 1 and len(losses_plot) > window:
        smoothed = np.convolve(losses_plot, np.ones(window)/window, mode="valid")
        ax1.plot(steps_plot[window-1:], smoothed, color="red", linewidth=2,
                 label=f"smooth (w={window})")
    ax1.set_xlabel("Step")
    ax1.set_ylabel("Loss")
    ax1.set_title("训练 Loss 曲线")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    if len(loss_log) > 0 and "lr" in loss_log[0]:
        lrs = [x["lr"] for x in loss_log]
        ax2.plot(steps_plot, lrs, color="green", linewidth=1.5)
        ax2.set_xlabel("Step")
        ax2.set_ylabel("Learning Rate")
        ax2.set_title("学习率调度")
        ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    save_path = os.path.join(save_dir, "loss_curve.png")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Loss 曲线: {save_path}")
    return save_path


def generate_training_report(
    loss_log: list[dict],
    param_history: list[dict],
    model_config: dict,
    training_args: dict,
    save_dir: str = None,
) -> str:
    """
    生成完整的训练报告 (Markdown), 汇总所有关键指标。
    """
    if save_dir is None:
        save_dir = str(LOGS_DIR)
    os.makedirs(save_dir, exist_ok=True)

    report_path = os.path.join(save_dir, "training_report.md")
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    with open(report_path, "w", encoding="utf-8") as f:
        f.write(f"# Mamba-2 中文模型训练报告\n\n")
        f.write(f"**生成时间**: {ts}\n\n")

        f.write("## 模型配置\n\n")
        f.write("| 参数 | 值 |\n|------|----|\n")
        for k, v in model_config.items():
            f.write(f"| {k} | {v} |\n")

        f.write("\n## 训练超参\n\n")
        f.write("| 参数 | 值 |\n|------|----|\n")
        for k, v in training_args.items():
            f.write(f"| {k} | {v} |\n")

        if loss_log:
            final_loss = loss_log[-1]["loss"]
            min_loss = min(x["loss"] for x in loss_log)
            first_loss = loss_log[0]["loss"]
            f.write(f"\n## 训练结果\n\n")
            f.write(f"- **初始 Loss**: {first_loss:.4f}\n")
            f.write(f"- **最终 Loss**: {final_loss:.4f}\n")
            f.write(f"- **最低 Loss**: {min_loss:.4f}\n")
            f.write(f"- **Loss 下降**: {first_loss - final_loss:.4f} ({(first_loss - final_loss)/first_loss*100:.1f}%)\n")
            f.write(f"- **总训练步数**: {loss_log[-1]['step']}\n")

        if param_history:
            f.write(f"\n## 动态学习观测\n\n")
            f.write(f"共采集了 **{len(param_history)}** 次参数快照。\n\n")
            f.write("各参数演化图见 `logs/observations/param_evolution/` 目录。\n")
            f.write("各层激活差异对比见 `logs/observations/activation_plots/` 目录。\n")

        f.write(f"\n## 文件清单\n\n")
        f.write("```\n")
        f.write("checkpoints/     # 模型权重\n")
        f.write("logs/            # 训练日志和可视化\n")
        f.write("  loss_log.json  # 逐步 loss 记录\n")
        f.write("  loss_curve.png # loss 曲线图\n")
        f.write("  observations/  # 动态学习观测数据\n")
        f.write("    param_snapshots/   # 参数快照 (JSON)\n")
        f.write("    param_evolution/   # 参数演化图 (PNG)\n")
        f.write("    activation_diffs/  # 激活差异数据 (JSON)\n")
        f.write("    activation_plots/  # 激活差异图 (PNG)\n")
        f.write("data/            # 语料和分词器\n")
        f.write("```\n")

    print(f"  训练报告: {report_path}")
    return report_path
