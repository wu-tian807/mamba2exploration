"""
Mamba-2 中文模型训练主循环

特色:
- BF16 混合精度 + GradScaler
- 动态学习观测点: 每 N 步自动抓取全层 Mamba 参数快照并落盘
- 训练结束后生成完整报告 + 所有可视化
- 余弦退火学习率调度
- 梯度裁剪
- checkpoint 自动保存/恢复

用法:
    python train.py
    python train.py --resume checkpoints/step_5000.pt
"""

import os
import sys
import time
import math
import json
import argparse
from pathlib import Path

import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler

from model import Mamba2LM, build_model
from data import build_dataloader, load_tokenizer, DATA_DIR, TOKENIZER_PATH
from observe import (
    snapshot_mamba_params,
    snapshot_all_layers,
    plot_param_evolution,
    plot_training_loss,
    generate_training_report,
    OBSERVE_DIR,
    LOGS_DIR as OBS_LOGS_DIR,
)

CHECKPOINT_DIR = Path(__file__).parent / "checkpoints"
LOGS_DIR = Path(__file__).parent / "logs"
CHECKPOINT_DIR.mkdir(exist_ok=True)
LOGS_DIR.mkdir(exist_ok=True)


def get_lr(step: int, warmup_steps: int, max_steps: int, max_lr: float, min_lr: float) -> float:
    if step < warmup_steps:
        return max_lr * (step + 1) / warmup_steps
    if step >= max_steps:
        return min_lr
    progress = (step - warmup_steps) / (max_steps - warmup_steps)
    return min_lr + 0.5 * (max_lr - min_lr) * (1.0 + math.cos(math.pi * progress))


def train(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"设备: {device}")
    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name()}")
        print(f"显存: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    tokenizer = load_tokenizer()
    vocab_size = tokenizer.get_vocab_size()
    print(f"词表大小: {vocab_size}")

    model = build_model(vocab_size=vocab_size, device=device)

    dataloader = build_dataloader(
        corpus_file=args.corpus,
        tokenizer_path=str(TOKENIZER_PATH),
        seq_len=args.seq_len,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.max_lr,
        betas=(0.9, 0.95),
        weight_decay=0.1,
    )

    scaler = GradScaler(enabled=(args.precision == "bf16"))
    start_step = 0
    param_history = []

    if args.resume:
        print(f"恢复训练: {args.resume}")
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        start_step = ckpt["step"]
        param_history = ckpt.get("param_history", [])
        print(f"  从第 {start_step} 步继续")

    max_steps = args.max_steps
    log_interval = args.log_interval
    save_interval = args.save_interval
    observe_interval = args.observe_interval

    training_args_dict = vars(args)

    loss_log = []
    tokens_processed = 0
    t0 = time.time()

    print(f"\n{'='*60}")
    print(f"开始训练! 总步数: {max_steps}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  序列长度: {args.seq_len}")
    print(f"  每步 tokens: {args.batch_size * args.seq_len:,}")
    print(f"  精度: {args.precision}")
    print(f"  动态学习观测间隔: 每 {observe_interval} 步")
    print(f"  观测数据目录: {OBSERVE_DIR}")
    print(f"{'='*60}\n")

    # 训练前做一次初始快照 (step=0)
    print("[观测] 训练前初始参数快照...")
    init_snap = snapshot_mamba_params(model, layer_idx=0, step=0, persist=True)
    param_history.append(init_snap)

    model.train()
    data_iter = iter(dataloader)
    step = start_step

    while step < max_steps:
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(dataloader)
            batch = next(data_iter)

        input_ids = batch["input_ids"].to(device)
        labels = batch["labels"].to(device)

        lr = get_lr(step, args.warmup_steps, max_steps, args.max_lr, args.min_lr)
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

        autocast_dtype = torch.bfloat16 if args.precision == "bf16" else torch.float32
        with torch.autocast(device_type="cuda", dtype=autocast_dtype):
            outputs = model(input_ids, labels=labels)
            loss = outputs["loss"]

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)

        tokens_processed += input_ids.numel()
        loss_val = loss.item()
        elapsed = time.time() - t0
        tps = tokens_processed / elapsed if elapsed > 0 else 0

        loss_log.append({
            "step": step,
            "loss": loss_val,
            "lr": lr,
            "grad_norm": float(grad_norm),
            "tokens_per_sec": tps,
            "elapsed_sec": elapsed,
            "gpu_mem_mb": torch.cuda.memory_allocated() / 1e6 if device == "cuda" else 0,
        })

        if step % log_interval == 0:
            print(
                f"步 {step:>6d}/{max_steps} | "
                f"loss: {loss_val:.4f} | "
                f"lr: {lr:.2e} | "
                f"grad_norm: {grad_norm:.2f} | "
                f"tokens/s: {tps:,.0f} | "
                f"显存: {torch.cuda.memory_allocated()/1e6:.0f}MB | "
                f"已用时: {elapsed:.0f}s"
            )

        # === 动态学习观测点 ===
        if step > 0 and step % observe_interval == 0:
            print(f"\n  [观测] 第 {step} 步 — 抓取 Mamba 参数快照并落盘...")
            snap = snapshot_mamba_params(model, layer_idx=0, step=step, persist=True)
            param_history.append(snap)

            params = snap["params"]
            key_params = [k for k in params.keys() if any(
                kw in k.lower() for kw in ["dt", "a_log", "d", "norm"]
            )]
            if not key_params:
                key_params = list(params.keys())[:3]
            for pname in key_params:
                info = params[pname]
                print(
                    f"    {pname}: "
                    f"mean={info['mean']:.6f}, "
                    f"std={info['std']:.6f}, "
                    f"range=[{info['min']:.4f}, {info['max']:.4f}]"
                )

            # 每个观测点同时保存一次全层快照
            if step % (observe_interval * 5) == 0:
                print(f"  [观测] 全层快照 (16层)...")
                snapshot_all_layers(model, step=step, persist=True)

            print()

        if step > 0 and step % save_interval == 0:
            ckpt_path = CHECKPOINT_DIR / f"step_{step}.pt"
            torch.save({
                "step": step,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "config": model.config,
                "param_history": param_history,
                "loss_log": loss_log,
            }, ckpt_path)
            print(f"  [保存] Checkpoint: {ckpt_path}")

            # 每次保存 checkpoint 时也落盘 loss_log
            with open(LOGS_DIR / "loss_log.json", "w", encoding="utf-8") as f:
                json.dump(loss_log, f, ensure_ascii=False)

        step += 1

    # ===== 训练结束: 保存所有产物 =====
    total_time = time.time() - t0
    print(f"\n{'='*60}")
    print(f"训练完成! 总用时: {total_time:.0f}s ({total_time/3600:.1f}h)")
    print(f"{'='*60}")

    # 最终模型
    final_path = CHECKPOINT_DIR / "final.pt"
    torch.save({
        "step": step,
        "model": model.state_dict(),
        "config": model.config,
        "param_history": param_history,
        "loss_log": loss_log,
    }, final_path)
    print(f"最终模型: {final_path}")

    # Loss 日志
    with open(LOGS_DIR / "loss_log.json", "w", encoding="utf-8") as f:
        json.dump(loss_log, f, ensure_ascii=False)
    print(f"Loss 日志: {LOGS_DIR / 'loss_log.json'}")

    # Loss 曲线
    plot_training_loss(loss_log)

    # 参数演化图
    if len(param_history) > 1:
        print("\n绘制动态学习参数演化图...")
        first_params = param_history[0].get("params", param_history[0])
        for pname in first_params.keys():
            try:
                plot_param_evolution(param_history, pname)
            except Exception as e:
                print(f"  跳过 {pname}: {e}")

    # 训练报告
    generate_training_report(
        loss_log=loss_log,
        param_history=param_history,
        model_config=model.config,
        training_args=training_args_dict,
    )

    print(f"\n所有训练产物已保存完毕!")


def main():
    parser = argparse.ArgumentParser(description="Mamba-2 中文 LM 训练")
    parser.add_argument("--corpus", type=str, default=str(DATA_DIR / "corpus.txt"))
    parser.add_argument("--seq_len", type=int, default=2048)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--max_steps", type=int, default=10000)
    parser.add_argument("--max_lr", type=float, default=3e-4)
    parser.add_argument("--min_lr", type=float, default=3e-5)
    parser.add_argument("--warmup_steps", type=int, default=200)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--precision", choices=["bf16", "fp32"], default="bf16")
    parser.add_argument("--log_interval", type=int, default=10)
    parser.add_argument("--save_interval", type=int, default=1000)
    parser.add_argument("--observe_interval", type=int, default=200,
                        help="每隔多少步抓取一次 Mamba 参数快照 (动态学习观测)")
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--resume", type=str, default=None)
    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
