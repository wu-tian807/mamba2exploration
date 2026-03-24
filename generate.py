"""
Mamba-2 文本生成脚本 —— 观察模型学到了什么

用法:
    python generate.py --prompt "从前有一个小女孩"
    python generate.py --prompt "今天天气" --temperature 0.8 --top_k 50
    python generate.py --interactive   # 交互模式, 持续对话

    观察动态学习:
    python generate.py --prompt "从前" --observe  # 生成时同时输出各层激活分析
"""

import argparse
import torch
from pathlib import Path

from model import Mamba2LM
from data import load_tokenizer
from observe import compare_layer_activations


CHECKPOINT_DIR = Path(__file__).parent / "checkpoints"


def load_model(checkpoint_path: str, device: str = "cuda"):
    print(f"加载模型: {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = ckpt["config"]

    tokenizer = load_tokenizer()
    config["vocab_size"] = tokenizer.get_vocab_size()

    model = Mamba2LM(**config).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    param_count = sum(p.numel() for p in model.parameters())
    print(f"模型已加载: {param_count / 1e6:.1f}M 参数, step={ckpt.get('step', '?')}")
    return model, tokenizer


@torch.no_grad()
def generate(
    model,
    tokenizer,
    prompt: str,
    max_new_tokens: int = 256,
    temperature: float = 0.9,
    top_k: int = 50,
    top_p: float = 0.95,
    device: str = "cuda",
) -> str:
    encoded = tokenizer.encode(prompt)
    input_ids = torch.tensor([encoded.ids], dtype=torch.long, device=device)

    bos_id = tokenizer.token_to_id("<bos>")
    eos_id = tokenizer.token_to_id("<eos>")

    for _ in range(max_new_tokens):
        logits = model(input_ids)["logits"][:, -1, :]
        logits = logits / temperature

        if top_k > 0:
            v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            logits[logits < v[:, [-1]]] = -float("inf")

        if top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            cumulative_probs = torch.softmax(sorted_logits, dim=-1).cumsum(dim=-1)
            mask = cumulative_probs - torch.softmax(sorted_logits, dim=-1) >= top_p
            sorted_logits[mask] = -float("inf")
            logits = sorted_logits.scatter(1, sorted_indices, sorted_logits)

        probs = torch.softmax(logits, dim=-1)
        next_id = torch.multinomial(probs, num_samples=1)

        if next_id.item() == eos_id:
            break

        input_ids = torch.cat([input_ids, next_id], dim=1)

    output_ids = input_ids[0].tolist()
    if output_ids[0] == bos_id:
        output_ids = output_ids[1:]

    return tokenizer.decode(output_ids)


def interactive_mode(model, tokenizer, device, **gen_kwargs):
    print("\n=== Mamba-2 中文生成 — 交互模式 ===")
    print("输入提示词开始生成, 输入 'quit' 退出\n")

    while True:
        try:
            prompt = input("你: ").strip()
        except (EOFError, KeyboardInterrupt):
            break

        if not prompt or prompt.lower() in ("quit", "exit", "q"):
            print("再见!")
            break

        result = generate(model, tokenizer, prompt, device=device, **gen_kwargs)
        print(f"Mamba: {result}\n")


def observe_generation(model, tokenizer, prompt: str, device: str = "cuda"):
    """生成文本的同时, 观察不同输入的层间激活差异。"""
    from observe import compare_layer_activations, plot_activation_diff

    contrast_prompt = "但是事情并没有那么简单"

    encoded_a = tokenizer.encode(prompt)
    encoded_b = tokenizer.encode(contrast_prompt)
    ids_a = torch.tensor([encoded_a.ids], dtype=torch.long, device=device)
    ids_b = torch.tensor([encoded_b.ids], dtype=torch.long, device=device)

    print(f"\n=== 动态学习观测 ===")
    print(f"文本 A: 「{prompt}」")
    print(f"文本 B: 「{contrast_prompt}」\n")

    diffs = compare_layer_activations(model, ids_a, ids_b)
    for d in diffs:
        sim_bar = "█" * int(d["cosine_similarity"] * 30)
        print(f"  Layer {d['layer']:2d} | cos_sim: {d['cosine_similarity']:.4f} {sim_bar}")

    plot_activation_diff(diffs, prompt, contrast_prompt)

    print(f"\n--- 正常生成 ---")
    result = generate(model, tokenizer, prompt, device=device)
    print(f"Mamba: {result}")


def main():
    parser = argparse.ArgumentParser(description="Mamba-2 中文文本生成")
    parser.add_argument("--checkpoint", type=str, default=str(CHECKPOINT_DIR / "final.pt"))
    parser.add_argument("--prompt", type=str, default="从前有一个小女孩")
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.9)
    parser.add_argument("--top_k", type=int, default=50)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--interactive", action="store_true")
    parser.add_argument("--observe", action="store_true",
                        help="生成时同时输出动态学习观测")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, tokenizer = load_model(args.checkpoint, device)

    gen_kwargs = {
        "max_new_tokens": args.max_new_tokens,
        "temperature": args.temperature,
        "top_k": args.top_k,
        "top_p": args.top_p,
    }

    if args.interactive:
        interactive_mode(model, tokenizer, device, **gen_kwargs)
    elif args.observe:
        observe_generation(model, tokenizer, args.prompt, device)
    else:
        result = generate(model, tokenizer, args.prompt, device=device, **gen_kwargs)
        print(f"\n提示: {args.prompt}")
        print(f"生成: {result}")


if __name__ == "__main__":
    main()
