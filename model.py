"""
Mamba-2 语言模型 —— 动态学习研究用架构

核心看点: Mamba2 使用 SSD (Structured State Space Duality) 算法,
比 Mamba-1 更好地利用 Tensor Core, 同时保持了选择性状态空间的动态学习能力。
"""

import torch
import torch.nn as nn
from mamba_ssm import Mamba2


class Mamba2LM(nn.Module):
    """
    基于 Mamba-2 的因果语言模型。

    架构: Embedding -> [RMSNorm -> Mamba2 + Residual] x N -> RMSNorm -> LM Head
    """

    def __init__(
        self,
        vocab_size: int = 32000,
        d_model: int = 768,
        n_layer: int = 16,
        d_state: int = 64,      # Mamba-2 推荐更大的 state dim
        d_conv: int = 4,
        expand: int = 2,
        pad_token_id: int = 0,
    ):
        super().__init__()
        self.config = {
            "vocab_size": vocab_size,
            "d_model": d_model,
            "n_layer": n_layer,
            "d_state": d_state,
            "d_conv": d_conv,
            "expand": expand,
        }

        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=pad_token_id)

        self.layers = nn.ModuleList([
            nn.ModuleDict({
                "norm": nn.RMSNorm(d_model),
                "mamba": Mamba2(
                    d_model=d_model,
                    d_state=d_state,
                    d_conv=d_conv,
                    expand=expand,
                    headdim=64,
                ),
            })
            for _ in range(n_layer)
        ])

        self.norm_f = nn.RMSNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

        # 权重共享: embedding 和 lm_head 共用权重, 减少参数量
        self.lm_head.weight = self.embedding.weight

        self._init_weights()

    def _init_weights(self):
        for name, p in self.named_parameters():
            if "embedding" in name or "lm_head" in name:
                continue
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, input_ids: torch.Tensor, labels: torch.Tensor = None):
        x = self.embedding(input_ids)

        for layer in self.layers:
            residual = x
            x = layer["norm"](x)
            x = layer["mamba"](x) + residual

        x = self.norm_f(x)
        logits = self.lm_head(x)

        loss = None
        if labels is not None:
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()
            loss = nn.functional.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                ignore_index=0,
            )

        return {"loss": loss, "logits": logits}

    def count_parameters(self) -> dict:
        total = sum(p.numel() for p in self.parameters())
        embed_params = sum(p.numel() for n, p in self.named_parameters() if "embedding" in n)
        mamba_params = sum(p.numel() for n, p in self.named_parameters() if "mamba" in n)
        other_params = total - embed_params - mamba_params
        return {
            "total": total,
            "embedding": embed_params,
            "mamba_layers": mamba_params,
            "other (norm + shared_head)": other_params,
            "mamba_ratio": f"{mamba_params / total * 100:.1f}%",
        }


def build_model(vocab_size: int = 32000, d_model: int = 768, n_layer: int = 16, device: str = "cuda") -> Mamba2LM:
    model = Mamba2LM(vocab_size=vocab_size, d_model=d_model, n_layer=n_layer)
    print("\n=== Mamba-2 LM 参数分配 ===")
    for k, v in model.count_parameters().items():
        if isinstance(v, int):
            print(f"  {k}: {v:,} ({v / 1e6:.1f}M)")
        else:
            print(f"  {k}: {v}")
    print()
    return model.to(device)
