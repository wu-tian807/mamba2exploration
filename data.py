"""
数据管道: 中文分词器训练 + TinyStories-zh 数据加载

两个核心功能:
1. train_tokenizer() - 在你的中文语料上训练 32K BPE 分词器
2. build_dataloader() - 把文本切成定长 token 序列, 喂给模型
"""

import os
import json
from pathlib import Path

import torch
from torch.utils.data import Dataset, DataLoader
from tokenizers import Tokenizer, models, trainers, pre_tokenizers, processors, decoders

DATA_DIR = Path(__file__).parent / "data"
TOKENIZER_PATH = DATA_DIR / "zh_tokenizer.json"


def train_tokenizer(
    corpus_files: list[str],
    vocab_size: int = 32000,
    save_path: str = str(TOKENIZER_PATH),
) -> Tokenizer:
    """
    在给定的中文语料上训练 BPE 分词器。

    Args:
        corpus_files: 文本文件路径列表, 每个文件是纯文本 (.txt)
        vocab_size: 词表大小
        save_path: 分词器保存路径
    """
    tokenizer = Tokenizer(models.BPE(unk_token="<unk>"))

    tokenizer.pre_tokenizer = pre_tokenizers.Sequence([
        pre_tokenizers.UnicodeScripts(),
        pre_tokenizers.ByteLevel(add_prefix_space=False),
    ])

    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=["<pad>", "<unk>", "<bos>", "<eos>"],
        show_progress=True,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
    )

    tokenizer.train(files=corpus_files, trainer=trainer)

    tokenizer.post_processor = processors.TemplateProcessing(
        single="<bos> $A <eos>",
        special_tokens=[("<bos>", tokenizer.token_to_id("<bos>")), ("<eos>", tokenizer.token_to_id("<eos>"))],
    )
    tokenizer.decoder = decoders.ByteLevel()

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    tokenizer.save(save_path)
    print(f"分词器已保存到 {save_path}, 词表大小: {tokenizer.get_vocab_size()}")
    return tokenizer


def load_tokenizer(path: str = str(TOKENIZER_PATH)) -> Tokenizer:
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"分词器不存在: {path}\n"
            "请先运行: python prepare_data.py --action train_tokenizer"
        )
    return Tokenizer.from_file(path)


class TextTokenDataset(Dataset):
    """
    将 token 化后的长序列切成等长块。
    连续切割, 不丢弃任何数据 (类似 GPT 预训练方式)。
    """

    def __init__(self, token_ids: list[int], seq_len: int = 2048):
        self.seq_len = seq_len
        n_chunks = len(token_ids) // seq_len
        self.data = torch.tensor(
            token_ids[:n_chunks * seq_len],
            dtype=torch.long,
        ).view(n_chunks, seq_len)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x = self.data[idx]
        return {"input_ids": x, "labels": x}


def prepare_corpus_from_tinystories_zh(output_file: str = str(DATA_DIR / "corpus.txt")):
    """
    下载中文 TinyStories 数据集并合并为纯文本文件。
    尝试多个数据源，优先选择数据量最大的。
    """
    from datasets import load_dataset

    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    datasets_to_try = [
        ("BrianatCambridge/TinyStoryInChinese", "train", ["story", "text", "content"]),
        ("adam89/TinyStoriesChinese", "train", ["story_zh", "story", "text"]),
    ]

    max_stories = 20000  # 限制数量，避免 OOM

    for ds_name, split, text_fields in datasets_to_try:
        try:
            print(f"尝试下载 (streaming): {ds_name} ...")
            ds = load_dataset(ds_name, split=split, streaming=True)

            count = 0
            with open(output_file, "w", encoding="utf-8") as f:
                for item in ds:
                    text = ""
                    for field in text_fields:
                        if field in item and item[field] and str(item[field]).strip():
                            text = str(item[field]).strip()
                            break
                    if text and len(text) > 20:
                        f.write(text + "\n\n")
                        count += 1
                        if count % 1000 == 0:
                            print(f"  已下载 {count} 条...")
                    if count >= max_stories:
                        break

            if count > 100:
                print(f"语料已保存到 {output_file}, 有效故事: {count} 条")
                return output_file
            else:
                print(f"  数据量太少 ({count}), 跳过")
                continue
        except Exception as e:
            print(f"  失败: {e}")
            continue

    raise RuntimeError("所有中文 TinyStories 数据源均不可用，请手动提供语料文件")


def build_dataloader(
    corpus_file: str,
    tokenizer_path: str = str(TOKENIZER_PATH),
    seq_len: int = 2048,
    batch_size: int = 8,
    num_workers: int = 4,
) -> DataLoader:
    """
    完整的数据管道: 读取语料 -> 分词 -> 切块 -> DataLoader
    """
    tokenizer = load_tokenizer(tokenizer_path)

    print(f"正在对语料进行分词: {corpus_file}")
    with open(corpus_file, "r", encoding="utf-8") as f:
        text = f.read()

    encoded = tokenizer.encode(text)
    token_ids = encoded.ids
    print(f"  总 token 数: {len(token_ids):,}")
    print(f"  序列长度: {seq_len}")
    print(f"  训练样本数: {len(token_ids) // seq_len:,}")

    dataset = TextTokenDataset(token_ids, seq_len=seq_len)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )
