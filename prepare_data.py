"""
数据准备脚本 —— 一键完成: 下载语料 -> 训练分词器 -> 预览效果

用法:
    python prepare_data.py --action all            # 全部执行
    python prepare_data.py --action download       # 只下载语料
    python prepare_data.py --action train_tokenizer # 只训练分词器
    python prepare_data.py --action preview        # 预览分词效果

    也可以用自己的语料:
    python prepare_data.py --action train_tokenizer --corpus /path/to/your/novel.txt
"""

import argparse
from data import (
    train_tokenizer,
    load_tokenizer,
    prepare_corpus_from_tinystories_zh,
    DATA_DIR,
    TOKENIZER_PATH,
)


def preview_tokenizer():
    tokenizer = load_tokenizer()
    test_texts = [
        "从前有一个小女孩，她叫小红。",
        "人工智能正在改变世界的运作方式。",
        "今天天气真好，我们去公园玩吧！",
        "倒吸一口凉气，此子恐怖如斯。",
        "但是，事情并没有那么简单。因为他忘记了一件非常重要的事情。",
    ]
    print("\n=== 分词器效果预览 ===")
    print(f"词表大小: {tokenizer.get_vocab_size()}\n")
    for text in test_texts:
        encoded = tokenizer.encode(text)
        tokens = encoded.tokens
        print(f"原文: {text}")
        print(f"  Token 数: {len(tokens)}")
        print(f"  Tokens: {tokens[:30]}{'...' if len(tokens) > 30 else ''}")
        decoded = tokenizer.decode(encoded.ids)
        print(f"  还原: {decoded}")
        print()


def main():
    parser = argparse.ArgumentParser(description="Mamba-2 中文数据准备")
    parser.add_argument("--action", choices=["all", "download", "train_tokenizer", "preview"],
                        default="all")
    parser.add_argument("--corpus", type=str, default=None,
                        help="自定义语料文件路径 (纯文本 .txt)")
    parser.add_argument("--vocab_size", type=int, default=32000)
    args = parser.parse_args()

    corpus_file = args.corpus or str(DATA_DIR / "corpus.txt")

    if args.action in ("all", "download"):
        if args.corpus:
            print(f"使用自定义语料: {args.corpus}")
        else:
            prepare_corpus_from_tinystories_zh(corpus_file)

    if args.action in ("all", "train_tokenizer"):
        print(f"\n--- 训练分词器 (vocab_size={args.vocab_size}) ---")
        train_tokenizer([corpus_file], vocab_size=args.vocab_size)

    if args.action in ("all", "preview"):
        preview_tokenizer()


if __name__ == "__main__":
    main()
