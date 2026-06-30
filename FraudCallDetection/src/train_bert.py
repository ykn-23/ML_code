import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from datasets import Dataset
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

from utils import SEED, ensure_dir, set_seed


def load_csv_dataset(path: str) -> Dataset:
    df = pd.read_csv(path)
    required = {"text", "label"}
    if not required.issubset(df.columns):
        raise ValueError(f"{path} 必须包含 text 和 label 两列。")
    df = df[["text", "label"]].dropna()
    df["text"] = df["text"].astype(str)
    df["label"] = df["label"].astype(int)
    return Dataset.from_pandas(df, preserve_index=False)


def tokenize_dataset(dataset: Dataset, tokenizer, max_length: int) -> Dataset:
    def tokenize(batch):
        return tokenizer(batch["text"], truncation=True, max_length=max_length)

    return dataset.map(tokenize, batched=True)


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {
        "accuracy": accuracy_score(labels, preds),
        "precision": precision_score(labels, preds, zero_division=0),
        "recall": recall_score(labels, preds, zero_division=0),
        "f1": f1_score(labels, preds, zero_division=0),
    }


def parse_args():
    parser = argparse.ArgumentParser(description="使用中文 BERT/RoBERTa 训练虚假通话二分类模型。")
    parser.add_argument("--train_file", default="data/processed/train.csv")
    parser.add_argument("--valid_file", default="data/processed/valid.csv")
    parser.add_argument("--model_name", default="hfl/chinese-roberta-wwm-ext")
    parser.add_argument("--fallback_model_name", default="bert-base-chinese")
    parser.add_argument("--output_dir", default="saved_model")
    parser.add_argument("--max_length", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--hf_endpoint", default=None, help="HuggingFace 镜像地址，例如 https://hf-mirror.com。")
    parser.add_argument("--cache_dir", default=None, help="模型缓存目录。")
    parser.add_argument("--download_timeout", type=int, default=60, help="模型下载超时时间，单位秒。")
    parser.add_argument("--local_files_only", action="store_true", help="只从本地缓存或本地模型目录加载。")
    return parser.parse_args()


def configure_huggingface_env(args) -> None:
    if args.hf_endpoint:
        os.environ["HF_ENDPOINT"] = args.hf_endpoint
        print(f"使用 HuggingFace endpoint: {args.hf_endpoint}")
    os.environ["HF_HUB_DOWNLOAD_TIMEOUT"] = str(args.download_timeout)
    os.environ["HF_HUB_ETAG_TIMEOUT"] = str(args.download_timeout)


def import_transformers_after_env():
    from transformers import (
        AutoModelForSequenceClassification,
        AutoTokenizer,
        DataCollatorWithPadding,
        Trainer,
        TrainingArguments,
    )

    return AutoModelForSequenceClassification, AutoTokenizer, DataCollatorWithPadding, Trainer, TrainingArguments


def from_pretrained_kwargs(args) -> dict:
    kwargs = {
        "cache_dir": args.cache_dir,
        "local_files_only": args.local_files_only,
    }
    return {key: value for key, value in kwargs.items() if value not in [None, False]}


def load_model_and_tokenizer(args, auto_model_cls, auto_tokenizer_cls):
    kwargs = from_pretrained_kwargs(args)
    candidates = [args.model_name]
    if args.fallback_model_name and args.fallback_model_name != args.model_name:
        candidates.append(args.fallback_model_name)

    last_error = None
    for model_name in candidates:
        try:
            tokenizer = auto_tokenizer_cls.from_pretrained(model_name, **kwargs)
            model = auto_model_cls.from_pretrained(model_name, num_labels=2, **kwargs)
            print(f"已加载模型: {model_name}")
            return model, tokenizer
        except Exception as exc:
            last_error = exc
            print(f"加载 {model_name} 失败: {exc}")

    raise RuntimeError(
        "无法加载预训练模型。请检查网络、镜像站、缓存目录，或把 --model_name 指向本地模型目录。"
    ) from last_error


def main():
    args = parse_args()
    configure_huggingface_env(args)
    AutoModelForSequenceClassification, AutoTokenizer, DataCollatorWithPadding, Trainer, TrainingArguments = (
        import_transformers_after_env()
    )

    set_seed(SEED)
    output_dir = ensure_dir(args.output_dir)

    model, tokenizer = load_model_and_tokenizer(args, AutoModelForSequenceClassification, AutoTokenizer)
    train_dataset = tokenize_dataset(load_csv_dataset(args.train_file), tokenizer, args.max_length)
    valid_dataset = tokenize_dataset(load_csv_dataset(args.valid_file), tokenizer, args.max_length)

    common_args = {
        "output_dir": str(output_dir),
        "save_strategy": "epoch",
        "learning_rate": args.learning_rate,
        "per_device_train_batch_size": args.batch_size,
        "per_device_eval_batch_size": args.batch_size,
        "num_train_epochs": args.epochs,
        "weight_decay": 0.01,
        "logging_steps": 20,
        "load_best_model_at_end": True,
        "metric_for_best_model": "f1",
        "greater_is_better": True,
        "seed": SEED,
        "report_to": "none",
    }
    try:
        training_args = TrainingArguments(eval_strategy="epoch", **common_args)
    except TypeError:
        training_args = TrainingArguments(evaluation_strategy="epoch", **common_args)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=valid_dataset,
        tokenizer=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
        compute_metrics=compute_metrics,
    )

    trainer.train()
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)

    device = "GPU" if torch.cuda.is_available() else "CPU"
    print(f"训练完成，模型保存到: {Path(output_dir).resolve()}，设备: {device}")


if __name__ == "__main__":
    main()
