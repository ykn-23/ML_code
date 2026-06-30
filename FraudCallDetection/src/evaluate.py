import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from utils import (
    check_columns,
    compute_metrics_from_arrays,
    dataset_name_from_path,
    ensure_dir,
    normalize_label,
    read_table,
    save_json,
    set_seed,
)


SUMMARY_COLUMNS = [
    "dataset",
    "accuracy",
    "precision",
    "recall",
    "f1",
    "fraud_detection_rate",
    "attack_success_rate",
    "fraud_total",
    "fraud_detected",
    "fraud_missed",
]


def predict_texts(model, tokenizer, texts, batch_size: int, max_length: int, device: str):
    preds, probs = [], []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(texts), batch_size):
            batch_texts = texts[start : start + batch_size]
            inputs = tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            ).to(device)
            outputs = model(**inputs)
            batch_probs = torch.softmax(outputs.logits, dim=-1)
            preds.extend(torch.argmax(batch_probs, dim=-1).cpu().numpy().tolist())
            probs.extend(batch_probs[:, 1].cpu().numpy().tolist())
    return np.array(preds), np.array(probs)


def load_model(model_dir: str):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir).to(device)
    return model, tokenizer, device


def update_summary(results_dir: Path, dataset_name: str, metrics: dict) -> None:
    summary_path = results_dir / "summary.csv"
    row = {"dataset": dataset_name, **{col: metrics.get(col, "") for col in SUMMARY_COLUMNS if col != "dataset"}}

    if summary_path.exists():
        summary_df = pd.read_csv(summary_path)
        summary_df = summary_df[summary_df["dataset"] != dataset_name]
        summary_df = pd.concat([summary_df, pd.DataFrame([row])], ignore_index=True)
    else:
        summary_df = pd.DataFrame([row])

    order = [
        "original_test",
        "textfooler_original",
        "textfooler_attack",
        "textfooler",
        "fraudr1_round1_trust",
        "fraudr1_round2_trust_urgency",
        "fraudr1_round3_trust_urgency_emotion",
    ]
    summary_df["sort_key"] = summary_df["dataset"].apply(lambda x: order.index(x) if x in order else len(order))
    summary_df = summary_df.sort_values(["sort_key", "dataset"]).drop(columns=["sort_key"])
    summary_df = summary_df.reindex(columns=SUMMARY_COLUMNS)
    summary_df.to_csv(summary_path, index=False, encoding="utf-8-sig")


def save_latex_table(results_dir: Path) -> None:
    summary_path = results_dir / "summary.csv"
    if not summary_path.exists():
        return
    df = pd.read_csv(summary_path)
    rename = {
        "dataset": "Dataset",
        "accuracy": "Accuracy",
        "precision": "Precision",
        "recall": "Recall",
        "f1": "F1-score",
        "fraud_detection_rate": "Fraud Detection Rate",
        "attack_success_rate": "Attack Success Rate",
    }
    table_df = df[list(rename.keys())].rename(columns=rename)
    for col in table_df.columns:
        if col != "Dataset":
            table_df[col] = table_df[col].map(lambda x: f"{float(x):.4f}" if pd.notna(x) else "")
    (results_dir / "summary_table.md").write_text(table_df.to_markdown(index=False), encoding="utf-8")
    (results_dir / "summary_table.tex").write_text(table_df.to_latex(index=False, escape=False), encoding="utf-8")


def evaluate_dataframe(
    df: pd.DataFrame,
    model,
    tokenizer,
    device: str,
    dataset_name: str,
    text_col: str,
    label_col: str,
    results_dir: Path,
    batch_size: int,
    max_length: int,
) -> dict:
    check_columns(df, [text_col, label_col])
    work_df = df.dropna(subset=[text_col, label_col]).copy()
    texts = work_df[text_col].astype(str).tolist()
    try:
        labels = work_df[label_col].astype(int).to_numpy()
    except ValueError:
        labels = work_df[label_col].apply(normalize_label).to_numpy()

    preds, fraud_probs = predict_texts(model, tokenizer, texts, batch_size, max_length, device)
    metrics = compute_metrics_from_arrays(labels, preds)

    save_json(metrics, results_dir / f"metrics_{dataset_name}.json")
    output_df = work_df.copy()
    output_df["pred_label"] = preds
    output_df["fraud_probability"] = fraud_probs
    output_df.to_csv(results_dir / f"predictions_{dataset_name}.csv", index=False, encoding="utf-8-sig")
    update_summary(results_dir, dataset_name, metrics)
    save_latex_table(results_dir)
    return metrics


def print_metrics(dataset_name: str, metrics: dict, results_dir: Path) -> None:
    print(f"评估完成: {dataset_name}")
    print(f"Accuracy:              {metrics['accuracy']:.4f}")
    print(f"Precision:             {metrics['precision']:.4f}")
    print(f"Recall:                {metrics['recall']:.4f}")
    print(f"F1-score:              {metrics['f1']:.4f}")
    print(f"Fraud detection rate:  {metrics['fraud_detection_rate']:.4f}")
    print(f"Attack success rate:   {metrics['attack_success_rate']:.4f}")
    print(f"Confusion matrix [[TN, FP], [FN, TP]]: {metrics['confusion_matrix']}")
    print(f"结果目录: {results_dir.resolve()}")


def main():
    parser = argparse.ArgumentParser(description="评估模型在原始/攻击测试集上的表现。")
    parser.add_argument("--model_dir", default="saved_model")
    parser.add_argument("--test_file", default="data/test.csv")
    parser.add_argument("--sheet_name", default=0)
    parser.add_argument("--text_col", default="text")
    parser.add_argument("--label_col", default="label")
    parser.add_argument("--dataset_name", default=None)
    parser.add_argument("--results_dir", default="results")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--max_length", type=int, default=256)
    args = parser.parse_args()

    set_seed()
    results_dir = ensure_dir(args.results_dir)
    sheet_name = int(args.sheet_name) if str(args.sheet_name).isdigit() else args.sheet_name
    dataset_name = args.dataset_name or dataset_name_from_path(args.test_file)
    df = read_table(args.test_file, sheet_name=sheet_name)
    model, tokenizer, device = load_model(args.model_dir)
    metrics = evaluate_dataframe(
        df=df,
        model=model,
        tokenizer=tokenizer,
        device=device,
        dataset_name=dataset_name,
        text_col=args.text_col,
        label_col=args.label_col,
        results_dir=results_dir,
        batch_size=args.batch_size,
        max_length=args.max_length,
    )
    print_metrics(dataset_name, metrics, results_dir)


if __name__ == "__main__":
    main()
