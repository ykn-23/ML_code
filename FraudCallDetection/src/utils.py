import json
import random
from pathlib import Path
from typing import Iterable, Union

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score


SEED = 42


def set_seed(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def ensure_dir(path: Union[str, Path]) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def read_table(file_path: Union[str, Path], sheet_name=0) -> pd.DataFrame:
    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"找不到数据文件: {file_path}")

    suffix = file_path.suffix.lower()
    if suffix == ".csv":
        try:
            return pd.read_csv(file_path, encoding="utf-8-sig")
        except UnicodeDecodeError:
            return pd.read_csv(file_path, encoding="gbk")
    if suffix in {".xlsx", ".xls"}:
        return pd.read_excel(file_path, sheet_name=sheet_name)

    raise ValueError("仅支持 .csv、.xlsx、.xls 文件。")


def check_columns(df: pd.DataFrame, required_cols: Iterable[str]) -> None:
    missing = [col for col in required_cols if col and col not in df.columns]
    if missing:
        raise ValueError(
            f"找不到列名: {missing}\n"
            f"当前文件列名: {df.columns.tolist()}\n"
            "请先运行 src/inspect_data.py 查看列名，再重新指定参数。"
        )


def normalize_label(value) -> int:
    if pd.isna(value):
        raise ValueError("标签列存在空值，请先清洗数据。")

    text = str(value).strip().lower()
    fraud_values = {"1", "1.0", "欺诈", "诈骗", "虚假", "fraud", "fake", "scam", "yes", "true"}
    normal_values = {"0", "0.0", "正常", "真实", "非欺诈", "非诈骗", "normal", "real", "benign", "no", "false"}

    if text in fraud_values:
        return 1
    if text in normal_values:
        return 0

    raise ValueError(f"无法识别标签值 {value}，请整理为 fraud/1 或 normal/0。")


def compute_metrics_from_arrays(y_true, y_pred) -> dict:
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    fraud_mask = y_true == 1
    fraud_total = int(fraud_mask.sum())
    fraud_detected = int(((y_pred == 1) & fraud_mask).sum())
    fraud_missed = int(((y_pred == 0) & fraud_mask).sum())

    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "fraud_detection_rate": float(fraud_detected / fraud_total) if fraud_total else 0.0,
        "attack_success_rate": float(fraud_missed / fraud_total) if fraud_total else 0.0,
        "fraud_total": fraud_total,
        "fraud_detected": fraud_detected,
        "fraud_missed": fraud_missed,
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=[0, 1]).tolist(),
    }
    return metrics


def save_json(data: dict, path: Union[str, Path]) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def dataset_name_from_path(path: Union[str, Path]) -> str:
    name = Path(path).stem
    mapping = {
        "test": "original_test",
        "test_textfooler": "textfooler",
        "test_round1_trust": "fraudr1_round1_trust",
        "test_round2_trust_urgency": "fraudr1_round2_trust_urgency",
        "test_round3_trust_urgency_emotion": "fraudr1_round3_trust_urgency_emotion",
    }
    return mapping.get(name, name)
