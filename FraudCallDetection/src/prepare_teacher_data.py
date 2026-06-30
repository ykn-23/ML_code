import argparse
import csv
import random
from collections import Counter, defaultdict
from pathlib import Path


SEED = 42


def normalize_label(value) -> int | None:
    text = str(value).strip().lower()
    if not text:
        return None
    if text in {"true", "1", "1.0", "是", "欺诈", "诈骗", "fraud", "scam"}:
        return 1
    if text in {"false", "0", "0.0", "否", "正常", "非欺诈", "normal", "benign"}:
        return 0
    raise ValueError(f"无法识别标签值: {value}")


def read_teacher_csv(path: Path, text_col: str, label_col: str) -> list[dict]:
    rows = []
    skipped = 0
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        missing = [col for col in [text_col, label_col] if col not in (reader.fieldnames or [])]
        if missing:
            raise ValueError(f"{path} 缺少列 {missing}，当前列名: {reader.fieldnames}")
        for row in reader:
            text = str(row[text_col]).strip()
            label = normalize_label(row[label_col])
            if not text or label is None:
                skipped += 1
                continue
            rows.append({"text": text, "label": label})
    print(f"{path.name}: 有效样本={len(rows)}, 跳过空文本/空标签={skipped}")
    return rows


def validate_label_conflicts(rows: list[dict]) -> None:
    labels_by_text = defaultdict(set)
    for row in rows:
        labels_by_text[row["text"]].add(row["label"])
    conflicts = [text for text, labels in labels_by_text.items() if len(labels) > 1]
    if conflicts:
        raise ValueError(f"同一 text 对应不同 label，冲突文本数量: {len(conflicts)}")


def group_by_text(rows: list[dict]) -> list[list[dict]]:
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["text"]].append(row)
    return list(grouped.values())


def train_valid_split(rows: list[dict], valid_ratio: float) -> tuple[list[dict], list[dict]]:
    validate_label_conflicts(rows)
    rng = random.Random(SEED)
    train_rows, valid_rows = [], []

    rows_by_label = defaultdict(list)
    for row in rows:
        rows_by_label[row["label"]].append(row)

    for label, label_rows in rows_by_label.items():
        groups = group_by_text(label_rows)
        rng.shuffle(groups)
        target_valid = round(len(label_rows) * valid_ratio)
        current_valid = 0
        label_train, label_valid = [], []

        for group in sorted(groups, key=len, reverse=True):
            if current_valid < target_valid:
                label_valid.extend(group)
                current_valid += len(group)
            else:
                label_train.extend(group)

        train_rows.extend(label_train)
        valid_rows.extend(label_valid)

    rng.shuffle(train_rows)
    rng.shuffle(valid_rows)
    return train_rows, valid_rows


def assert_no_overlap(train_rows: list[dict], valid_rows: list[dict], test_rows: list[dict]) -> None:
    train_texts = {row["text"] for row in train_rows}
    valid_texts = {row["text"] for row in valid_rows}
    test_texts = {row["text"] for row in test_rows}
    overlaps = {
        "train-valid": train_texts & valid_texts,
        "train-test": train_texts & test_texts,
        "valid-test": valid_texts & test_texts,
    }
    leaked = {name: texts for name, texts in overlaps.items() if texts}
    if leaked:
        detail = ", ".join(f"{name}: {len(texts)}" for name, texts in leaked.items())
        raise RuntimeError(f"检测到 text 泄漏: {detail}")
    print("已确认 train/valid/test 之间无重复 text。")


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["text", "label"])
        writer.writeheader()
        writer.writerows(rows)


def print_stats(name: str, rows: list[dict]) -> None:
    counts = Counter(row["label"] for row in rows)
    total = len(rows)
    fraud = counts[1]
    normal = counts[0]
    print(
        f"{name}: 样本数={total}, 欺诈={fraud}, 非欺诈={normal}, "
        f"欺诈比例={fraud / total:.4f}"
    )


def main():
    parser = argparse.ArgumentParser(description="按老师给定 train/test 文件准备监督学习数据。")
    parser.add_argument("--train_file", required=True)
    parser.add_argument("--test_file", required=True)
    parser.add_argument("--text_col", default="specific_dialogue_content")
    parser.add_argument("--label_col", default="is_fraud")
    parser.add_argument("--output_dir", default="data/supervised")
    parser.add_argument("--valid_ratio", type=float, default=0.1)
    args = parser.parse_args()

    teacher_train = read_teacher_csv(Path(args.train_file), args.text_col, args.label_col)
    teacher_test = read_teacher_csv(Path(args.test_file), args.text_col, args.label_col)
    train_rows, valid_rows = train_valid_split(teacher_train, args.valid_ratio)
    validate_label_conflicts(teacher_test)
    assert_no_overlap(train_rows, valid_rows, teacher_test)

    output_dir = Path(args.output_dir)
    write_csv(output_dir / "train.csv", train_rows)
    write_csv(output_dir / "valid.csv", valid_rows)
    write_csv(output_dir / "test.csv", teacher_test)

    print(f"输出目录: {output_dir.resolve()}")
    print_stats("train.csv", train_rows)
    print_stats("valid.csv", valid_rows)
    print_stats("test.csv", teacher_test)


if __name__ == "__main__":
    main()
