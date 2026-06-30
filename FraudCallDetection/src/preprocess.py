import argparse
import csv
import random
from collections import Counter, defaultdict
from pathlib import Path


SEED = 42


def normalize_label(value: str) -> int | None:
    text = str(value).strip().lower()
    if not text:
        return None
    if text in {"true", "1", "1.0", "是", "欺诈", "诈骗", "fraud", "scam"}:
        return 1
    if text in {"false", "0", "0.0", "否", "正常", "非欺诈", "normal", "benign"}:
        return 0
    raise ValueError(f"无法识别 is_fraud 标签值: {value}")


def read_official_csv(path: Path, text_col: str, label_col: str) -> list[dict]:
    rows = []
    skipped_empty = 0

    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames or []
        missing = [col for col in [text_col, label_col] if col not in fieldnames]
        if missing:
            raise ValueError(f"{path} 缺少列 {missing}，当前列名: {fieldnames}")

        for row in reader:
            text = str(row[text_col]).strip()
            label = normalize_label(row[label_col])
            if not text or label is None:
                skipped_empty += 1
                continue
            rows.append({"text": text, "label": label})

    print(f"{path.name}: 有效样本={len(rows)}, 过滤空文本/空 is_fraud={skipped_empty}")
    return rows


def validate_label_conflicts(rows: list[dict], name: str) -> None:
    labels_by_text = defaultdict(set)
    for row in rows:
        labels_by_text[row["text"]].add(row["label"])

    conflict_count = sum(1 for labels in labels_by_text.values() if len(labels) > 1)
    if conflict_count:
        raise ValueError(f"{name} 中存在同一 text 对应不同 label，冲突文本数={conflict_count}，请先人工清洗。")


def group_by_text(rows: list[dict]) -> list[list[dict]]:
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["text"]].append(row)
    return list(grouped.values())


def split_label_groups(groups: list[list[dict]], valid_ratio: float, rng: random.Random):
    rng.shuffle(groups)
    target_valid = round(sum(len(group) for group in groups) * valid_ratio)
    valid_rows, train_rows = [], []
    valid_count = 0

    # Large duplicate groups are placed first, then greedily assigned to validation.
    # This keeps identical texts in one split while preserving the target label ratio.
    for group in sorted(groups, key=len, reverse=True):
        if valid_count < target_valid:
            valid_rows.extend(group)
            valid_count += len(group)
        else:
            train_rows.extend(group)

    return train_rows, valid_rows


def stratified_train_valid_split(rows: list[dict], valid_ratio: float):
    rows_by_label = defaultdict(list)
    for row in rows:
        rows_by_label[row["label"]].append(row)

    if set(rows_by_label) != {0, 1}:
        raise ValueError(f"训练集必须同时包含 label=0 和 label=1，当前标签={sorted(rows_by_label)}")

    rng = random.Random(SEED)
    train_rows, valid_rows = [], []
    for label in sorted(rows_by_label):
        groups = group_by_text(rows_by_label[label])
        label_train, label_valid = split_label_groups(groups, valid_ratio, rng)
        train_rows.extend(label_train)
        valid_rows.extend(label_valid)

    rng.shuffle(train_rows)
    rng.shuffle(valid_rows)
    return train_rows, valid_rows


def count_text_overlap(left_rows: list[dict], right_rows: list[dict]) -> int:
    return len({row["text"] for row in left_rows} & {row["text"] for row in right_rows})


def report_text_overlaps(train_rows: list[dict], valid_rows: list[dict], test_rows: list[dict]) -> None:
    overlaps = {
        "train-valid": count_text_overlap(train_rows, valid_rows),
        "train-test": count_text_overlap(train_rows, test_rows),
        "valid-test": count_text_overlap(valid_rows, test_rows),
    }
    total_overlap = sum(overlaps.values())
    print("重复 text 检查:")
    for name, count in overlaps.items():
        print(f"  {name}: {count}")

    if total_overlap:
        print("处理建议: 老师测试集不应参与训练；若 train-test 或 valid-test 存在重复，建议从训练/验证集中移除这些重复 text，保留测试集不变。")
    else:
        print("  未发现 train、valid、test 之间存在重复 text。")


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["text", "label"])
        writer.writeheader()
        writer.writerows({"text": row["text"], "label": row["label"]} for row in rows)


def print_stats(name: str, rows: list[dict]) -> None:
    counts = Counter(row["label"] for row in rows)
    total = len(rows)
    fraud = counts[1]
    normal = counts[0]
    fraud_ratio = fraud / total if total else 0.0
    normal_ratio = normal / total if total else 0.0
    print(
        f"{name}: 样本数量={total}, 欺诈样本数量={fraud}, 非欺诈样本数量={normal}, "
        f"欺诈比例={fraud_ratio:.4f}, 非欺诈比例={normal_ratio:.4f}"
    )


def main():
    parser = argparse.ArgumentParser(description="预处理老师提供的正式训练集和测试集。")
    parser.add_argument("--train_file", default="data/训练集结果.csv", help="老师提供的训练集结果 CSV。")
    parser.add_argument("--test_file", default="data/测试集结果.csv", help="老师提供的测试集结果 CSV。")
    parser.add_argument("--text_col", default="specific_dialogue_content")
    parser.add_argument("--label_col", default="is_fraud")
    parser.add_argument("--output_dir", default="data/processed")
    parser.add_argument("--valid_ratio", type=float, default=0.1, help="从训练集划分验证集的比例。")
    args = parser.parse_args()

    train_source = read_official_csv(Path(args.train_file), args.text_col, args.label_col)
    test_rows = read_official_csv(Path(args.test_file), args.text_col, args.label_col)

    validate_label_conflicts(train_source, "训练集结果.csv")
    validate_label_conflicts(test_rows, "测试集结果.csv")

    train_rows, valid_rows = stratified_train_valid_split(train_source, args.valid_ratio)
    report_text_overlaps(train_rows, valid_rows, test_rows)

    output_dir = Path(args.output_dir)
    write_csv(output_dir / "train.csv", train_rows)
    write_csv(output_dir / "valid.csv", valid_rows)
    write_csv(output_dir / "test.csv", test_rows)

    print(f"输出目录: {output_dir.resolve()}")
    print_stats("train.csv", train_rows)
    print_stats("valid.csv", valid_rows)
    print_stats("test.csv", test_rows)


if __name__ == "__main__":
    main()
