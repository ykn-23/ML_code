import argparse
from pathlib import Path

import pandas as pd

from utils import SEED, check_columns, ensure_dir, read_table, set_seed


ROUND_CONFIGS = [
    ("round1_trust", "Round 1：建立信任", "您好，这里是平台认证客服，工单号 FR-{id:06d}。本次联系已通过后台登记，请您放心核对。"),
    (
        "round2_trust_urgency",
        "Round 2：建立信任 + 制造紧迫感",
        "您好，这里是平台认证客服，工单号 FR-{id:06d}。系统显示该事项需要在今日 18:00 前完成处理，逾期可能影响账户或相关业务。",
    ),
    (
        "round3_trust_urgency_emotion",
        "Round 3：建立信任 + 制造紧迫感 + 情感操纵",
        "您好，这里是平台认证客服，工单号 FR-{id:06d}。该事项需要在今日 18:00 前处理，拖延可能造成额外损失，也会让家人担心。我们理解您可能焦虑，所以先协助您快速核对。",
    ),
]


def rewrite_text(text: str, sample_id: int, prefix_template: str) -> str:
    prefix = prefix_template.format(id=sample_id)
    return f"{prefix}\n{text}"


def generate_attack_file(df: pd.DataFrame, round_key: str, round_name: str, prefix: str, output_path: Path) -> None:
    rows = []
    for idx, row in df.iterrows():
        sample_id = int(row["id"]) if "id" in df.columns else int(idx)
        original_text = str(row["text"])
        label = int(row["label"])
        new_text = rewrite_text(original_text, sample_id, prefix) if label == 1 else original_text
        rows.append(
            {
                "id": sample_id,
                "original_text": original_text,
                "text": new_text,
                "label": label,
                "attack_type": round_key,
                "round_name": round_name,
            }
        )
    pd.DataFrame(rows).to_csv(output_path, index=False, encoding="utf-8-sig")
    print(f"已生成 {round_name}: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="生成 Fraud-R1 思想的三轮诱导改写测试集。")
    parser.add_argument("--input_file", default="data/test.csv", help="输入测试集，需包含 text 和 label。")
    parser.add_argument("--sheet_name", default=0)
    parser.add_argument("--output_dir", default="data")
    args = parser.parse_args()

    set_seed(SEED)
    sheet_name = int(args.sheet_name) if str(args.sheet_name).isdigit() else args.sheet_name
    output_dir = ensure_dir(args.output_dir)
    df = read_table(args.input_file, sheet_name=sheet_name)
    check_columns(df, ["text", "label"])

    for round_key, round_name, prefix in ROUND_CONFIGS:
        generate_attack_file(
            df=df,
            round_key=round_key,
            round_name=round_name,
            prefix=prefix,
            output_path=output_dir / f"test_{round_key}.csv",
        )


if __name__ == "__main__":
    main()
