import argparse

import pandas as pd

from evaluate import evaluate_dataframe, load_model, print_metrics
from utils import check_columns, ensure_dir, read_table, set_seed


def build_fraud_only_df(source_df: pd.DataFrame, text_col: str) -> pd.DataFrame:
    check_columns(source_df, [text_col])
    out = pd.DataFrame()
    out["id"] = range(len(source_df))
    out["text"] = source_df[text_col].astype(str).str.strip()
    out["label"] = 1
    out = out.replace({"": pd.NA, "nan": pd.NA, "None": pd.NA}).dropna(subset=["text"])
    return out.reset_index(drop=True)


def main():
    parser = argparse.ArgumentParser(description="评估 TextFooler 攻击前后欺诈通话的模型表现差异。")
    parser.add_argument("--model_dir", default="saved_model")
    parser.add_argument("--attack_file", default="data/data.xlsx", help="TextFooler 攻击 Excel/CSV 文件。")
    parser.add_argument("--sheet_name", default=0)
    parser.add_argument("--original_col", default="原始的通话记录")
    parser.add_argument("--attack_col", default="textfooler攻击后的通话记录")
    parser.add_argument("--results_dir", default="results")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--max_length", type=int, default=256)
    args = parser.parse_args()

    set_seed()
    results_dir = ensure_dir(args.results_dir)
    sheet_name = int(args.sheet_name) if str(args.sheet_name).isdigit() else args.sheet_name
    source_df = read_table(args.attack_file, sheet_name=sheet_name)
    check_columns(source_df, [args.original_col, args.attack_col])

    model, tokenizer, device = load_model(args.model_dir)
    datasets = [
        ("textfooler_original", build_fraud_only_df(source_df, args.original_col)),
        ("textfooler_attack", build_fraud_only_df(source_df, args.attack_col)),
    ]

    for dataset_name, df in datasets:
        metrics = evaluate_dataframe(
            df=df,
            model=model,
            tokenizer=tokenizer,
            device=device,
            dataset_name=dataset_name,
            text_col="text",
            label_col="label",
            results_dir=results_dir,
            batch_size=args.batch_size,
            max_length=args.max_length,
        )
        print_metrics(dataset_name, metrics, results_dir)

    print("TextFooler 对比完成。重点查看 results/summary.csv 中的 fraud_detection_rate 与 attack_success_rate。")


if __name__ == "__main__":
    main()
