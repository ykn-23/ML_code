import argparse
import pandas as pd

from utils import read_table


def list_excel_sheets(file_path: str) -> None:
    if not file_path.lower().endswith((".xlsx", ".xls")):
        return
    try:
        excel = pd.ExcelFile(file_path)
    except Exception as exc:
        print(f"无法读取 sheet 列表: {exc}")
        return
    print("\n========== Sheet ==========")
    for i, sheet in enumerate(excel.sheet_names):
        print(f"{i}: {sheet}")


def main():
    parser = argparse.ArgumentParser(description="检查数据文件的 sheet、列名、样例和标签分布。")
    parser.add_argument("--file_path", required=True, help="输入数据文件，支持 csv/xlsx/xls。")
    parser.add_argument("--sheet_name", default=0, help="Excel sheet 名或序号，默认第一个 sheet。")
    parser.add_argument("--label_col", default=None, help="可选：标签列名。")
    parser.add_argument("--text_col", default=None, help="可选：原始文本列名。")
    parser.add_argument("--attack_text_col", default=None, help="可选：攻击后文本列名。")
    args = parser.parse_args()

    sheet_name = int(args.sheet_name) if str(args.sheet_name).isdigit() else args.sheet_name
    list_excel_sheets(args.file_path)
    df = read_table(args.file_path, sheet_name=sheet_name)

    print("\n========== 列名 ==========")
    for i, col in enumerate(df.columns):
        print(f"{i}: {col}")

    print("\n========== 前 5 行 ==========")
    print(df.head().to_string(index=False))

    if args.label_col:
        if args.label_col in df.columns:
            print("\n========== 标签分布 ==========")
            print(df[args.label_col].value_counts(dropna=False))
        else:
            print(f"\n找不到标签列: {args.label_col}")

    for col, title in [(args.text_col, "原始文本样例"), (args.attack_text_col, "攻击后文本样例")]:
        if not col:
            continue
        if col not in df.columns:
            print(f"\n找不到列: {col}")
            continue
        print(f"\n========== {title} ==========")
        print(df[col].dropna().astype(str).head(3).to_string(index=False))


if __name__ == "__main__":
    main()
