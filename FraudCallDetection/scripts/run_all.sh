#!/usr/bin/env bash
set -euo pipefail

# 修改为课堂提供的有标签训练数据路径和列名。
TRAIN_DATA_FILE="data/classroom_dataset.xlsx"
TRAIN_TEXT_COL="通话文本列名"
TRAIN_LABEL_COL="标签列名"
TRAIN_SHEET_NAME="0"

# 当前项目中已存在的 TextFooler 攻击文件。
TEXTFOOLER_FILE="data/data.xlsx"
TEXTFOOLER_SHEET_NAME="textFooler攻击结果"
TEXTFOOLER_ORIGINAL_COL="原始的通话记录"
TEXTFOOLER_ATTACK_COL="textfooler攻击后的通话记录"

MODEL_DIR="saved_model"
RESULTS_DIR="results"

python src/inspect_data.py \
  --file_path "$TRAIN_DATA_FILE" \
  --sheet_name "$TRAIN_SHEET_NAME" \
  --label_col "$TRAIN_LABEL_COL"

python src/preprocess.py \
  --file_path "$TRAIN_DATA_FILE" \
  --sheet_name "$TRAIN_SHEET_NAME" \
  --text_col "$TRAIN_TEXT_COL" \
  --label_col "$TRAIN_LABEL_COL" \
  --output_dir data

python src/train_bert.py \
  --train_file data/train.csv \
  --valid_file data/valid.csv \
  --output_dir "$MODEL_DIR"

python src/generate_fraudr1_attack.py \
  --input_file data/test.csv \
  --output_dir data

python src/evaluate.py --model_dir "$MODEL_DIR" --test_file data/test.csv --dataset_name original_test --results_dir "$RESULTS_DIR"
python src/evaluate.py --model_dir "$MODEL_DIR" --test_file data/test_round1_trust.csv --dataset_name fraudr1_round1_trust --results_dir "$RESULTS_DIR"
python src/evaluate.py --model_dir "$MODEL_DIR" --test_file data/test_round2_trust_urgency.csv --dataset_name fraudr1_round2_trust_urgency --results_dir "$RESULTS_DIR"
python src/evaluate.py --model_dir "$MODEL_DIR" --test_file data/test_round3_trust_urgency_emotion.csv --dataset_name fraudr1_round3_trust_urgency_emotion --results_dir "$RESULTS_DIR"

python src/attack_test.py \
  --model_dir "$MODEL_DIR" \
  --attack_file "$TEXTFOOLER_FILE" \
  --sheet_name "$TEXTFOOLER_SHEET_NAME" \
  --original_col "$TEXTFOOLER_ORIGINAL_COL" \
  --attack_col "$TEXTFOOLER_ATTACK_COL" \
  --results_dir "$RESULTS_DIR"

echo "全部实验完成，结果见 $RESULTS_DIR/summary.csv"
