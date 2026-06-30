# FraudCallDetection

机器学习课程期末大作业实验项目：**基于 Fraud-R1 方法的虚假通话检测与鲁棒性分析**。

本项目使用中文 BERT/RoBERTa 训练虚假通话二分类模型，并比较模型在原始文本、TextFooler 攻击文本、Fraud-R1 式多轮诱导改写文本上的表现差异。

## 项目结构

```text
FraudCallDetection/
├── README.md
├── requirements.txt
├── data/
│   └── data.xlsx
├── src/
│   ├── inspect_data.py
│   ├── preprocess.py
│   ├── train_bert.py
│   ├── evaluate.py
│   ├── attack_test.py
│   ├── generate_fraudr1_attack.py
│   └── utils.py
├── results/
└── scripts/
    └── run_all.sh
```

当前已检查到 `data/data.xlsx` 的结构如下：

- sheet：`textFooler攻击结果`
- 原始欺诈通话列：`原始的通话记录`
- TextFooler 攻击后通话列：`textfooler攻击后的通话记录`

这个文件只有欺诈样本的原始文本和攻击后文本，不含正常样本标签，因此适合做攻击鲁棒性测试；训练二分类模型仍需要课堂提供的有标签数据集，至少包含“通话文本列”和“标签列”。

## 环境安装

建议 Python 3.10 或更高版本。

```bash
cd FraudCallDetection
pip install -r requirements.txt
```

默认模型为 `hfl/chinese-roberta-wwm-ext`。如果服务器无法下载，可在训练时改用 `bert-base-chinese`。

## 1. 检查数据列名

检查当前 TextFooler Excel：

```bash
python src/inspect_data.py --file_path data/data.xlsx --sheet_name textFooler攻击结果 --text_col 原始的通话记录 --attack_text_col textfooler攻击后的通话记录
```

检查课堂训练数据：

```bash
python src/inspect_data.py --file_path data/课堂训练数据.xlsx --label_col 标签列名
```

## 2. 预处理有标签训练数据

将课堂数据切分为 `train.csv`、`valid.csv`、`test.csv`：

```bash
python src/preprocess.py --file_path data/课堂训练数据.xlsx --text_col 通话文本列名 --label_col 标签列名 --output_dir data
```

标签会自动映射：

- 欺诈类：`欺诈`、`诈骗`、`虚假`、`fraud`、`fake`、`scam`、`1`
- 正常类：`正常`、`真实`、`非欺诈`、`normal`、`real`、`0`

如果有标签数据中也包含攻击后文本列，可额外指定：

```bash
python src/preprocess.py --file_path data/课堂训练数据.xlsx --text_col 原始文本列名 --label_col 标签列名 --attack_text_col TextFooler攻击后文本列名
```

## 3. 训练中文 BERT/RoBERTa

```bash
python src/train_bert.py --train_file data/train.csv --valid_file data/valid.csv --model_name hfl/chinese-roberta-wwm-ext --output_dir saved_model --epochs 3 --batch_size 8
```

## 4. 生成 Fraud-R1 式三轮改写

脚本只改写 `label=1` 的欺诈样本，正常样本保持不变。

```bash
python src/generate_fraudr1_attack.py --input_file data/test.csv --output_dir data
```

输出：

- `data/test_round1_trust.csv`：建立信任
- `data/test_round2_trust_urgency.csv`：建立信任 + 制造紧迫感
- `data/test_round3_trust_urgency_emotion.csv`：建立信任 + 制造紧迫感 + 情感操纵

## 5. 评估原始测试集和 Fraud-R1 改写集

```bash
python src/evaluate.py --model_dir saved_model --test_file data/test.csv --dataset_name original_test
python src/evaluate.py --model_dir saved_model --test_file data/test_round1_trust.csv --dataset_name fraudr1_round1_trust
python src/evaluate.py --model_dir saved_model --test_file data/test_round2_trust_urgency.csv --dataset_name fraudr1_round2_trust_urgency
python src/evaluate.py --model_dir saved_model --test_file data/test_round3_trust_urgency_emotion.csv --dataset_name fraudr1_round3_trust_urgency_emotion
```

## 6. 评估 TextFooler 攻击前后差异

使用当前 `data/data.xlsx`：

```bash
python src/attack_test.py --model_dir saved_model --attack_file data/data.xlsx --sheet_name textFooler攻击结果 --original_col 原始的通话记录 --attack_col textfooler攻击后的通话记录
```

该脚本会把两列都视为欺诈样本，分别输出：

- `textfooler_original`：原始欺诈通话检出结果
- `textfooler_attack`：TextFooler 攻击后欺诈通话检出结果

## 结果文件

评估后会在 `results/` 下生成：

- `summary.csv`：所有实验指标汇总
- `summary_table.md`：论文可用 Markdown 表格
- `summary_table.tex`：论文可用 LaTeX 表格
- `metrics_*.json`：单个数据集指标
- `predictions_*.csv`：逐样本预测结果

主要指标包括：

- Accuracy
- Precision
- Recall
- F1-score
- Fraud Detection Rate：欺诈检出率，即欺诈样本中预测为欺诈的比例
- Attack Success Rate：攻击成功率，即欺诈样本中被模型漏检为正常的比例

## 论文分析建议

报告中可重点比较：

1. `original_test` 的 F1 与欺诈检出率，说明基础检测能力。
2. `textfooler_original` 与 `textfooler_attack` 的欺诈检出率下降幅度，说明 TextFooler 对模型鲁棒性的影响。
3. 三个 Fraud-R1 round 的攻击成功率变化，分析“信任、紧迫感、情感操纵”逐步叠加后模型是否更容易漏检。
