# DistilroBERT（DistilRoBERTa 学生基线）

DistilRoBERTa 直接训练基线。教师蒸馏请到各教师目录运行 `train_student_dis.py`。

## 数据

`dataset/IDoFT` 或 `dataset/FlakeFlagger` 下的 `flaky_db.csv`、`flaky_train.csv`、`flaky_test.csv`。

## 脚本

- `train.py` → `run.py`：直接微调
- `test.py`：评测

默认 `--requires_grad 1`，`code_length=512`。lr / epochs / batch 见根目录 README。

## 运行

```bash
cd code
python train.py
python test.py
```
