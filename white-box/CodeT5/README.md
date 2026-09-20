# CodeT5

教师模型：微调与白盒 logits 蒸馏。蒸馏入口默认 `--requires_grad 0`（冻结编码器），与其它教师目录不同。

## 数据

`dataset/IDoFT` 或 `dataset/FlakeFlagger` 下的 `flaky_db.csv`、`flaky_train.csv`、`flaky_test.csv`。

## 脚本

| 入口 | 作用 |
|------|------|
| `train.py` | 微调教师 |
| `train_student_dis.py` | 蒸馏到 DistilRoBERTa |
| `train_student_tiny.py` | 蒸馏到 TinyBERT |
| `test.py` / `test_student_*.py` | 评测 |

蒸馏超参：`α=0.7`，`T=3.0`。

## 运行

```bash
cd code
python train.py
python train_student_dis.py
python train_student_tiny.py
python test.py
```

在入口脚本改 `dataset`。`CodeT5small` 目录仅直接微调。
