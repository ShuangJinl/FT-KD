# UniXCoder

教师模型：微调与白盒 logits 蒸馏。微调默认 `code_length=256, data_flow_length=64, seed=42`；蒸馏默认 `code_length=448`，`α=0.7`，`T=3.0`。

## 数据

`dataset/IDoFT` 或 `dataset/FlakeFlagger` 下的 `flaky_db.csv`、`flaky_train.csv`、`flaky_test.csv`。

## 脚本

| 入口 | 作用 |
|------|------|
| `train.py` | 微调教师 |
| `train_student_dis.py` | 蒸馏到 DistilRoBERTa |
| `train_student_tiny.py` | 蒸馏到 TinyBERT |
| `test.py` / `test_student_*.py` | 评测 |

## 运行

```bash
cd code
python train.py
python train_student_dis.py
python train_student_tiny.py
python test.py
```

在入口脚本改 `dataset`；`--requires_grad 0/1` 控制是否冻结编码器。
