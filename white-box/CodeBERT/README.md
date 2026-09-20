# CodeBERT

教师模型：直接微调、白盒 logits 蒸馏（学生为 DistilRoBERTa / TinyBERT），以及 `Q*` 量化基线。

## 数据

使用仓库根目录 `dataset/IDoFT` 或 `dataset/FlakeFlagger` 下的：

- `flaky_db.csv`
- `flaky_train.csv`
- `flaky_test.csv`

## 脚本

| 入口 | 作用 |
|------|------|
| `train.py` | 微调教师 |
| `train_student_dis.py` | 蒸馏到 DistilRoBERTa |
| `train_student_tiny.py` | 蒸馏到 TinyBERT |
| `test.py` / `test_student_*.py` | 评测 |
| `Qtrain.py` / `Qtest.py` | 量化基线 |

蒸馏超参：`α=0.7`，`T=3.0`（见 `run2_dis.py` / `run2_tiny.py`）。其它默认见根目录 README。

## 运行

```bash
cd code
python train.py
python train_student_dis.py
python train_student_tiny.py
python test.py
```

```bash
cd code
python Qtrain.py
python Qtest.py
```

在入口脚本里改 `dataset`；`--requires_grad 0/1` 控制是否冻结编码器。
