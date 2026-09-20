# FT-KD

不稳定测试（Flaky Test）二分类检测实验代码。方法是白盒 logits 知识蒸馏：教师模型微调后，将 logits 蒸馏到轻量学生模型，并与 CodeBERT 上的量化基线对比。

对应论文：《基于知识蒸馏策略的不稳定测试检测方法研究》。

## 模型与目录

| 角色 | 目录 | 说明 |
|------|------|------|
| 教师 | `white-box/CodeBERT` | 微调 + 蒸馏 + `Q*` 量化基线 |
| 教师 | `white-box/GraphCodeBERT` | 微调 + 蒸馏（含 data flow） |
| 教师 | `white-box/PLBART` | 微调 + 蒸馏 |
| 教师 | `white-box/CodeT5` | 微调 + 蒸馏（蒸馏时默认冻结编码器） |
| 教师 | `white-box/UniXCoder` | 微调 + 蒸馏 |
| 教师 | `white-box/CodeT5small` | 仅微调 |
| 学生基线 | `white-box/DistilroBERT` | DistilRoBERTa 直接训练 |
| 学生基线 | `white-box/TinyBERT` | TinyBERT 直接训练 |

数据集放在 `dataset/IDoFT`、`dataset/FlakeFlagger`，各需 `flaky_db.csv`、`flaky_train.csv`、`flaky_test.csv`。

## 脚本约定

- `train.py` → 调用 `run.py`：教师或学生直接微调
- `train_student_dis.py` → `run2_dis.py`：蒸馏到 DistilRoBERTa
- `train_student_tiny.py` → `run2_tiny.py`：蒸馏到 TinyBERT
- `test.py` / `test_student_*.py`：评测
- CodeBERT 的 `Qtrain.py` / `Qtest.py` / `Qrun.py`：量化基线

入口脚本里改 `dataset = 'IDoFT'` 或 `'FlakeFlagger'` 即可切换数据；`--requires_grad 0` 冻结编码器，`1` 端到端微调。

## 快速开始

```bash
cd white-box/CodeBERT/code
python train.py
python train_student_dis.py
python train_student_tiny.py
python test.py
```

量化基线：

```bash
cd white-box/CodeBERT/code
python Qtrain.py
python Qtest.py
```

## 实验超参

蒸馏损失（各 `model.py` 中 `DistillationLoss`）：

```text
L = α · CE(student, label) + (1-α) · T² · KL(log_softmax(s/T), softmax(t/T))
```

### 共享训练默认

| 参数 | 值 |
|------|-----|
| learning_rate | `2e-5` |
| epochs | `10` |
| train / eval batch size | `4` / `4` |
| weight_decay | `0.0` |
| adam_epsilon | `1e-8` |
| max_grad_norm | `1.0` |
| gradient_accumulation_steps | `1` |
| warmup | 运行时设为 `max_steps // 5`（`max_steps = epochs × |train|`） |
| 蒸馏 α / T | `0.7` / `3.0` |

`train_*.py` 只设置数据集、模型路径与 `requires_grad`。

### 序列长度与其它差异

| 设置 | code_length | data_flow_length | 备注 |
|------|-------------|------------------|------|
| CodeBERT / CodeT5 / PLBART / 学生基线 | 512 | — | seed 多为 `123456` |
| GraphCodeBERT 微调 | 384 | 128 | |
| GraphCodeBERT 蒸馏 | 448 | 64 | |
| UniXCoder 微调 | 256 | 64 | seed=`42` |
| UniXCoder 蒸馏 | 448 | 64 | `run2_dis` seed=`114514` |

CodeT5 蒸馏入口默认 `--requires_grad 0`；其余多数为 `1`。

## 子目录说明

- [CodeBERT](white-box/CodeBERT/README.md)
- [GraphCodeBERT](white-box/GraphCodeBERT/README.md)
- [PLBART](white-box/PLBART/README.md)
- [CodeT5](white-box/CodeT5/README.md)
- [UniXCoder](white-box/UniXCoder/README.md)
- [DistilroBERT](white-box/DistilroBERT/README.md)
- [TinyBERT](white-box/TinyBERT/README.md)
