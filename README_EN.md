# FT-KD

Code for binary flaky-test detection with white-box logits distillation (teacher to student), plus a CodeBERT quantization baseline.

Paper: *Research on Flaky Test Detection Methods Based on Knowledge Distillation Strategies*.

## Models and layout

| Role | Directory | Notes |
|------|-----------|-------|
| Teacher | `white-box/CodeBERT` | FT + KD + `Q*` quantization baseline |
| Teacher | `white-box/GraphCodeBERT` | FT + KD (data-flow inputs) |
| Teacher | `white-box/PLBART` | FT + KD |
| Teacher | `white-box/CodeT5` | FT + KD (encoder frozen for distillation by default) |
| Teacher | `white-box/UniXCoder` | FT + KD |
| Teacher | `white-box/CodeT5small` | Fine-tuning only |
| Student baseline | `white-box/DistilroBERT` | Direct DistilRoBERTa training |
| Student baseline | `white-box/TinyBERT` | Direct TinyBERT training |

Datasets live under `dataset/IDoFT` and `dataset/FlakeFlagger`, each with `flaky_db.csv`, `flaky_train.csv`, `flaky_test.csv`.

## Script convention

- `train.py` → `run.py`: direct fine-tuning
- `train_student_dis.py` → `run2_dis.py`: distill to DistilRoBERTa
- `train_student_tiny.py` → `run2_tiny.py`: distill to TinyBERT
- `test.py` / `test_student_*.py`: evaluation
- CodeBERT `Qtrain.py` / `Qtest.py` / `Qrun.py`: quantization baseline

Change `dataset = 'IDoFT'` or `'FlakeFlagger'` in the entry scripts. `--requires_grad 0` freezes the encoder; `1` trains end-to-end.

## Quick start

```bash
cd white-box/CodeBERT/code
python train.py
python train_student_dis.py
python train_student_tiny.py
python test.py
```

Quantization baseline:

```bash
cd white-box/CodeBERT/code
python Qtrain.py
python Qtest.py
```

## Hyperparameters

Distillation loss (`DistillationLoss` in each `model.py`):

```text
L = α · CE(student, label) + (1-α) · T² · KL(log_softmax(s/T), softmax(t/T))
```

### Shared training defaults

| Parameter | Value |
|-----------|-------|
| learning_rate | `2e-5` |
| epochs | `10` |
| train / eval batch size | `4` / `4` |
| weight_decay | `0.0` |
| adam_epsilon | `1e-8` |
| max_grad_norm | `1.0` |
| gradient_accumulation_steps | `1` |
| warmup | set at runtime to `max_steps // 5` |
| distill α / T | `0.7` / `3.0` |

`train_*.py` only sets dataset, model paths, and `requires_grad`.

### Sequence length and other differences

| Setting | code_length | data_flow_length | Notes |
|---------|-------------|------------------|-------|
| CodeBERT / CodeT5 / PLBART / student baselines | 512 | — | seed usually `123456` |
| GraphCodeBERT fine-tune | 384 | 128 | |
| GraphCodeBERT distill | 448 | 64 | |
| UniXCoder fine-tune | 256 | 64 | seed=`42` |
| UniXCoder distill | 448 | 64 | `run2_dis` seed=`114514` |

CodeT5 distillation entries default to `--requires_grad 0`; most others use `1`.

## Per-model READMEs

- [CodeBERT](white-box/CodeBERT/README_EN.md)
- [GraphCodeBERT](white-box/GraphCodeBERT/README_EN.md)
- [PLBART](white-box/PLBART/README_EN.md)
- [CodeT5](white-box/CodeT5/README_EN.md)
- [UniXCoder](white-box/UniXCoder/README_EN.md)
- [DistilroBERT](white-box/DistilroBERT/README_EN.md)
- [TinyBERT](white-box/TinyBERT/README_EN.md)
