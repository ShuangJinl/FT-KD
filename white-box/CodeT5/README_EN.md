# CodeT5

Teacher fine-tuning and white-box logits distillation. Distillation entry scripts default to `--requires_grad 0` (freeze encoder), unlike most other teachers.

## Data

`flaky_db.csv`, `flaky_train.csv`, `flaky_test.csv` under `dataset/IDoFT` or `dataset/FlakeFlagger`.

## Scripts

| Entry | Role |
|-------|------|
| `train.py` | Fine-tune teacher |
| `train_student_dis.py` | Distill to DistilRoBERTa |
| `train_student_tiny.py` | Distill to TinyBERT |
| `test.py` / `test_student_*.py` | Evaluation |

Distillation: `α=0.7`, `T=3.0`.

## Run

```bash
cd code
python train.py
python train_student_dis.py
python train_student_tiny.py
python test.py
```

Change `dataset` in entry scripts. `CodeT5small` is fine-tuning only.
