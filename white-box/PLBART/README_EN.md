# PLBART

Teacher fine-tuning and white-box logits distillation. Older entry points: `train_student.py` / `run2.py`.

## Data

`flaky_db.csv`, `flaky_train.csv`, `flaky_test.csv` under `dataset/IDoFT` or `dataset/FlakeFlagger`.

## Scripts

| Entry | Role |
|-------|------|
| `train.py` | Fine-tune teacher |
| `train_student_dis.py` | Distill to DistilRoBERTa |
| `train_student_tiny.py` | Distill to TinyBERT |
| `train_student.py` | Older distill entry |
| `test.py` / `test_student_*.py` | Evaluation |

Distillation: `α=0.7`, `T=3.0` (see `run2_dis.py` / `run2_tiny.py` / `run2.py`).

## Run

```bash
cd code
python train.py
python train_student_dis.py
python train_student_tiny.py
python test.py
```

Change `dataset` in entry scripts; `--requires_grad 0/1` controls encoder freezing.
