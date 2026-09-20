# UniXCoder

Teacher fine-tuning and white-box logits distillation. Fine-tune defaults: `code_length=256, data_flow_length=64, seed=42`. Distillation uses `code_length=448`, `α=0.7`, `T=3.0`.

## Data

`flaky_db.csv`, `flaky_train.csv`, `flaky_test.csv` under `dataset/IDoFT` or `dataset/FlakeFlagger`.

## Scripts

| Entry | Role |
|-------|------|
| `train.py` | Fine-tune teacher |
| `train_student_dis.py` | Distill to DistilRoBERTa |
| `train_student_tiny.py` | Distill to TinyBERT |
| `test.py` / `test_student_*.py` | Evaluation |

## Run

```bash
cd code
python train.py
python train_student_dis.py
python train_student_tiny.py
python test.py
```

Change `dataset` in entry scripts; `--requires_grad 0/1` controls encoder freezing.
