# GraphCodeBERT

Teacher fine-tuning and white-box logits distillation. Inputs include data flow; fine-tune defaults are `code_length=384, data_flow_length=128`, distillation defaults `448 / 64`.

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

Change `dataset` in entry scripts; `--requires_grad 0/1` controls encoder freezing.
