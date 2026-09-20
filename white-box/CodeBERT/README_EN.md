# CodeBERT

Teacher experiments: fine-tuning, white-box logits distillation (DistilRoBERTa / TinyBERT students), and `Q*` quantization baseline.

## Data

Use files under repo `dataset/IDoFT` or `dataset/FlakeFlagger`:

- `flaky_db.csv`
- `flaky_train.csv`
- `flaky_test.csv`

## Scripts

| Entry | Role |
|-------|------|
| `train.py` | Fine-tune teacher |
| `train_student_dis.py` | Distill to DistilRoBERTa |
| `train_student_tiny.py` | Distill to TinyBERT |
| `test.py` / `test_student_*.py` | Evaluation |
| `Qtrain.py` / `Qtest.py` | Quantization baseline |

Distillation: `α=0.7`, `T=3.0` (see `run2_dis.py` / `run2_tiny.py`). Other defaults: root README.

## Run

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

Switch `dataset` in the entry scripts; `--requires_grad 0/1` freezes or unfreezes the encoder.
