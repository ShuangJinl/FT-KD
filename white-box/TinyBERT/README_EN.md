# TinyBERT (student baseline)

Direct TinyBERT training baseline. For teacher distillation, run `train_student_tiny.py` under each teacher directory.

## Data

`flaky_db.csv`, `flaky_train.csv`, `flaky_test.csv` under `dataset/IDoFT` or `dataset/FlakeFlagger`.

## Scripts

- `train.py` → `run.py`: direct fine-tuning
- `test.py`: evaluation

Defaults: `--requires_grad 1`, `code_length=512`. See root README for lr / epochs / batch.

## Run

```bash
cd code
python train.py
python test.py
```
