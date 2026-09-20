import os

os.system("CUDA_VISIBLE_DEVICES=2 python run.py \
        --output_dir=saved_models/Flaky \
        --config_name=../../t5small-base \
        --model_name_or_path=../../t5small-base \
        --tokenizer_name=../../t5small-base \
        --requires_grad 1 \
        --do_train \
        --code_db_file=../../dataset/IDoFT/flaky_db.csv \
        --train_data_file=../../dataset/IDoFT/flaky_train.csv \
        --eval_data_file=../../dataset/IDoFT/flaky_test.csv \
        --test_data_file=../../dataset/IDoFT/flaky_test.csv 2>&1")