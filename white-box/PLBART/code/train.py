import os
dataset = 'FlakeFlagger'
os.system(f"CUDA_VISIBLE_DEVICES=0 python run.py \
        --output_dir=saved_models/{dataset} \
        --config_name=../../plbart \
        --model_name_or_path=../../plbart \
        --tokenizer_name=../../plbart \
        --requires_grad 3 \
        --do_train \
        --code_db_file=../../../dataset/{dataset}/flaky_db.csv \
        --train_data_file=../../../dataset/{dataset}/flaky_train.csv \
        --eval_data_file=../../../dataset/{dataset}/flaky_test.csv \
        --test_data_file=../../../dataset/{dataset}/flaky_test.csv 2>&1")