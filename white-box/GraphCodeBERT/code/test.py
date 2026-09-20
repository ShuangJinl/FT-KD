import os
dataset = 'IDoFT'
os.system(f"CUDA_VISIBLE_DEVICES=1 python run.py \
        --output_dir=saved_models/{dataset} \
        --config_name=../../graphcodebert-base \
        --model_name_or_path=../../graphcodebert-base \
        --tokenizer_name=../../graphcodebert-base \
        --requires_grad 1 \
        --do_test \
        --code_db_file=../../../dataset/{dataset}/flaky_db.csv \
        --train_data_file=../../../dataset/{dataset}/flaky_train.csv \
        --eval_data_file=../../../dataset/{dataset}/flaky_test.csv \
        --test_data_file=../../../dataset/{dataset}/flaky_test.csv 2>&1")