import os
dataset = 'FlakeFlagger'
os.system(f"CUDA_VISIBLE_DEVICES=0 python Qrun.py \
        --output_dir=saved_models/{dataset} \
        --config_name=../../codebert-base \
        --model_name_or_path=../../codebert-base \
        --tokenizer_name=../../codebert-base \
        --requires_grad 1 \
        --do_test \
        --do_quantize \
        --code_db_file=../../../dataset/{dataset}/flaky_db.csv \
        --train_data_file=../../../dataset/{dataset}/flaky_train.csv \
        --eval_data_file=../../../dataset/{dataset}/flaky_test.csv \
        --test_data_file=../../../dataset/{dataset}/flaky_test.csv 2>&1")