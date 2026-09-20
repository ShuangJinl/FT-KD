import os
dataset = 'FlakeFlagger'
os.system(f"CUDA_VISIBLE_DEVICES=1 python run2.py \
        --output_dir=saved_models/{dataset} \
        --teacher_config_name=../../plbart \
        --student_config_name=../../tinybert-base \
        --teacher_model_name_or_path=../../plbart \
        --student_model_name_or_path=../../tinybert-base \
        --teacher_tokenizer_name=../../plbart \
        --student_tokenizer_name=../../tinybert-base \
        --requires_grad 1 \
        --do_test \
        --code_db_file=../../../dataset/{dataset}/flaky_db.csv \
        --train_data_file=../../../dataset/{dataset}/flaky_train.csv \
        --eval_data_file=../../../dataset/{dataset}/flaky_test.csv \
        --test_data_file=../../../dataset/{dataset}/flaky_test.csv 2>&1")