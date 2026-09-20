import os
dataset = 'FlakeFlagger'
os.system(f"CUDA_VISIBLE_DEVICES=1 python run2_dis.py \
        --output_dir=saved_models/{dataset} \
        --teacher_config_name=../../graphcodebert-base \
        --student_config_name=../../distilroberta-base \
        --teacher_model_name_or_path=../../graphcodebert-base \
        --student_model_name_or_path=../../distilroberta-base \
        --teacher_tokenizer_name=../../graphcodebert-base \
        --student_tokenizer_name=../../distilroberta-base \
        --requires_grad 1 \
        --do_eval \
        --code_db_file=../../../dataset/{dataset}/flaky_db.csv \
        --train_data_file=../../../dataset/{dataset}/flaky_train.csv \
        --eval_data_file=../../../dataset/{dataset}/flaky_test.csv \
        --test_data_file=../../../dataset/{dataset}/flaky_test.csv 2>&1")