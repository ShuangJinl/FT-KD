import os
dataset = 'IDoFT'
os.system(f"CUDA_VISIBLE_DEVICES=1 python run2_dis.py \
        --output_dir=saved_models/{dataset} \
        --teacher_config_name=../../codet5-base \
        --student_config_name=../../distilroberta-base \
        --teacher_model_name_or_path=../../codet5-base \
        --student_model_name_or_path=../../distilroberta-base \
        --teacher_tokenizer_name=../../codet5-base \
        --student_tokenizer_name=../../distilroberta-base \
        --requires_grad 0 \
        --do_test \
        --code_db_file=../../../dataset/{dataset}/flaky_db.csv \
        --train_data_file=../../../dataset/{dataset}/flaky_train.csv \
        --eval_data_file=../../../dataset/{dataset}/flaky_test.csv \
        --test_data_file=../../../dataset/{dataset}/flaky_test.csv 2>&1")