import os

os.system("CUDA_VISIBLE_DEVICES=2 python run.py \
        --output_dir=saved_models/Flaky \
        --config_name=../../codet5small-base \
        --model_name_or_path=../../codet5small-base \
        --tokenizer_name=../../codet5small-base \
        --requires_grad 1 \
        --do_test \
        --code_db_file=../../dataset_projects/FlakeFlagger/flaky_db.csv \
        --train_data_file=../../dataset_projects/FlakeFlagger/flaky_train.csv \
        --eval_data_file=../../dataset_projects/FlakeFlagger/flaky_test.csv \
        --test_data_file=../../dataset_projects/FlakeFlagger/flaky_test.csv 2>&1")