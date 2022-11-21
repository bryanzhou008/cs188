TASK_NAME="com2sense"
DATA_DIR="datasets/com2sense"

# https://arxiv.org/pdf/1810.04805.pdf
MODEL_TYPE="bert-large-cased-whole-word-masking" 
SUBDIR="wholeword"
python3 -m trainers.train \
  --model_name_or_path ${MODEL_TYPE} \
  --do_train \
  --do_eval \
  --evaluate_during_training \
  --per_gpu_train_batch_size 16 \
  --per_gpu_eval_batch_size 1 \
  --gradient_accumulation_steps 4 \
  --max_steps 2000 \
  --learning_rate 5e-5 \
  --max_seq_length 128 \
  --weight_decay 0.01 \
  --output_dir "${TASK_NAME}/${SUBDIR}" \
  --task_name "${TASK_NAME}" \
  --data_dir "${DATA_DIR}" \
  --overwrite_output_dir \
  --save_steps 200 \
  --logging_steps 20 \
  --eval_split "dev" \
  --score_average_method "binary" \
  --eval_all_checkpoints

# https://arxiv.org/pdf/2209.14557.pdf
MODEL_TYPE="mediabiasgroup/DA-RoBERTa-BABE" 
SUBDIR="mediabias"  
python3 -m trainers.train \
  --model_name_or_path ${MODEL_TYPE} \
  --do_train \
  --do_eval \
  --evaluate_during_training \
  --per_gpu_train_batch_size 16 \
  --per_gpu_eval_batch_size 1 \
  --gradient_accumulation_steps 8 \
  --learning_rate 5e-5 \
  --max_steps 2000 \
  --max_seq_length 128 \
  --output_dir "${TASK_NAME}/${SUBDIR}" \
  --task_name "${TASK_NAME}" \
  --data_dir "${DATA_DIR}" \
  --overwrite_output_dir \
  --save_steps 200 \
  --logging_steps 20 \
  --warmup_steps 100 \
  --eval_split "dev" \
  --score_average_method "binary" \
  --eval_all_checkpoints

# https://huggingface.co/deepset/deberta-v3-large-squad2
MODEL_TYPE="deepset/deberta-v3-large-squad2" 
SUBDIR="debertasquad"  
python3 -m trainers.train \
  --model_name_or_path ${MODEL_TYPE} \
  --do_train \
  --do_eval \
  --evaluate_during_training \
  --per_gpu_train_batch_size 2 \
  --per_gpu_eval_batch_size 1 \
  --gradient_accumulation_steps 32 \
  --learning_rate 7e-6 \
  --max_steps 2000 \
  --max_seq_length 512 \
  --output_dir "${TASK_NAME}/${SUBDIR}" \
  --task_name "${TASK_NAME}" \
  --data_dir "${DATA_DIR}" \
  --overwrite_output_dir \
  --save_steps 200 \
  --logging_steps 20 \
  --warmup_steps 400 \
  --eval_split "dev" \
  --score_average_method "binary" \
  --eval_all_checkpoints

# https://arxiv.org/pdf/1907.11692.pdf
MODEL_TYPE="roberta-base" 
SUBDIR="robertabase"  
python3 -m trainers.train \
  --model_name_or_path ${MODEL_TYPE} \
  --do_train \
  --do_eval \
  --evaluate_during_training \
  --per_gpu_train_batch_size 16 \
  --per_gpu_eval_batch_size 1 \
  --gradient_accumulation_steps 4 \
  --learning_rate 1e-5 \
  --max_steps 2000 \
  --max_seq_length 128 \
  --weight_decay 0.01 \
  --output_dir "${TASK_NAME}/${SUBDIR}" \
  --task_name "${TASK_NAME}" \
  --data_dir "${DATA_DIR}" \
  --overwrite_output_dir \
  --save_steps 200 \
  --logging_steps 20 \
  --warmup_steps 120 \
  --eval_split "dev" \
  --score_average_method "binary" \
  --eval_all_checkpoints

# https://huggingface.co/deepset/roberta-base-squad2
MODEL_TYPE="deepset/roberta-base-squad2" 
SUBDIR="robertasquad"  
python3 -m trainers.train \
  --model_name_or_path ${MODEL_TYPE} \
  --do_train \
  --do_eval \
  --evaluate_during_training \
  --per_gpu_train_batch_size 16 \
  --per_gpu_eval_batch_size 1 \
  --gradient_accumulation_steps 4 \
  --learning_rate 1e-5 \
  --max_steps 2000 \
  --max_seq_length 128 \
  --weight_decay 0.01 \
  --output_dir "${TASK_NAME}/${SUBDIR}" \
  --task_name "${TASK_NAME}" \
  --data_dir "${DATA_DIR}" \
  --overwrite_output_dir \
  --save_steps 200 \
  --logging_steps 20 \
  --warmup_steps 120 \
  --eval_split "dev" \
  --score_average_method "binary" \
  --eval_all_checkpoints

# https://huggingface.co/deepset/tinyroberta-squad2
MODEL_TYPE="deepset/tinyroberta-squad2" 
SUBDIR="tinyroberta"  
python3 -m trainers.train \
  --model_name_or_path ${MODEL_TYPE} \
  --do_train \
  --do_eval \
  --evaluate_during_training \
  --per_gpu_train_batch_size 16 \
  --per_gpu_eval_batch_size 1 \
  --gradient_accumulation_steps 4 \
  --learning_rate 1e-5 \
  --max_steps 2000 \
  --max_seq_length 128 \
  --weight_decay 0.01 \
  --output_dir "${TASK_NAME}/${SUBDIR}" \
  --task_name "${TASK_NAME}" \
  --data_dir "${DATA_DIR}" \
  --overwrite_output_dir \
  --save_steps 200 \
  --logging_steps 20 \
  --warmup_steps 120 \
  --eval_split "dev" \
  --score_average_method "binary" \
  --eval_all_checkpoints

MODEL_TYPE="outputs/semeval/debertav3large/checkpoint-80" 
SUBDIR="semevalckpt80"  
python3 -m trainers.train \
  --model_name_or_path ${MODEL_TYPE} \
  --do_train \
  --do_eval \
  --evaluate_during_training \
  --per_gpu_train_batch_size 8 \
  --per_gpu_eval_batch_size 1 \
  --gradient_accumulation_steps 4 \
  --learning_rate 9e-6 \
  --adam_epsilon 1e-6 \
  --max_steps 1600 \
  --max_seq_length 128 \
  --weight_decay 0.01 \
  --output_dir "${TASK_NAME}/${SUBDIR}" \
  --task_name "${TASK_NAME}" \
  --data_dir "${DATA_DIR}" \
  --overwrite_output_dir \
  --save_steps 100 \
  --logging_steps 20 \
  --warmup_steps 500 \
  --eval_split "dev" \
  --score_average_method "binary" \
  --eval_all_checkpoints

TASK_NAME="semeval"
DATA_DIR="datasets/semeval_2020_task4"
MODEL_TYPE="microsoft/deberta-v3-large" 
SUBDIR="debertav3large"
python3 -m trainers.train \
  --model_name_or_path ${MODEL_TYPE} \
  --do_train \
  --do_eval \
  --evaluate_during_training \
  --per_gpu_train_batch_size 4 \
  --per_gpu_eval_batch_size 1 \
  --gradient_accumulation_steps 8 \
  --max_steps 200 \
  --learning_rate 1e-5 \
  --max_seq_length 128 \
  --weight_decay 0.01 \
  --output_dir "${TASK_NAME}/${SUBDIR}" \
  --task_name "${TASK_NAME}" \
  --data_dir "${DATA_DIR}" \
  --overwrite_output_dir \
  --save_steps 20 \
  --logging_steps 5 \
  --eval_split "dev" \
  --score_average_method "binary" \
  --eval_all_checkpoints

MODEL_TYPE="outputs/com2sense/semevalckpt80/checkpoint-800" 
SUBDIR="semevalckpt80-resume"
python3 -m trainers.train \
  --model_name_or_path ${MODEL_TYPE} \
  --do_train \
  --do_eval \
  --evaluate_during_training \
  --per_gpu_train_batch_size 8 \
  --per_gpu_eval_batch_size 1 \
  --gradient_accumulation_steps 4 \
  --learning_rate 9e-6 \
  --adam_epsilon 1e-6 \
  --max_steps 800 \
  --max_seq_length 128 \
  --weight_decay 0.01 \
  --output_dir "${TASK_NAME}/${SUBDIR}" \
  --task_name "${TASK_NAME}" \
  --data_dir "${DATA_DIR}" \
  --overwrite_output_dir \
  --save_steps 100 \
  --logging_steps 20 \
  --warmup_steps 100 \
  --eval_split "dev" \
  --score_average_method "binary" \
  --eval_all_checkpoints