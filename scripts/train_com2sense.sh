TASK_NAME="com2sense"
DATA_DIR="datasets/com2sense"
MODEL_TYPE="outputs/semeval/debertav3large/checkpoint-100" 
SUBDIR="semevalckpt" 
python3 -m trainers.train \
  --model_name_or_path ${MODEL_TYPE} \
  --do_train \
  --do_eval \
  --evaluate_during_training \
  --per_gpu_train_batch_size 6 \
  --per_gpu_eval_batch_size 1 \
  --gradient_accumulation_steps 8 \
  --max_steps 2000 \
  --max_seq_length 128 \
  --learning_rate 9e-6 \
  --adam_epsilon 1e-6 \
  --weight_decay 0.01 \
  --output_dir "${TASK_NAME}/${SUBDIR}" \
  --task_name "${TASK_NAME}" \
  --data_dir "${DATA_DIR}" \
  --overwrite_output_dir \
  --save_steps 400 \
  --logging_steps 20 \
  --warmup_steps 500 \
  --eval_split "dev" \
  --score_average_method "binary" \
  --eval_all_checkpoints