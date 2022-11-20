TASK_NAME="com2sense"
DATA_DIR="datasets/com2sense"

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
  --save_steps 200 \
  --logging_steps 20 \
  --warmup_steps 100 \
  --score_average_method "binary" \
  --eval_all_checkpoints
  --eval_split "dev" \
  --overwrite_output_dir \
