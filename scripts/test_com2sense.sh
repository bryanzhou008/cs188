TASK_NAME="com2sense"
DATA_DIR="datasets/com2sense"
SUBDIR="semevalckpt" 

python3 -m trainers.train \
  --model_name_or_path "outputs/${TASK_NAME}/${SUBDIR}/checkpoint-best" \
  --do_eval \
  --iters_to_eval "best" \
  --per_gpu_eval_batch_size 1 \
  --max_seq_length 128 \
  --output_dir "${TASK_NAME}/${SUBDIR}" \
  --task_name "${TASK_NAME}" \
  --data_dir "${DATA_DIR}" \
  --eval_split "test" \
  --do_not_load_optimizer