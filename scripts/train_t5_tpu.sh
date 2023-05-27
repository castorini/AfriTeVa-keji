MODEL_DIR="gs://afriqa-bucket/T5Models"

python -m t5x.main \
--run_mode="train" \
--gin.MODEL_DIR=${MODEL_DIR} \
--gin_file="config/runs/tpu/pretrain.gin" \
---alsologtostderr
