export CUDA_VISIBLE_DEVICES="3"
export NCCL_P2P_DISABLE=1 # May be necessary on CUDA 12.

bash scripts/t5_utils.sh \
finetune \
base \
lafand_mt \
runs/T5_1_1_base_lafand \
--gin.TASK_FEATURE_LENGTHS="{'inputs': 512, 'targets': 200}" \
--gin.TRAIN_STEPS=564288 \
--gin.BATCH_SIZE=1 \
--gin.EVAL_PERIOD=3000 \
--gin.train.infer_eval_dataset_cfg=None \
--gin.utils.SaveCheckpointConfig.period=3000 \
>& logs/T5_1_1_base_lafand_ft.log &


bash scripts/t5_utils.sh \
eval \
base \
lafand_mt \
gs://awarawa/T5_1_1_base_lafand/eval_564288 \
--gin.TASK_FEATURE_LENGTHS="{'inputs': 512, 'targets': 200}" \
--gin.CHECKPOINT_PATH=\"gs://awarawa/T5_1_1_base_lafand/checkpoint_564288\" \
--gin.utils.DatasetConfig.batch_size=16 \
>& logs/T5_1_1_base_lafand_eval.log &