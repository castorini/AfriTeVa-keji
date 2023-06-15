export CUDA_VISIBLE_DEVICES="0,1,2,3"
export NCCL_P2P_DISABLE=1 # May be necessary on CUDA 12.

bash scripts/t5_utils.sh \
finetune \
base \
xlsum \
runs/T5_1_1_base_xlsum \
--gin.TASK_FEATURE_LENGTHS="{'inputs': 512, 'targets': 64}" \
--gin.TRAIN_STEPS=559288 \
--gin.BATCH_SIZE=128 \
--gin.EVAL_PERIOD=5000 \
--gin.train.infer_eval_dataset_cfg=None \
--gin.utils.SaveCheckpointConfig.period=5000 \
>& logs/T5_1_1_base_xlsum_ft.log &