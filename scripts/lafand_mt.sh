export CUDA_VISIBLE_DEVICES="3"
export NCCL_P2P_DISABLE=1 # May be necessary on CUDA 12.

bash scripts/t5_train.sh \
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