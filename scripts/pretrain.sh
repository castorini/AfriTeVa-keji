#!/bin/bash
set -x;

MODEL_DIR="models/T5_1_1_base"
MODEL_SIZE="base"
BATCH_SIZE=512
TASK="news_and_wiki"
FEATURE_LENGTHS="{'inputs': 512, 'targets': 114}"
TRAIN_STEPS=524288
CHECKPOINT_PERIOD=50000
EVAL_PERIOD=25000
NUM_MICROBATCHES=4
LOGFILE="logs/t5_1_1_base_pretrain_${BATCH_SIZE}_batch_size_${TASK}_.log"

bash scripts/t5_utils.sh \
    --action pretrain \
    --batch_size $BATCH_SIZE \
    --checkpoint_period $CHECKPOINT_PERIOD \
    --feature_lengths "$FEATURE_LENGTHS" \
    --model_size $MODEL_SIZE \
    --task $TASK \
    --train_steps $TRAIN_STEPS \
    --eval_period $EVAL_PERIOD \
    --output_dir $MODEL_DIR \
    --gin.trainer.Trainer.num_microbatches=$NUM_MICROBATCHES >& $LOGFILE &
