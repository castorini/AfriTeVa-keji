#!/bin/bash
{
    set -e;
    set -x;

    export TASKS_TO_LOAD="aya"

    CHECKPOINT="models/T5_1_1_large/checkpoint_524288"
    EXPERIMENT_NAME=afriteva_v2_large_aya_30k_all_african_languages

    # Training configurations
    task="aya"
    FT_NUM_STEPS=30000
    TRAIN_BATCH_SIZE=256
    NUM_MICROBATCHES=16
    EVAL_BATCH_SIZE=32
    INFER_BATCH_SIZE=32
    LEARNING_RATE=0.0003
    LEARNING_RATE_SCHEDULE="constant"
    WARMUP_STEPS=3000
    CHECKPOINT_PERIOD=5000
    EVAL_PERIOD=5000
    FEATURE_LENGTHS="{'inputs': 1024, 'targets': 1024}"

    ADDITIONAL_GIN_CONFIGS=("--gin.LOSS_NORMALIZING_FACTOR=\"AVERAGE_PER_SEQUENCE\"")
    REMOVE_CHECKPOINTS=false

    # ----------------------
    if [ -z "$TASKS" ]; then
        TASKS=()

        if [ -z "$LANGUAGES" ] && [ -z "$task" ]; then
            echo "Error: All of \`TASKS\`, \`LANGUAGES\` and \`task\` are unset."
            exit 1
        fi
    fi

    if [ -n "$LANGUAGES" ]; then
        for language in $LANGUAGES; do
            TASKS+=("${language}_aya")
        done
    fi

    if [ -n "$task" ]; then
        TASKS+=("$task")
    fi

    # ------------------------------------
    PRETRAINED_STEPS=${CHECKPOINT##*_}
    MODEL_SIZE=${CHECKPOINT%%/checkpoint*}
    MODEL_SIZE=${MODEL_SIZE##*_}
    OUTPUT_DIR="runs/$EXPERIMENT_NAME"
    mkdir -p "$OUTPUT_DIR"
    # ------------------------------------

    for task in "${TASKS[@]}"; do
        train_steps=$((PRETRAINED_STEPS + FT_NUM_STEPS))

        scripts/t5_utils.sh \
        --action finetune \
        --task $task \
        --feature_lengths "$FEATURE_LENGTHS" \
        --batch_size $TRAIN_BATCH_SIZE \
        --num_microbatches $NUM_MICROBATCHES \
        --warmup_steps $WARMUP_STEPS \
        --learning_rate $LEARNING_RATE \
        --learning_rate_schedule $LEARNING_RATE_SCHEDULE \
        --checkpoint $CHECKPOINT \
        --checkpoint_period $CHECKPOINT_PERIOD \
        --eval_period $EVAL_PERIOD \
        --train_steps $train_steps \
        --model_size $MODEL_SIZE \
        --output_dir $OUTPUT_DIR \
        --gin.infer_eval/utils.DatasetConfig.batch_size=$INFER_BATCH_SIZE \
        --gin.train_eval/utils.DatasetConfig.batch_size=$EVAL_BATCH_SIZE \
        "${ADDITIONAL_GIN_CONFIGS[@]}" \
        >& "$OUTPUT_DIR/${task}_$(date +"%m-%d_%H-%M-%S").log" \
        && finetuned=true

        checkpoints=($(ls $OUTPUT_DIR | grep checkpoint))

        if [[ $REMOVE_CHECKPOINTS == "true" ]]; then
            for ckpt in "${checkpoints[@]}";
            do
                rm -rf "$OUTPUT_DIR/$ckpt"
            done
        fi
    done
}