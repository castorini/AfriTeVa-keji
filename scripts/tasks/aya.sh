#!/bin/bash
{
    set -e;
    set -x;

    export TASKS_TO_LOAD="aya"

    CHECKPOINT="models/T5_1_1_base/checkpoint_524288"
    OUTPUT_DIR=afriteva_v2_base_aya_30k_afrikaans

    # Dataset configurations
    FT_NUM_STEPS=30000
    TRAIN_BATCH_SIZE=256
    NUM_MICROBATCHES=4
    EVAL_BATCH_SIZE=128
    INFER_BATCH_SIZE=128
    LEARNING_RATE=0.0003
    LEARNING_RATE_SCHEDULE="constant"
    WARMUP_STEPS=3000

    FEATURE_LENGTHS="{'inputs': 1024, 'targets': 1024}"
    CHECKPOINT_PERIOD=5000
    EVAL_PERIOD=5000

    ADDITIONAL_GIN_CONFIGS=("--gin.LOSS_NORMALIZING_FACTOR=\"AVERAGE_PER_SEQUENCE\"")
    REMOVE_CHECKPOINTS=false

    # --------------------------------
    PRETRAINED_STEPS=${CHECKPOINT##*_}
    MODEL_SIZE=${CHECKPOINT%%/checkpoint*}
    MODEL_SIZE=${MODEL_SIZE##*_}
    mkdir -p {logs/$OUTPUT_DIR,runs/$OUTPUT_DIR}
    # ---------------------------------------------

    # Solve the following bash task 
    # TODO: if a variable task is set, print the task
    # TODO: If a variable LANGUAGES exist, for each language in the list, set task to language and print task 

    if [ -z "$TASKS" ]; then
        TASKS=()

        if [ -z "$LANGUAGES" ] && [ -z "$task" ]; then
            echo "Error: All of `TASKS`, `LANGUAGES` and `task` are unset."
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

    for task in "${TASKS[@]}"; do
        train_steps=$((PRETRAINED_STEPS + FT_NUM_STEPS))

        bash scripts/t5_utils.sh \
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
        --output_dir $seed_output_dir \
        --gin.infer_eval/utils.DatasetConfig.batch_size=$INFER_BATCH_SIZE \
        "${ADDITIONAL_GIN_CONFIGS[@]}" \
        >& logs/$OUTPUT_DIR/${task}_${seed}_ft.log \
        && finetuned=true

        checkpoints=($(ls $seed_output_dir | grep checkpoint))

        if [[ $REMOVE_CHECKPOINTS == "true" ]]; then
            for ckpt in "${checkpoints[@]}";
            do
                rm -rf "$seed_output_dir/$ckpt"
            done
        fi
    done
}