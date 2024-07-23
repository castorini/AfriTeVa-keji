#!/bin/bash
{
    # Use only Host 0
    # export TPU_CHIPS_PER_PROCESS_BOUNDS="2,2,1"
    # export TPU_PROCESS_BOUNDS="1,1,1"
    # export TPU_VISIBLE_DEVICES="0,1,2,3"

    set -e;
    set -x;

    export TASKS_TO_LOAD="aya"
    
    cd ~/nfs_share/AfriTeVa-keji
    source ~/venv/bin/activate

    CHECKPOINT="models/T5_1_1_large/checkpoint_524288"
    EXPERIMENT_NAME=afriteva_v2_large_aya_30k_all_african_languages

    # ------------------------------------------------------
    # Save model and exit: Model is saved and program exits.
    # Note: If the experiment has not run, the checkpoint passed is saved as a msgpack model
    # Else the latest checkpoint found in the output directory is saved.
    SAVE_MODEL_AND_EXIT=true
    # ------------------------------------------------------
    # Training configurations
    task="human_aya"
    FT_NUM_STEPS=5000
    TRAIN_BATCH_SIZE=256
    NUM_MICROBATCHES=16
    EVAL_BATCH_SIZE=32
    INFER_BATCH_SIZE=32
    LEARNING_RATE=0.0003
    LEARNING_RATE_SCHEDULE="constant"
    WARMUP_STEPS=3000
    CHECKPOINT_PERIOD=500
    EVAL_PERIOD=500
    FEATURE_LENGTHS="{'inputs': 1024, 'targets': 1024}"
    TRAIN_EVAL=false
    INFER_EVAL=false

    ADDITIONAL_GIN_CONFIGS=(
        "--gin.LOSS_NORMALIZING_FACTOR=\"AVERAGE_PER_SEQUENCE\""
        "--gin.train.use_orbax=True"
        # "--nig.multiprocess_gpu"
    )
    REMOVE_CHECKPOINTS=false

    # -----------------------------------
    if [[ $SAVE_MODEL_AND_EXIT == "true" ]]; then
        ADDITIONAL_GIN_CONFIGS+=("--gin.train.save_model_and_exit=True")
    fi

    if [[ $TRAIN_EVAL == "false" ]]; then
        ADDITIONAL_GIN_CONFIGS+=("--no_train_eval")
    elif [[ $TRAIN_EVAL == "true" ]]; then
        ADDITIONAL_GIN_CONFIGS+=("--gin.train_eval/utils.DatasetConfig.batch_size=$EVAL_BATCH_SIZE")
    else
        echo "TRAIN_EVAL should be true/false"
        exit 1
    fi

    if [[ $INFER_EVAL == "false" ]]; then
        ADDITIONAL_GIN_CONFIGS+=("--no_infer_eval")
    elif [[ $TRAIN_EVAL == "true" ]]; then
        ADDITIONAL_GIN_CONFIGS+=("--gin.infer_eval/utils.DatasetConfig.batch_size=$INFER_BATCH_SIZE")
    else
        echo "INFER_EVAL should be true/false"
        exit 1
    fi

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