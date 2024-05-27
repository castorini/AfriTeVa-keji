#!/bin/bash
{
    set -e;
    set -x;

    export TASKS_TO_LOAD="aya"

    CHECKPOINT="gs://afriqa-bucket/T5_1_1_base/checkpoint_524288"
    OUTPUT_DIR=afriteva_v2_base_aya_30k_afrikaans

    # Dataset configurations
    FT_NUM_STEPS=30000
    TRAIN_BATCH_SIZE=16
    EVAL_BATCH_SIZE=8
    INFER_BATCH_SIZE=4
    FEATURE_LENGTHS="{'inputs': 512, 'targets': 200}"
    CHECKPOINT_PERIOD=5000
    EVAL_PERIOD=5000

    ADDITIONAL_GIN_CONFIGS=("--gin.LOSS_NORMALIZING_FACTOR=AVERAGE_PER_SEQUENCE")
    REMOVE_CHECKPOINTS=false

    # --------------------------------
    PRETRAINED_STEPS=${CHECKPOINT##*_}
    MODEL_SIZE=${CHECKPOINT%%/checkpoint*}
    MODEL_SIZE=${MODEL_SIZE##*_}
    mkdir -p {logs/$OUTPUT_DIR,runs/$OUTPUT_DIR}
    # ---------------------------------------------

    LANGUAGES=("afrikaans")

    for language in "${LANGUAGES[@]}"
    do
        task="${language}_aya"
        train_steps=$((PRETRAINED_STEPS + FT_NUM_STEPS))

        for seed in 1
        do
            seed_output_dir=runs/$OUTPUT_DIR/${task}_${seed}

            # --gin.infer_eval/utils.DatasetConfig.batch_size=$INFER_BATCH_SIZE \
            bash scripts/t5_utils.sh \
            --action finetune \
            --task $task \
            --feature_lengths "$FEATURE_LENGTHS" \
            --batch_size $TRAIN_BATCH_SIZE \
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
    done
}