#!/bin/bash

{   
    # set -e;
    
    export CUDA_VISIBLE_DEVICES=0
    export WANDB_ENTITY="theyorubayesian"
    export WANDB_PROJECT="scaling-lms-for-african-languages"
    export WANDB_LOG_MODEL=false
    export WANDB_RUN_GROUP="classification/masakhanews"

    MODEL_SIZE="large"
    BATCH_SIZE=32
    NUM_EPOCHS=20
    LR=0.00005
    LR_SCHEDULER_TYPE="constant"
    # WARMUP_STEPS=0
    # WARMUP_RATIO=
    # --warmup_ratio --warmup_steps

    # LANGUAGES=("amh" "eng" "fra" "hau")
    LANGUAGES=("ibo" "lin" "lug" "orm")
    # LANGUAGES=("pcm" "run" "sna" "som")
    # LANGUAGES=("swa" "tir" "xho" "yor")
    # LANGUAGES=("xho" "yor")
    # LANGUAGES=("sna" "som")

    for language in "${LANGUAGES[@]}"  
    do
        for SEED in 1 2 3 4 5
        do
            RUN_DIR="runs/classification/afriteva_v2_${MODEL_SIZE}/${language}_${SEED}"

            if [ -d "$RUN_DIR" ]; then
                echo "$RUN_DIR exist. Skipping."
                continue
            fi

            mkdir -p $RUN_DIR

            set -x;

            python -m teva.torch.classification \
            --model_name_or_path "castorini/afriteva_v2_${MODEL_SIZE}" \
            --seed $SEED \
            --do_train \
            --do_eval \
            --do_predict \
            --num_train_epochs $NUM_EPOCHS \
            --dataset_name "masakhane/masakhanews" \
            --dataset_config "$language" \
            --source_prefix "classify: " \
            --output_dir "$RUN_DIR" \
            --learning_rate $LR \
            --lr_scheduler_type="$LR_SCHEDULER_TYPE" \
            --per_device_train_batch_size=$BATCH_SIZE \
            --per_device_eval_batch_size=$BATCH_SIZE \
            --overwrite_output_dir \
            --predict_with_generate \
            --eval_strategy "epoch" \
            --generation_max_length=2 \
            --generation_num_beams=1 \
            --report "wandb" >& "runs/classification/afriteva_v2_${MODEL_SIZE}/${language}_${SEED}-$(date +"%m-%d_%H-%M-%S").log" 

            du -a $RUN_DIR | grep checkpoint | cut -f 2 | xargs rm -r
            
            set +x;
        done
    done
}