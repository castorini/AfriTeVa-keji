#!/bin/bash

{
    set -e;

    export CUDA_VISIBLE_DEVICES=0,1,2,3
    export WANDB_ENTITY="theyorubayesian"
    export WANDB_PROJECT="scaling-lms-for-african-languages"
    export WANDB_LOG_MODEL=false
    export WANDB_RUN_GROUP="summarization/xlsum"

    delimit() {
        local IFS="$1"
        shift
        echo "$*"
    }

    FT_CONFIG=all                               # Finetune on all languages in AfriTeVa V2 pretraining data
    # FT_CONFIG=african_only
    # FT_CONFIG=non_african_only
    # FT_CONFIG=hausa                             # Finetune on a single language alone

    MODEL_SIZE="base"

    BATCH_SIZE=32
    NUM_MICROBATCHES=2
    SAMPLING_FACTOR=0.5
    SOURCE_LENGTH=512
    TARGET_LENGTH=64

    # TODO: Explore FT on HR languages before finetuning on LR
    # NUM_EPOCHS=6-10                       # They used epochs for single language finetuning, 32 batch size, slanted LR schedule Howard & Ruder, 2018. Eval constantly to battle overfitting
    NUM_STEPS=50000                         # 35000
    EVAL_STRATEGY="steps"
    EVAL_STEPS=5000
    LABEL_SMOOTHING_FACTOR=0.1
    BEAM_SIZE=4
    LENGTH_PENALTY=0.6

    # https://github.com/huggingface/transformers/blob/dcdda5324bcc7a750b5e40e11dd795442204ff27/src/transformers/optimization.py#L457
    LR=0.0005
    LR_SCHEDULER_TYPE="inverse_sqrt"            # "constant_with_warmup" "constant"
    WARMUP_STEPS=5000
    WEIGHT_DECAY=0.01

    # -------------------------------------
    readarray -d , -t GPUS <<< "$CUDA_VISIBLE_DEVICES"
    N_GPU=${#GPUS[@]}

    AFRICAN_LANGUAGES=(
        "amharic"  "hausa" "igbo" "kirundi"
        "oromo" "pidgin" "somali" "swahili"
        "tigrinya" "yoruba"
    )
    NON_AFRICAN_LANGUAGES=("arabic" "english" "french" "portuguese")

    if [[ $FT_CONFIG == "african_only" ]]; then
        dataset_config="$(delimit , ${AFRICAN_LANGUAGES[@]})"
    elif [[ $FT_CONFIG == "non_african_only" ]]; then
        dataset_config="$(delimit , ${NON_AFRICAN_LANGUAGES[@]})"
    elif [[ $FT_CONFIG == "all" ]]; then
        dataset_config="$(delimit , ${AFRICAN_LANGUAGES[@]} ${NON_AFRICAN_LANGUAGES[@]})"
    else
        dataset_config="$FT_CONFIG"
    fi
    # ------------------------------
    set -x;

    RUN_DIR="runs/summarization/afriteva_v2_${MODEL_SIZE}/${FT_CONFIG}"

    if [ -d "$RUN_DIR" ]; then
        echo "$RUN_DIR exist. Skipping."
        # exit;
    fi

    mkdir -p $RUN_DIR

    OMP_NUM_THREADS=12 torchrun --nproc_per_node "$N_GPU" --master_port=25678 \
    -m teva.torch.summarization \
    --model_name_or_path "castorini/afriteva_v2_${MODEL_SIZE}" \
    --do_train \
    --do_eval \
    --do_predict \
    --max_steps $NUM_STEPS \
    --dataset_name "csebuetnlp/xlsum" \
    --sampling_factor $SAMPLING_FACTOR \
    --dataset_config "$dataset_config" \
    --source_prefix "summarize: " \
    --text_column "text" \
    --max_source_length $SOURCE_LENGTH \
    --summary_column "summary" \
    --max_target_length $TARGET_LENGTH \
    --output_dir "$RUN_DIR" \
    --optim adafactor \
    --learning_rate $LR \
    --dispatch_batches True \
    --lr_scheduler_type="$LR_SCHEDULER_TYPE" \
    --label_smoothing_factor=$LABEL_SMOOTHING_FACTOR \
    --per_device_train_batch_size=$BATCH_SIZE \
    --gradient_accumulation_steps=$NUM_MICROBATCHES \
    --per_device_eval_batch_size=$BATCH_SIZE \
    --ignore_pad_token_for_loss \
    --overwrite_output_dir \
    --predict_with_generate \
    --eval_strategy $EVAL_STRATEGY \
    --eval_steps $EVAL_STEPS \
    --save_steps $EVAL_STEPS \
    --generation_num_beams=$BEAM_SIZE \
    --generation_max_length=$TARGET_LENGTH \
    --length_penalty=$LENGTH_PENALTY \
    --report "wandb" >& "runs/summarization/afriteva_v2_${MODEL_SIZE}/${FT_CONFIG}-$(date +"%m-%d_%H-%M-%S").log"
}