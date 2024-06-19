#!/bin/bash

{   
    set -e;

    export CUDA_VISIBLE_DEVICES=0
    export WANDB_ENTITY="theyorubayesian"
    export WANDB_PROJECT="scaling-lms-for-african-languages"
    export WANDB_LOG_MODEL=false
    export WANDB_RUN_GROUP="translation/mafand"

    declare -A KNOWN_LANGUAGES=(
        ["hau"]="en"
        ["ibo"]="en"
        ["pcm"]="en"
        ["swa"]="en"
        ["yor"]="en"
        ["zul"]="en"
        # ["sna"]="en"  # No train dataset
        # ["amh"]="en"  # No train dataset
        # ["kin"]="en"  # No train dataset
        # ["nya"]="en"  # No train dataset
        # ["xho"]="en"  # No train dataset
    )
    declare -A UNKNOWN_LANGUAGES=(
        ["lug"]="en"
        ["luo"]="en"
        ["tsn"]="en"
        ["twi"]="en"
        ["bam"]="fr"
        ["bbj"]="fr"
        ["ewe"]="fr"
        ["fon"]="fr"
        ["mos"]="fr"
        ["wol"]="fr"
    )
    declare -A LANGUAGE_MAP=(
        ["amh"]="Amharic"
        ["en"]="English"
        ["fr"]="French"
        ["hau"]="Hausa"
        ["ibo"]="Igbo"
        ["kin"]="Kinyarwanda"
        ["nya"]="Chichewa"
        ["pcm"]="Pidgin"
        ["sna"]="Shona"
        ["swa"]="Swahili"
        ["xho"]="Xhosa"
        ["yor"]="Yoruba"
        ["zul"]="Zulu"
        ["bam"]="Bambara"
        ["bbj"]="Ghomálá'"
        ["ewe"]="Éwé"
        ["fon"]="Fon"
        ["lug"]="Luganda"
        ["luo"]="Luo"
        ["mos"]="Mos"
        ["tsn"]="Setswana"
        ["twi"]="Twi"
        ["wol"]="Wolof"
    )

    LR=0.00005
    LR_SCHEDULER_TYPE="constant"

    MODEL_SIZE="large"
    BATCH_SIZE=10
    NUM_EPOCHS=3  # They used 10 for models pretrained on NEWS alone
    SOURCE_LENGTH=200
    TARGET_LENGTH=200
    BEAM_SIZE=10
    
    for language in ${!UNKNOWN_LANGUAGES[@]}
    do
        set -x;

        pivot=${UNKNOWN_LANGUAGES[$language]}

        for config in "$language-$pivot" "$pivot-$language"
        do
            source=${config%-*}
            target=${config#*-}

            RUN_DIR="runs/translation/afriteva_v2_${MODEL_SIZE}/${source}_${target}"

            if [ -d "$RUN_DIR" ]; then
                echo "$RUN_DIR exist. Skipping."
                continue
            fi

            mkdir -p $RUN_DIR

            python -m teva.torch.translation \
            --model_name_or_path "castorini/afriteva_v2_${MODEL_SIZE}" \
            --do_train \
            --do_eval \
            --do_predict \
            --num_train_epochs $NUM_EPOCHS \
            --source_lang "$source" \
            --max_source_length $SOURCE_LENGTH \
            --target_lang  $target \
            --max_target_length $TARGET_LENGTH \
            --source_prefix "Translate ${LANGUAGE_MAP[$source]} to ${LANGUAGE_MAP[$target]}: " \
            --dataset_name "data/mafand" \
            --dataset_config_name "${pivot}-${language}" \
            --output_dir "$RUN_DIR" \
            --learning_rate $LR \
            --lr_scheduler_type="$LR_SCHEDULER_TYPE" \
            --per_device_train_batch_size=$BATCH_SIZE \
            --per_device_eval_batch_size=$BATCH_SIZE \
            --ignore_pad_token_for_loss \
            --overwrite_output_dir \
            --predict_with_generate \
            --eval_strategy "epoch" \
            --generation_num_beams=$BEAM_SIZE \
            --generation_max_length=$TARGET_LENGTH \
            --report "wandb" >& "runs/translation/afriteva_v2_${MODEL_SIZE}/${source}_${target}-$(date +"%m-%d_%H-%M-%S").log" 

            du -a $RUN_DIR | grep checkpoint | cut -f 2 | xargs rm -r
        done
        set +x;
    done
}