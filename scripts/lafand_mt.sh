{
    set -x;
    set -e;
    export CUDA_VISIBLE_DEVICES="2,3"

    # Pass `true` if you you set env var `DATA_GCP_DIR` to a local path on your machine
    USING_LOCAL_DATASET=true
    # Pass full bucket dir for dataset if dataset is not on local.
    DATASET_DIR=data/lafand
    TRAIN_BATCH_SIZE=16
    EVAL_BATCH_SIZE=8
    INFER_BATCH_SIZE=4
    FT_NUM_EPOCHS=5
    # Please pass FEATURE_LENGTHS as string dictionary.
    FEATURE_LENGTHS="{'inputs': 512, 'targets': 200}"
    # Note that we expect the checkpoint path to be of the form `/path/to/T5_1_1_MODEL_SIZE/checkpoint_PRETRAINED_STEPS/``
    CHECKPOINT="gs://awarawa/T5_1_1_large/checkpoint_300000"
    OUTPUT_DIR="arawat5_large_lafand_beam_search_5"
    CHECKPOINT_PERIOD=auto
    EVAL_PERIOD=auto
    REMOVE_CHECKPOINTS=true

    # --------------------------------
    PRETRAINED_STEPS=${CHECKPOINT##*_}
    MODEL_SIZE=${CHECKPOINT%%/checkpoint*}
    MODEL_SIZE=${MODEL_SIZE##*_}
    mkdir -p {logs/$OUTPUT_DIR,runs/$OUTPUT_DIR}
    # ---------------------------------------------

    LANGUAGES=("hau" "pcm" "swa" "ibo" "yor" "zul" "tsn" "twi")
    # LANGUAGES=("pcm" "swa")
    # LANGUAGES=("ibo" "yor" "zul")
    for language in ${LANGUAGES[@]}
    do
        # TODO: You can check the task name format in src/teva/tasks.py
        for task in "${language}_en_lafand_mt" "en_${language}_lafand_mt"
        do
            if [[ $USING_LOCAL_DATASET == "true" ]]; then
                num_examples=$(wc -l $DATASET_DIR/en-${language}/train.json | cut -f 1 -d " ")
            else
                num_examples=$(gsutil cat $DATASET_DIR/en-${language}/train.json | wc -l)
            fi

            num_steps_per_epoch=$((num_examples/TRAIN_BATCH_SIZE))
            ft_steps=$((FT_NUM_EPOCHS * num_steps_per_epoch))
            # TRAIN_STEPS MUST ALWAYS BE pre-trained steps + no. of fine-tuning steps.
            train_steps=$((PRETRAINED_STEPS + ft_steps))

            [[ $EVAL_PERIOD == "auto" ]] && _EVAL_PERIOD=$num_steps_per_epoch || _EVAL_PERIOD=$EVAL_PERIOD
            [[ $CHECKPOINT_PERIOD == "auto" ]] && _CHECKPOINT_PERIOD=$num_steps_per_epoch || _CHECKPOINT_PERIOD=$CHECKPOINT_PERIOD

            for seed in 1
            do
                seed_output_dir=runs/$OUTPUT_DIR/${task}_${seed}

                # TODO: Remove the `--cuda_12` command if you're not on CUDA 12
                # --gin.infer_eval/utils.DatasetConfig.batch_size=$INFER_BATCH_SIZE \
                bash scripts/t5_utils.sh \
                --action finetune \
                --task $task \
                --feature_lengths "$FEATURE_LENGTHS" \
                --batch_size $TRAIN_BATCH_SIZE \
                --checkpoint $CHECKPOINT \
                --checkpoint_period $_CHECKPOINT_PERIOD \
                --eval_period $_EVAL_PERIOD \
                --train_steps $train_steps \
                --model_size $MODEL_SIZE \
                --output_dir $seed_output_dir \
                --cuda_12 \
                --gin.infer_eval/utils.DatasetConfig.batch_size=$INFER_BATCH_SIZE \
                >& logs/$OUTPUT_DIR/${task}_${seed}_ft.log \
                && finetuned=true

                checkpoints=($(ls $seed_output_dir | grep checkpoint))

                # Uncomment if you are using `no_infer_eval` when finetuning.
                # for checkpoint in ${checkpoints[@]}
                # do
                #     checkpoint_steps=${checkpoint##*_}
                #     bash scripts/t5_utils.sh \
                #     --action eval \
                #     --task $task \
                #     --feature_lengths "$FEATURE_LENGTHS" \
                #     --model_size $MODEL_SIZE \
                #     --checkpoint $seed_output_dir/$checkpoint \
                #     --batch_size $EVAL_BATCH_SIZE \
                #     --output_dir $seed_output_dir/eval_${checkpoint_steps} \
                #     --cuda_12 \
                #     >& logs/$OUTPUT_DIR/${task}_${seed}_eval_${checkpoint_steps}.log
                # done

                if [[ $REMOVE_CHECKPOINTS == "true" ]]; then
                    for ckpt in ${checkpoints[@]};
                    do
                        rm -rf $seed_output_dir/$ckpt
                    done
                fi
            done
        done
    done
}