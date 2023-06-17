{
    set -x;
    set -e;
    export CUDA_VISIBLE_DEVICES="0,1"

    # Pass `true` if you you set env var `DATA_GCP_DIR` to a local path on your machine
    USING_LOCAL_DATASET=true
    # Pass full bucket dir for dataset if dataset is not on local.
    DATASET_DIR=data/lafand
    TRAIN_BATCH_SIZE=16
    EVAL_BATCH_SIZE=64
    INFER_BATCH_SIZE=64
    CHECKPOINT="gs://awarawa/T5_1_1_base/checkpoint_524288"
    CHECKPOINT_PERIOD=auto
    MODEL_SIZE="base"
    EVAL_PERIOD=auto
    # Please pass FEATURE_LENGTHS as string dictionary.
    FEATURE_LENGTHS="{'inputs': 512, 'targets': 200}"
    # We pretrained for 524288 steps if you use the final checkpoints.
    # If you use any other checkpoint, take note of its pre-trained steps.
    PRETRAINED_STEPS=524288
    FT_NUM_EPOCHS=5
    OUTPUT_DIR="arawat5_base_lafand_hau_pcm_swa"
    mkdir -p logs/$OUTPUT_DIR
    REMOVE_CHECKPOINTS=true
    # ---------------------------------------------

    # LANGUAGES=("hau" "pcm" "swa" "ibo" "yor" "zul")
    LANGUAGES=("hau" "pcm" "swa")
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

            [[ $EVAL_PERIOD == "auto" ]] && EVAL_PERIOD=$num_steps_per_epoch
            [[ $CHECKPOINT_PERIOD == "auto" ]] && CHECKPOINT_PERIOD=$num_steps_per_epoch

            for seed in 1 2 3
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
                --checkpoint_period $CHECKPOINT_PERIOD \
                --eval_period $EVAL_PERIOD \
                --train_steps $train_steps \
                --model_size $MODEL_SIZE \
                --output_dir $seed_output_dir \
                --cuda_12 \
                --gin.infer_eval/utils.DatasetConfig.batch_size=$INFER_BATCH_SIZE \
                >& logs/$OUTPUT_DIR/${task}_${seed}_ft.log \
                && finetuned=true

                checkpoints=($(ls $seed_output_dir | grep checkpoint | grep -v "524288"))

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
                        rm -rf $ckpt
                    done
                fi
            done
        done
    done
}