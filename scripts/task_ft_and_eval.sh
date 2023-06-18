{
    set -x;
    export CUDA_VISIBLE_DEVICES="0,1,2,3"

    # Pass `true` if you you set env var `DATA_GCP_DIR` to a local path on your machine
    USING_LOCAL_DATASET=true
    # Pass full bucket dir for dataset if dataset is not on local.
    DATASET_DIR=data/lafand                                                 # TODO: Change based on your task
    TRAIN_BATCH_SIZE=32
    EVAL_BATCH_SIZE=16
    INFER_BATCH_SIZE=64                                                     # TODO: Reduce by half if OOM error during inference_evaluation
    CHECKPOINT="gs://awarawa/T5_1_1_base/checkpoint_524288"                 # TODO: Change to the checkpoint you want to value on
    CHECKPOINT_PERIOD=auto                                                  # If auto, we save checkpoint after every epoch. Otherwise set to value.
    MODEL_SIZE="base"
    EVAL_PERIOD=auto                                                        # If auto, we run evaluations after every epoch. Otherwise set to value.
    # Please pass FEATURE_LENGTHS as string dictionary.
    FEATURE_LENGTHS="{'inputs': 512, 'targets': 200}"                       # TODO: Change based on your task
    # We pretrained for 524288 steps if you use the final checkpoints.
    # If you use any other checkpoint, take note of its pre-trained steps.
    PRETRAINED_STEPS=524288
    FT_NUM_EPOCHS=5
    OUTPUT_DIR="arawat5_base_lafand"                                        # TODO: Change to unique output dir
    mkdir -p logs/$OUTPUT_DIR

    REMOVE_CHECKPOINTS=true
    # ---------------------------------------------

    LANGUAGES=("yor")                                                       # TODO: Use the list defined for the task in src/teva/tasks.py

    for language in ${LANGUAGES[@]}
    do
        LANGUAGE_DATASET_DIR=$DATASET_DIR/en-${language}/train.json         # TODO: Change path so that we can match the train set of each language
        task=                                                               # TODO: See src/teva/tasks.py. Please change this so that we get the correct task for each language.                                    
        # --------------------------------------------------------------------
        # Unfortunately, t5x uses number of steps rather than number of epochs
        # We dynamically calculate the number of steps for each language's finetuning task using
        # the number of examples it has, the `TRAIN_BATCH_SIZE` and `FT_NUM_EPOCHS`
        if [[ $USING_LOCAL_DATASET == "true" ]]; then
            num_examples=$(wc -l $LANGUAGE_DATASET_DIR | cut -f 1 -d " ")
        else
            num_examples=$(gsutil cat $LANGUAGE_DATASET_DIR | wc -l)
        fi

        num_steps_per_epoch=$((num_examples/TRAIN_BATCH_SIZE))
        ft_steps=$((FT_NUM_EPOCHS * num_steps_per_epoch))
        # TRAIN_STEPS MUST ALWAYS BE pre-trained steps + no. of fine-tuning steps.
        train_steps=$((PRETRAINED_STEPS + ft_steps))

        [[ $EVAL_PERIOD == "auto" ]] && _EVAL_PERIOD=$num_steps_per_epoch || _EVAL_PERIOD=$EVAL_PERIOD
        [[ $CHECKPOINT_PERIOD == "auto" ]] && _CHECKPOINT_PERIOD=$num_steps_per_epoch || _CHECKPOINT_PERIOD=$CHECKPOINT_PERIOD
        # ------------------------------------------------------------------------

        for seed in 1 2 3
        do
            seed_output_dir=runs/$OUTPUT_DIR/${task}_${seed}

            # For some tasks, running inference_evaluation during training causes 00M
            # no matter how small the `INFER_BATCH_SIZE`
            # Replace `--gininfer_eval.utils.DatasetConfig.batch_size=$INFER_BATCH_SIZE` with `--no_infer_eval`` 
            # to disable inference evaluation during training.
            # You will need to run evaluation on the checkpoints after training is done.
            # TODO: Pass `--cuda_12` command if you're on CUDA 12
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
            --gin.infer_eval/utils.DatasetConfig.batch_size=$INFER_BATCH_SIZE \
            >& logs/$OUTPUT_DIR/${task}_${seed}_ft.log \
            && finetuned=true

            checkpoints=($(ls $seed_output_dir | grep checkpoint | grep -v $PRETRAINED_STEPS))

            # Uncomment if you are using `no_infer_eval` when finetuning.
            # This will run inference evaluation on checkpoints produced during training
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
                for checkpoint in ${checkpoints[@]};
                do
                    rm -rf $seed_output_dir/$checkpoint
                done
            fi
        done
    done
}