#!/bin/bash
{
    set -x;
    set -e;
    
    export CUDA_VISIBLE_DEVICES="4,5"

    TRAIN_BATCH_SIZE=256                                                    # TODO: Reduce by half if OOM error during inference_evaluation
    FT_NUM_STEPS=50000
    NUM_MICROBATCHES=4
    # Note that we expect the checkpoint path to be of the form `/path/to/T5_1_1_MODEL_SIZE/checkpoint_PRETRAINED_STEPS/`
    CHECKPOINT="models/T5_1_1_base/checkpoint_524288"                       # TODO: Change to the checkpoint you want to value on
    CHECKPOINT_PERIOD=5000                                                  # If auto, we save checkpoint after every epoch. Otherwise set to value.
    EVAL_PERIOD=5000                                                        # If auto, we run evaluations after every epoch. Otherwise set to value.
    OUTPUT_DIR="arawat5_base_xlsum_actual_beam_search_4_correct"                    # TODO: Change to unique output dir
    # Please pass FEATURE_LENGTHS as string dictionary.
    FEATURE_LENGTHS="{'inputs': 512, 'targets': 64}"                        # TODO: Change based on your task
    ADDITIONAL_GIN_CONFIGS=()
    REMOVE_CHECKPOINTS=true
    
    # --------------------------------
    PRETRAINED_STEPS=${CHECKPOINT##*_}
    TRAIN_STEPS=$((PRETRAINED_STEPS + FT_NUM_STEPS))
    MODEL_SIZE=${CHECKPOINT%%/checkpoint*}
    MODEL_SIZE=${MODEL_SIZE##*_}
    mkdir -p {logs/$OUTPUT_DIR,runs/$OUTPUT_DIR}
    # ---------------------------------------------

    task="xlsum"
    seed_output_dir="runs/$OUTPUT_DIR/xlsum_xlingual_ft"

    # For some tasks, running inference_evaluation during training causes 00M
    # no matter how small the `INFER_BATCH_SIZE`
    # Replace `--gin.infer_eval.utils.DatasetConfig.batch_size=$INFER_BATCH_SIZE` with `--no_infer_eval`` 
    # to disable inference evaluation during training.
    # You will need to run evaluation on the checkpoints after training is done.
    # TODO: Remove the `--cuda_12` command if you're not on CUDA 12
    bash scripts/t5_utils.sh \
    --action finetune \
    --task $task \
    --feature_lengths "$FEATURE_LENGTHS" \
    --batch_size $TRAIN_BATCH_SIZE \
    --num_microbatches $NUM_MICROBATCHES \
    --checkpoint $CHECKPOINT \
    --checkpoint_period $CHECKPOINT_PERIOD \
    --eval_period $EVAL_PERIOD \
    --train_steps $TRAIN_STEPS \
    --model_size "$MODEL_SIZE" \
    --output_dir $seed_output_dir \
    --cuda_12 \
    "${ADDITIONAL_GIN_CONFIGS[@]}" \
    >& "logs/$OUTPUT_DIR/xlsum_xlingual_ft.log" \
    && finetuned=true

    checkpoints=($(ls $seed_output_dir | grep checkpoint))

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
        for checkpoint in "${checkpoints[@]}";
        do
            rm -rf "$seed_output_dir/${checkpoint:?}"
        done
    fi
}