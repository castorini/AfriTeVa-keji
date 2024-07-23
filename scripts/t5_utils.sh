#!/bin/bash
{   
    gin() {
    echo "NOTE: When supplying a string, dict, list, or tuple value via a flag, you must put it in quotes. 
        In the case of strings, it requires escaped quotes (\\\"<string>\\\"). 
        For example: 
        --gin.utils.DatasetConfig.split=\\\"validation\\\", 
        --gin.utils.DatasetConfig.task_feature_lengths=\"{'inputs': 512, 'targets': 84}\", and 
        --gin.dense.MlpBlock.activations=\"('dense', 'gelu')\""
    }

    usage() {
        echo "Usage: t5_utils --a ACTION --t TASK -f FEATURE_LENGTHS -o OUTPUT_DIR [OPTIONS]... [EXTRA_GIN_CONFIGS]"
        echo
        echo "----------"
        echo "Arguments:"
        echo "----------"
        echo "  -a, --action=ACTION                     Action to perform. ACTION is one of: 'pretrain', 'finetune', 'infer', 'eval'"
        echo "  -t, --task=TASK                         Registered seqio TASK for ACTION" 
        echo "  -f, --feature_lengths=FEATURE_LENGTHS   String dictionary mapping feature to maximum length, e.g. \"{'feature_a': LENGTH, 'feature_b': LENGTH}\""      
        echo "  -o, --output_dir=OUTPUT_DIR             Output directory. Must be local if ACTION is one of: 'finetune', 'infer', 'eval'"
        echo 
        echo "--------"
        echo "Options:"
        echo "--------"
        echo "  -h, --help                              Display help text and exit"
        echo "  -t, --train_steps=NUM                   Number of steps to train for. Optional if ACTION is not 'pretrain' or 'finetune'"
        echo "  -m, --model_size=MODEL_SIZE             MODEL_SIZE is one of: 'base', 'large'. (optional, default: base)"
        echo "  -b, --batch_size=NUM                    Use NUM as batch size for ACTION. (optional, default: 16)"
        echo "  -n, --num_microbatches=NUM              NUM microbatches to split 'batch_size' into (optional, default: 1)"
        echo "  -e, --eval_period=NUM                   EvaluationPeriod. Must be multiple of 'checkpoint_period' (optional, default: 5000)"
        echo "  -c, --checkpoint=CHECKPOINT             Path to checkpoint for ACTION. Optional if ACTION is not 'pretrain'"
        echo "  -cp, --checkpoint_period=NUM            SaveCheckpointPeriod. Must be multiple of 'eval_period' (optional, default: 5000)"
        echo "  --cuda_12                               Indicate that hardware is CUDA 12 so that 'NCCL_P2P_DISABLE=1' is set"
        echo "  --no_infer_eval                         Disable inference_evaluation is not performed during training."
        echo 
        echo "-------------------"
        echo "Gin Configurations:"
        echo "-------------------"
        echo "Arguments that begin with '--gin' are passed straight to t5x"
        echo
        gin
        echo
        echo "Author: Akintunde 'theyorubayesian' Oladipo"
        exit 1
    }
    # -----------------
    # Default arguments
    # -----------------
    batch_size=16
    eval_period=5000
    checkpoint_period=5000
    model_size="base"
    num_microbatches=1

    # --- 
    # CLI
    # ---
    ensure() {
        [[ -z "$2" ]] && echo "ERROR: \`$1\` is null or unset" && exit
    }

    other_configs=()

    while [ "$1" != "" ]; do
        # Collect any additional gin configs and pass to action command
        [[ $1 == --gin* ]] && other_configs+=($1) && shift && continue

        # Commands that start with --nig. are stripped and passed to the action script directly
        [[ $1 == --nig.* ]] && suffix=${1#*nig.} && other_configs+=("--${suffix}") && shift && continue

        case $1 in
            -a | --action )                 shift
                                            action=$1
                                            ;;
            -b | --batch_size )             shift
                                            batch_size=$1
                                            ;;
            --learning_rate )               shift
                                            learning_rate=$1
                                            ;;
            --learning_rate_schedule )      shift
                                            learning_rate_schedule=$1
                                            ;;
            --warmup_steps )                shift
                                            warmup_steps=$1
                                            ;;
            -n | --num_microbatches )       shift
                                            num_microbatches=$1
                                            ;;
            -c | --checkpoint )             shift
                                            checkpoint=\"$1\"
                                            ;;
            -cp | --checkpoint_period )     shift
                                            checkpoint_period=$1
                                            ;;
            --cuda_12 )                     export NCCL_P2P_DISABLE=1
                                            ;;
            --no_train_eval )               no_train_eval=true
                                            ;;
            --no_infer_eval )               no_infer_eval=true
                                            ;;
            -e | --eval_period )            shift
                                            eval_period=$1
                                            ;;
            -f | --feature_lengths )        shift
                                            feature_lengths=$1
                                            ;;
            -m | --model_size )             shift
                                            model_size=$1
                                            ;;
            --task )                        shift
                                            task=$1
                                            ;;
            -t | --train_steps )            shift
                                            train_steps=$1
                                            ;;
            -o | --output_dir )             shift
                                            output_dir=$1
                                            ;;
            -h | --help )                   usage
                                            ;;
        esac
        shift
    done

    ensure "action" "$action"
    ensure "output_dir" "$output_dir"
    ensure "task" "$task"
    ensure "feature_lengths" "$feature_lengths"

    if [[ $action == "pretrain" ]]; then
        : "${LEARNING_RATE_SCHEDULE="constant * rsqrt_decay"}"
        : "${LEARNING_RATE=1.0}"
        : "${WARMUP_STEPS=10000}"
    fi

    if [[ $action == "finetune" ]]; then
        : "${LEARNING_RATE_SCHEDULE="constant"}"
        : "${LEARNING_RATE=0.001}"
        : "${WARMUP_STEPS=10000}"
    fi

    if [[ $action == "pretrain" || $action == "finetune" ]]; then
        ensure "train_steps" "$train_steps"
    fi

    # ----------------------------------- #
    case $model_size in
        base)
            ;;
        large)
            ;;
        *)
            echo "Model size must be one of: base, large"
            exit
            ;;
    esac

    case $action in
        pretrain)
            set -x;
            python3 -m t5x.main \
            --run_mode="train" \
            --gin_file="config/models/t5_1_1/$model_size.gin" \
            --gin_file="config/runs/t5_1_1/pretrain.gin" \
            --gin.MIXTURE_OR_TASK_NAME=\"${task}\" \
            --gin.TASK_FEATURE_LENGTHS="${feature_lengths}" \
            --gin.TASK_FEATURE_LENGTHS="${feature_lengths}" \
            --gin.WARMUP_STEPS=${warmup_steps} \
            --gin.LEARNING_RATE=\"${learning_rate}\" \
            --gin.LEARNING_RATE_SCHEDULE=\"${learning_rate_schedule}\" \
            --gin.TRAIN_STEPS=${train_steps} \
            --gin.BATCH_SIZE=${batch_size} \
            --gin.trainer.Trainer.num_microbatches=${num_microbatches} \
            --gin.train.eval_period=${eval_period} \
            --gin.utils.SaveCheckpointConfig.period=${checkpoint_period} \
            --gin.MODEL_DIR=\"${output_dir}\" \
            "${other_configs[@]}" \
            --alsologtostderr
            ;;
        finetune)
            set -x;
            python3 -m t5x.train \
            --gin_file="config/models/t5_1_1/$model_size.gin" \
            --gin_file="config/runs/t5_1_1/finetune.gin" \
            --gin.MIXTURE_OR_TASK_NAME=\"${task}\" \
            --gin.TASK_FEATURE_LENGTHS="${feature_lengths}" \
            --gin.WARMUP_STEPS=${warmup_steps} \
            --gin.LEARNING_RATE=${learning_rate} \
            --gin.LEARNING_RATE_SCHEDULE=\"${learning_rate_schedule}\" \
            --gin.TRAIN_STEPS=${train_steps} \
            --gin.BATCH_SIZE=${batch_size} \
            --gin.trainer.Trainer.num_microbatches=${num_microbatches} \
            --gin.INITIAL_CHECKPOINT_PATH=${checkpoint} \
            --gin.EVAL_PERIOD=${eval_period} \
            --gin.utils.SaveCheckpointConfig.period=${checkpoint_period} \
            --gin.MODEL_DIR=\"${output_dir}\" \
            "${other_configs[@]}" \
            ${no_train_eval+--gin.train.train_eval_dataset_cfg=None} \
            ${no_infer_eval+--gin.train.infer_eval_dataset_cfg=None} \
            --alsologtostderr 
            ;;
        eval)
            set -x;
            python3 -m t5x.eval \
            --gin_file="config/models/t5_1_1/$model_size.gin" \
            --gin_file="config/runs/t5_1_1/eval.gin" \
            --gin.MIXTURE_OR_TASK_NAME=\"${task}\" \
            --gin.TASK_FEATURE_LENGTHS="${feature_lengths}" \
            --gin.CHECKPOINT_PATH=${checkpoint} \
            --gin.EVAL_OUTPUT_DIR=\"${output_dir}\" \
            --alsologtostderr \
            "${other_configs[@]}"
            ;;
        infer)
            set -x;
            python3 -m t5x.infer \
            --gin_file="config/models/t5_1_1/$model_size.gin" \
            --gin_file="config/runs/t5_1_1/infer.gin" \
            --gin.INFER_OUTPUT_DIR=\"${output_dir}\" \
            --alsologtostderr \
            "${other_configs[@]}"
            ;;
        *)
            echo "$action may only be one of: pretrain, finetune, eval, infer"
            ;;
    esac
}