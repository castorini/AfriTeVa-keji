{
    echo "
            Note: when supplying a string, dict, list, or tuple value via a flag, you must put it in quotes. 
            In the case of strings, it requires escaped quotes (\\\"<string>\\\"). 
            For example: 
            --gin.utils.DatasetConfig.split=\\\"validation\\\", 
            --gin.utils.DatasetConfig.task_feature_lengths=\"{'inputs': 512, 'targets': 84}\", and 
            --gin.dense.MlpBlock.activations=\"('dense', 'gelu')\"
        "

    # -----------------
    # Default arguments
    # -----------------
    batch_size=16
    eval_period=5000
    checkpoint_period=5000
    model_size="base"

    # --- 
    # CLI
    # ---
    ensure() {
        [[ -z "$2" ]] && echo "ERROR: \`$1\` is null or unset" && exit
    }

    other_configs=()

    while [ "$1" != "" ]; do
        echo $1
        # Collect any additional gin configs and pass to action command
        [[ $1 == --gin* ]] && other_configs+=($1) && shift && continue

        case $1 in
            -a | --action )                 shift
                                            action=$1
                                            ;;
            -b | --batch_size )             shift
                                            batch_size=$1
                                            ;;
            -c | --checkpoint )             shift
                                            checkpoint=\"$1\"
                                            ;;
            -cp | --checkpoint_period )     shift
                                            checkpoint_period=$1
                                            ;;
            --cuda_12 )                     export NCCL_P2P_DISABLE=1
                                            ;;
            --no_infer_eval )               no_infer_eval=false
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
            --task )                         shift
                                            task=$1
                                            ;;
            -t | --train_steps )            shift
                                            train_steps=$1
                                            ;;
            -o | --output_dir )             shift
                                            output_dir=$1
                                            ;;
            -h | --help )                   exit    # TODO: @theyorubayesian
                                            ;;
        esac
        shift
    done

    ensure "action" $action
    ensure "output_dir" $output_dir
    ensure "task" $task
    ensure "checkpoint" $checkpoint
    ensure "feature_lengths" $feature_lengths

    if [[ $action == "pretrain" || $action == "finetune" ]]; then
        ensure "train_steps" $train_steps
    fi

    echo ${other_configs[@]}
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
            --gin.TASK_FEATURE_LENGTHS="$feature_lengths" \
            --gin.TRAIN_STEPS=${train_steps} \
            --gin.BATCH_SIZE=${batch_size} \
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
            --gin.TRAIN_STEPS=${train_steps} \
            --gin.BATCH_SIZE=${batch_size} \
            --gin.INITIAL_CHECKPOINT_PATH=${checkpoint} \
            --gin.EVAL_PERIOD=${eval_period} \
            --gin.utils.SaveCheckpointConfig.period=${checkpoint_period} \
            --gin.MODEL_DIR=\"${output_dir}\" \
            "${other_configs[@]}" \
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