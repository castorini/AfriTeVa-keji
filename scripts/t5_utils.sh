ACTION=$1 && shift
MODEL_SIZE=$1 && shift
OUTPUT_DIR=$1 && shift

if [ $# -gt 0 ]; then
    # We assume the remaining arguments passed are gin configurations
    echo "
        Note: when supplying a string, dict, list, or tuple value via a flag, you must put it in quotes. 
        In the case of strings, it requires escaped quotes (\\\"<string>\\\"). 
        For example: 
        --gin.utils.DatasetConfig.split=\\\"validation\\\", 
        --gin.utils.DatasetConfig.task_feature_lengths=\"{'inputs': 512, 'targets': 84}\", and 
        --gin.dense.MlpBlock.activations=\"('dense', 'gelu')\"
    "
fi

case $MODEL_SIZE in
    base)
        ;;
    large)
        ;;
    *)
        echo "Model size must be one of: base, large"
        exit
esac

case $ACTION in
    pretrain)
        set -x;
        python -m t5x.main \
        --run_mode="train" \
        --gin_file="config/models/t5_1_1/$MODEL_SIZE.gin" \
        --gin_file="config/runs/t5_1_1/pretrain.gin" \
        --gin.MODEL_DIR=\"${OUTPUT_DIR}\" \
        "$@" \
        --alsologtostderr
        ;;
    finetune)
        set -x;
        python -m t5x.train \
        --gin_file="config/models/t5_1_1/$MODEL_SIZE.gin" \
        --gin_file="config/runs/t5_1_1/finetune.gin" \
        --gin.MODEL_DIR=\"${OUTPUT_DIR}\" \
        "$@" \
        --alsologtostderr 
        ;;
    eval)
        set -x;
        python -m t5x.eval \
        --gin_file="config/runs/t5_1_1/eval.gin" \
        --gin.EVAL_OUTPUT_DIR=\"${EVAL_OUTPUT_DIR}\" \
        --alsologtostderr \
        "$@"
        ;;
    infer)
        set -x;
        python -m t5x.infer \
        --gin_file="config/runs/t5_1_1/infer.gin" \
        --gin.INFER_OUTPUT_DIR=\"${OUTPUT_DIR}\" \
        --alsologtostderr \
        "$@"
        ;;
    *)
        echo "$ACTION may only be one of: pretrain, finetune, eval, infer"
        ;;
esac