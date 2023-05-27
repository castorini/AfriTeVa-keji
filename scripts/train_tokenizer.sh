#!/bin/sh
train () {
    TRAIN_DATA_PATH=$1
    VOCAB_SIZE=$2
    # OVERWRITE=$3

    OUTPUT_PATH="tokenizers/v$VOCAB_SIZE"

    [ -d $OUTPUT_PATH ] && echo "Tokenizer may already exist at $OUTPUT_PATH" && exit

    mkdir -p $OUTPUT_PATH

    echo "Learning subword units..."

    python scripts/python/train_sentencepiece.py \
    --input=$TRAIN_DATA_PATH \
    --model_prefix=$OUTPUT_PATH/sentencepiece.bpe \
    --vocab_size=$VOCAB_SIZE >& "logs/tokenizer/v${VOCAB_SIZE}_training.log"

    echo "Done! All files below saved in $OUTPUT_PATH"

    ls $OUTPUT_PATH
}

# while [ "$1" != "" ]; do
#     case $1 in
#         -f | --file )           shift
#                                 filename=$1
#                                 ;;
#         -i | --interactive )    interactive=1
#                                 ;;
#         -h | --help )           usage
#                                 exit
#                                 ;;
#         * )                     usage
#                                 exit 1
#     esac
#     shift
# done

train_data=$1 && shift
VOCAB_SIZES=( "$@" )

for vocab_size in ${VOCAB_SIZES[@]}
do
    train $train_data $vocab_size
done