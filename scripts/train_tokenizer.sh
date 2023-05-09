#!/bin/sh
{
    TRAIN_DATA_PATH=$1
    VOCAB_SIZE=$2

    OUTPUT_PATH="tokenizers/v$VOCAB_SIZE"

    [ -d $OUTPUT_PATH ] && echo "Tokenizer may already exist at $OUTPUT_PATH" && exit

    mkdir -p $OUTPUT_PATH

    echo "Learning subword units..."

    python scripts/python/train_sentencepiece.py \
    --input=$TRAIN_DATA_PATH \
    --model_prefix=$OUTPUT_PATH/sentencepiece.bpe \
    --vocab_size=$VOCAB_SIZE

    echo "Done! All files below saved in $OUTPUT_PATH"

    ls $OUTPUT_PATH
}