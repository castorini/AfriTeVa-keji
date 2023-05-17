{
    DATA_DIR="data/AwarawaV2Wiki"
    OUTPUT_DIR="data/AwarawaV2WikiPassages"

    TOKENIZER="tokenizers/v150000_new/sentencepiece.bpe.model"

    datasets=($(du -a $DATA_DIR | grep jsonl | grep -Ev 'eng|fra' | cut -f 2 ))
    datasets+=($(du -a $DATA_DIR | grep '1p5' | cut -f 2))

    for data in ${datasets[@]}
    do
        split=$(dirname $data)
        split=${split##*/}

        language=${data##*/}
        language=${language/'.jsonl'/''}

        python scripts/python/tokenize_data.py \
        --data-path $data \
        --language $language \
        --output-file "$OUTPUT_DIR/$split/$language.txt" \
        --chunk "sized-chunks" \
        --chunk-size 512 \
        --tokenizer-class "T5Tokenizer" \
        --tokenizer-name-or-path $TOKENIZER \
        >& "logs/tokenization/$language-$split.log" &
    done
}