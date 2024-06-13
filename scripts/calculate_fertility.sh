calc_fertility() {
    # set -x;
    VOCAB_SIZE=$1

    LANGUAGES=("hau" "ibo" "kin" "nya" "sna" "swa" "xho" "yor" "zul")

    POS_DATA_DIR="/home/aooladip/projects/AfriTeVa-keji/data/masakhane_pos"
    SPLIT="test"

    TOKENIZER_DIR="tokenizers/$1/sentencepiece.bpe.model"
    TOKENIZER_CLS="T5Tokenizer"

    OUTPUT_DIR=$(dirname $TOKENIZER_DIR)/stats

    for language in ${LANGUAGES[@]}
    do
        mkdir -p $OUTPUT_DIR/$language

        python scripts/python/fertility.py \
        --input-file $POS_DATA_DIR/$language/$SPLIT.txt \
        --multi-tokenizer-cls $TOKENIZER_CLS \
        --multi-tokenizer-name-or-path $TOKENIZER_DIR \
        --output-dir $OUTPUT_DIR/$language

    done
}

VOCAB_SIZES=("v100000" "v150000" "v200000" "v250000")

for vocab_size in ${VOCAB_SIZES[@]}
do
    calc_fertility $vocab_size
done