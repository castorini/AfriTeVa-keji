{
    DATA_DIR="data/AwarawaV2Passages/train"
    ALPHA="0.3"
    SEED="3697"

    language_files=($(du -a $DATA_DIR | grep txt | grep -Ev 'ful|lin|eng|fra' | cut -f 2))
    language_files+=($(du -a $DATA_DIR | grep -E 'eng|fra' | grep '1p5' | cut -f 2))

    number_of_sentences=()
    
    for file in ${language_files[@]}
    do
        number_of_sentences+=($(cat $file | wc -l))
    done

    python scripts/python/sample_sentences.py \
    --input-files ${language_files[@]} \
    --n-sentences ${number_of_sentences[@]} \
    --output-file "$(dirname $DATA_DIR)/sampled_sentences_0p3_modified.txt" \
    --alpha $ALPHA \
    --seed $SEED >& logs/passage_sampling_modified.log
}