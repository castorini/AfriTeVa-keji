#!/bin/bash
{
    set -e;
    
    MODEL_SIZE="large"

    LANGUAGES=("amh" "eng" "fra" "hau" "ibo" "lin" "lug" "orm" "pcm" "run" "sna" "som" "swa" "tir" "xho" "yor")

    for language in "${LANGUAGES[@]}" 
    do
        python -m teva.torch.classification.collate_results \
        --results-dir "runs/classification/afriteva_v2_${MODEL_SIZE}" \
        --language $language \
        --n-seeds 5 \
        --output-file "runs/classification/results/afriteva_v2_${MODEL_SIZE}/${language}.json"
    done
}