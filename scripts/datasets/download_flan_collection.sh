#!/bin/bash

{
    source scripts/datasets/utils.sh

    DOWNLOAD_DIR="$1"

    TARESCO_URL="https://huggingface.co/datasets/taresco/flan_subset/resolve/main/data/train-file_idx-of-total_files.parquet"

    declare -A SUBMIX_MAP=(
        [flan2021_submix_filtered]=11
        [niv2_submix_filtered]=15
        [cot_submix_filtered]=1
    )

    for submix in "${!SUBMIX_MAP[@]}"; do
        mkdir -p "$DOWNLOAD_DIR/$submix"

        total_files=${SUBMIX_MAP[$submix]}
        padded_total_files=$(printf "%05d" $total_files)

        URL=${TARESCO_URL/flan_subset/"$submix"}
        URL=${URL/total_files/"$padded_total_files"}

        for i in $(seq 0 $((total_files - 1))); do
            idx=$(printf "%05d" $i)
            _URL=${URL/file_idx/$idx}
            
            wget -P "$DOWNLOAD_DIR/$submix" $_URL
        done

        for file in $DOWNLOAD_DIR/$submix/*.parquet; do
            convert_parquet_to_jsonl $file && rm $file
        done
    done

    DPI_URL="https://huggingface.co/datasets/DataProvenanceInitiative/flan_subset/resolve/main/data"

    T0_SUBMIX_DIR="$DOWNLOAD_DIR/t0_submix"
    DIALOG_SUBMIX_DIR="$DOWNLOAD_DIR/dialog_submix"
    mkdir -p  $T0_SUBMIX_DIR $DIALOG_SUBMIX_DIR

    TO_SUBMIX_URL="${DPI_URL/flan_subset/t0_submix_original}/train-00000-of-00001-0a6693a25fc4a25e.parquet"
    wget -P "$T0_SUBMIX_DIR" "$TO_SUBMIX_URL" && convert_parquet_to_jsonl $T0_SUBMIX_DIR/*.parquet && rm $T0_SUBMIX_DIR/*.parquet

    DIALOG_SUBMIX_URL="${DPI_URL/flan_subset/dialog_submix_original}/train-00000-of-00001-0aecb489ddece98b.parquet"
    wget -P "$DIALOG_SUBMIX_DIR" "$DIALOG_SUBMIX_URL" && convert_parquet_to_jsonl $DIALOG_SUBMIX_DIR/*.parquet && rm $DIALOG_SUBMIX_DIR/*.parquet
}