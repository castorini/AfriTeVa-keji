#!/bin/bash
source scripts/datasets/utils.sh

download_octopack_osst(){
    set -e;
    local DOWNLOAD_DIR="$1"
    mkdir -p "$DOWNLOAD_DIR"
    wget -P "$DOWNLOAD_DIR" https://huggingface.co/datasets/bigcode/oasst-octopack/resolve/main/oasst_octopack.jsonl

    for lang in "en" "fr" "pt-BR"; do
        jq -c "select(.lang | IN(\"${lang}\"))" "$DOWNLOAD_DIR/oasst_octopack.jsonl" > "$DOWNLOAD_DIR/oasst_octopack_${lang}.jsonl"
    done
    
    rm "$DOWNLOAD_DIR/oasst_octopack.jsonl"
}

download_oig_small_chip2(){
    local DOWNLOAD_DIR="$1"
    mkdir -p "$DOWNLOAD_DIR"
    wget -P "$DOWNLOAD_DIR" "https://huggingface.co/datasets/0-hero/OIG-small-chip2/resolve/main/data/train-00000-of-00001-34df73631d1c6428.parquet" && \
        convert_parquet_to_jsonl $DOWNLOAD_DIR/train-00000-of-00001-34df73631d1c6428.parquet && \
        rm $DOWNLOAD_DIR/train-00000-of-00001-34df73631d1c6428.parquet
}

# Needs to be filtered for tasks that overlap evaluation data
# Or evaluation task categories - textual entailment, co-reference resolution and
# setence comparison tasks
download_tasksource_instruct(){
    local DOWNLOAD_DIR="$1"
    mkdir -p "$DOWNLOAD_DIR"
    git clone https://huggingface.co/datasets/tasksource/tasksource-instruct-v0 $DOWNLOAD_DIR-temp && \
        cd $DOWNLOAD_DIR-temp && git lfs pull && cd .. && \
        mv "$DOWNLOAD_DIR"-temp/data/* "$DOWNLOAD_DIR" &&
        rm -rf $DOWNLOAD_DIR-temp

    for split in $DOWNLOAD_DIR/*; do
        convert_parquet_to_jsonl $split && rm $split
    done
}

# A subset of the Flan collection 
# No code datasets included.
# Sample a maximum of 20K for each of these sources.

# ShareGPT - Not Included