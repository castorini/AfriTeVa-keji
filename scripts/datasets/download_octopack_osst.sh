#!/bin/bash

convert_parquet_to_jsonl() {
    parquet_file=$1
    json_file="${parquet_file/parquet/json}"

    if [ -f "$json_file" ]; then
        echo "JSON file already exists"
        return 0
    fi

    conversion_code="
import pyarrow.parquet as pq
parquet_file = pq.ParquetFile('$parquet_file')

for batch in parquet_file.iter_batches():
    batch.to_pandas().to_json(
        '$json_file', orient='records', lines=True, force_ascii=False, mode='a')
"

    python -c "$conversion_code" && \
    echo "Conversion completed: $json_file"
}

download_octopack_osst(){
    DOWNLOAD_DIR="/projects/AfriTeVa-keji/data/open-assistant"
    local DOWNLOAD_DIR="$1"
    mkdir -p "$DOWNLOAD_DIR"
    wget -P "$DOWNLOAD_DIR"  https://huggingface.co/datasets/bigcode/oasst-octopack/resolve/main/oasst_octopack.jsonl
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
    # git clone https://huggingface.co/datasets/tasksource/tasksource-instruct-v0 $DOWNLOAD_DIR-temp && \
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