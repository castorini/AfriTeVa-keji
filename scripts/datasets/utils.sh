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
