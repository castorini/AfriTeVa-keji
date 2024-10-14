#!/bin/bash

base_dir="data/xP3x"
output_file="data/xP3x/statistics.json"

json_output="{}"

for lang_dir in $base_dir/*; do
    lang=${lang_dir##*/}

    if [ $lang = "paths.json" ]; then
        continue
    fi

    lines=$(cat $lang_dir/*.jsonl | wc -l)

    json_output=$(echo "$json_output" | jq --arg lang "$lang" --argjson lines "$lines" '. + {($lang): $lines}')
done

echo "$json_output" > "$output_file"