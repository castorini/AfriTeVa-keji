#!/bin/bash

# Base directory
base_dir="/store/aya"

# Output file
output_file="/store/aya/statistics.json"

find "$base_dir" -type f -name "*.jsonl" | while read -r file; do
    rel_path=${file#$base_dir/}
    IFS="/" read -r task split lang_file <<< "$rel_path"
    language="${lang_file%.jsonl}"
    
    # Count lines in the file
    line_count=$(wc -l < "$file")
    
    # Add the count to stats
    echo -e "$task\t$split\t$language\t$line_count" >> $output_file
done

echo "Statistics saved to $output_file"
