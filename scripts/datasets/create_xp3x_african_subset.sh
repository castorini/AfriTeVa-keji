#!/bin/bash

LANGUAGE_PATHS_JSON_URL="https://huggingface.co/datasets/Muennighoff/xP3x/resolve/main/paths.json"
BASE_XP3X_URL="https://huggingface.co/datasets/CohereForAI/xP3x/resolve/main"

xP3x_LANGUAGE_CODES_WITH_SCRIPT=(
    "afr_Latn"
    "amh_Ethi" "aka_Latn"  "bam_Latn" "bem_Latn" "ewe_Latn"
    "fon_Latn" "gaz_Latn" "hau_Latn" "ibo_Latn" "kam_Latn" "kik_Latn"
    "kin_Latn" "kmb_Latn" "knc_Arab" "knc_Latn" "kon_Latn" "lin_Latn"
    "lug_Latn" "luo_Latn" "nso_Latn" "nya_Latn" "pcm_Latn" "plt_Latn"
    "run_Latn" "sna_Latn" "som_Latn" "sot_Latn" "ssw_Latn" "swh_Latn"
    "tir_Ethi" "tsn_Latn" "tum_Latn" "twi_Latn" "umb_Latn" "wol_Latn"
    "xho_Latn" "yor_Latn" "zul_Latn" 
    # Lingua Francas
    "aeb_Arab" "arb_Arab" "arb_Latn" "arq_Arab" "ary_Arab" "arz_Arab"
    "eng_Latn" "fra_Latn" "por_Latn"
)
xP3x_LANGUAGE_CODES=("${xP3x_LANGUAGE_CODES_WITH_SCRIPT[@]//_Arab/}")
xP3x_LANGUAGE_CODES=("${xP3x_LANGUAGE_CODES[@]//_Ethi/}")
xP3x_LANGUAGE_CODES=("${xP3x_LANGUAGE_CODES[@]//_Latn/}")
declare -A lang_codes_map
for code in "${xP3x_LANGUAGE_CODES[@]}"; do
    lang_codes_map["$code"]=1
done

mkdir -p data/xP3x

wget -O data/xP3x/paths.json "$LANGUAGE_PATHS_JSON_URL"

# Filter paths.json for African languages
language_json_filter=$(printf '%s\n' "${xP3x_LANGUAGE_CODES_WITH_SCRIPT[@]}" | jq -R . | jq -s .)
xP3x_LANGUAGE_PATHS_MAP=$(
    jq --argjson keys "$language_json_filter" '
    with_entries(select(.key as $k | $keys | index($k)))' data/xP3x/paths.json
)

filter_url() {
    # xP3x includes <src>_<tgt> type files.
    # This filter ensures that both src and tgt are languages of interest as listed in
    # xP3x_LANGUAGE_CODES
    local url="$1"
    modified_url=$(echo "$url" | sed -E 's/(_dev|_inverted_).*//')

    IFS='_' read -ra words <<< "$modified_url"  # Split the modified string into an array by underscores
    last_four_words=("${words[@]: -4}")         # Get the last four words <src>_<src_script>_<tgt>_<tgt_script>
    
    match_count=0
    for word in "${last_four_words[@]}"; do
        if [[ -n "${lang_codes_map[$word]}" ]]; then
            ((match_count++))
        fi
        # Break if we have found 2 matches
        if (( match_count >= 2 )); then
            break
        fi
    done
    
    # Print true if two or more matches found, otherwise false
    if (( match_count >= 2 )); then
        true
    else
        false
    fi
}

echo "$xP3x_LANGUAGE_PATHS_MAP" | jq -r 'to_entries[] | "\(.key) \(.value[])"' | while read -r key url; do
    url="$(echo "$url" | sed 's/?/%3F/g' | sed 's/,/%2C/g')"
    mkdir -p "data/xP3x/${key}"

    if filter_url "$url"; then
        wget -x -O "data/xP3x/${key}/${url/"data/${key}/"/}" "${BASE_XP3X_URL}/${url}"
        # echo "Downloading $url"
    else
        # echo "Skipping $url"
        :
    fi
done
