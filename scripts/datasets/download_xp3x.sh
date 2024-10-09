#!/bin/bash

DOWNLOAD_DIR="/projects/AfriTeVa-keji/data/xP3x"

BASE_XP3X_URL="https://huggingface.co/datasets/taresco/xP3x_african_subset/resolve/main"
xP3x_LANGUAGES=(
    "afr_Latn"
    "amh_Ethi" "aka_Latn"  "bam_Latn" "bem_Latn" "ewe_Latn"
    "fon_Latn" "gaz_Latn" "hau_Latn" "ibo_Latn" "kam_Latn" "kik_Latn"
    "kin_Latn" "kmb_Latn" "knc_Arab" "knc_Latn" "kon_Latn" "lin_Latn"
    "lug_Latn" "luo_Latn" "nso_Latn" "nya_Latn" "pcm_Latn" "plt_Latn"
    "run_Latn" "sna_Latn" "som_Latn" "sot_Latn" "ssw_Latn" "swh_Latn"
    "tir_Ethi" "tsn_Latn" "tum_Latn" "twi_Latn" "umb_Latn" "wol_Latn"
    "xho_Latn" "yor_Latn" "zul_Latn" 
    "aeb_Arab" "arb_Arab" "arb_Latn" "arq_Arab" "ary_Arab" "arz_Arab"
    "eng_Latn" "fra_Latn" "por_Latn"
)

mkdir -p $DOWNLOAD_DIR
XP3X_PATHS_URL="$BASE_XP3X_URL/paths.json"
wget --header="Authorization: Bearer $HUGGINGFACE_CLI_TOKEN" -O "$DOWNLOAD_DIR/paths.json" "$XP3X_PATHS_URL"

jq -r 'to_entries[] | "\(.key) \(.value[])"' "$DOWNLOAD_DIR/paths.json" | while read -r key url; do
    url="$(echo "$url" | sed 's/?/%3F/g' | sed 's/,/%2C/g')"
    mkdir -p "$DOWNLOAD_DIR/${key}"

    wget --header="Authorization: Bearer $HUGGINGFACE_CLI_TOKEN" \
    -x -O "$DOWNLOAD_DIR/${key}/${url/"data/${key}/"/}" "${BASE_XP3X_URL}/${url}"
done