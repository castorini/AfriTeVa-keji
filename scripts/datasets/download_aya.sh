#!/bin/bash
############################################################################################################
# REMEMDER TO SET HUGGINGFACE_CLI_TOKEN  AS ENVIRONMENT VARIABLE BEFORE RUNNING THIS SCRIPT
############################################################################################################

DOWNLOAD_DIR="/store/aya"
DOWNLOAD_CONFIG="all"   # TODO: @theyorubayesian Allow subset downlaod

AYA_TARESCO_URL="https://huggingface.co/datasets/taresco/aya_african_subset/resolve/main/data/dataset_name/dataset_split/language.jsonl"
AYA_DATASET_LANGUAGES=(
    "afrikaans" "algerian_arabic" "amharic"
    "egyptian_arabic" "english" "french"
    "hausa" "igbo" "kinyarwanda"
    "moroccan_arabic" "mozambican_portuguese" "nyanja"
    "plateau_malagasy" "portuguese" "shona"
    "somali" "swahili" "tunisian_arabic"
    "shona" "southern_sotho" "xhosa"
    "yoruba" "zulu" "bemba"
    "central_kanuri" "fon" "twi" "wolof"
)
HUMAN_ANNOTATED=("aya-dataset")
TEMPLATED_DATASETS=(
    "afriqa-inst" "afrisenti-inst" "amharic_qa"
    "joke-explaination-inst" "masakhanews-inst"
    "mintaka-inst" "ntx-llm-inst" "nusax-senti-inst"
    "scirepeval-biomimicry-inst" "soda-inst" "uner-llm-inst"
    "wiki-split-inst" "xlel_wd-inst" "xwikis-inst"
)
TRANSLATED_DATASETS=(
    "adversarial_qa_(t)" "cnn-daily-mail_(t)" "dolly-v2_(t)"
    "flan-coqa_(t)" "flan-cot-submix_(t)" "flan-gem-wiki-lingua_(t)"
    "flan-lambada_(t)" "flan-unified-qa_(t)" "hotpotqa_(t)"
    "joke-explaination-inst_(t)" "mintaka-inst_(t)" "mlqa-en_(t)"
    "nq-open_(t)" "paws-wiki_(t)" "piqa_(t)" "soda-inst_(t)"
    "wiki_qa_(t)" "wiki-split-inst_(t)" "xlel_wd-inst_(t)"
)

if [[ $DOWNLOAD_CONFIG == "all" ]]; then
    DATASETS=("${HUMAN_ANNOTATED[@]}" "${TEMPLATED_DATASETS[@]}" "${TRANSLATED_DATASETS[@]}")
fi
# elif [[ $DOWNLOAD_CONFIG == "human" ]]; then
#     DATASETS=("${HUMAN_ANNOTATED[@]}")
# elif [[ ]]
#     DATASETS=("${HUMAN_ANNOTATED[@]}"

for dataset in "${DATASETS[@]}"; do
    dataset_dir="$DOWNLOAD_DIR/$dataset"
    dataset_url="${AYA_TARESCO_URL/dataset_name/"$dataset"}"

    for language in "${AYA_DATASET_LANGUAGES[@]}"; do

        language_url="${dataset_url/language/"$language"}"

        for split in "train" "validation" "test"; do
            split_dir="$dataset_dir/$split"
            mkdir -p "$split_dir"

            language_file_path="$split_dir/$language.jsonl"

            if [ -f $language_file_path ]; then
                echo "$dataset:$language/$split already exists"
            else
                URL="${language_url/dataset_split/"$split"}"

                # check if url exists
                if wget --header="Authorization: Bearer $HUGGINGFACE_CLI_TOKEN" --spider "${URL}" 2>/dev/null; then
                    echo "Downloading $dataset:$language/$split ......"
                    wget  --header="Authorization: Bearer $HUGGINGFACE_CLI_TOKEN"  -O $language_file_path $URL
                else
                    echo " Online URL ${URL} not found !"
                fi
            fi
        done
    done
done