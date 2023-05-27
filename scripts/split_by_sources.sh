DATA_DIR="data/AwarawaV2"
WIKI_OUTPUT_DIR="data/AwarawaV2Wiki"
NOT_WIKI_OUTPUT_DIR="data/AwarawaV2NotWiki"

datasets=($(du -a $DATA_DIR | grep jsonl | grep -Ev 'eng|fra' | cut -f 2 ))
datasets+=($(du -a $DATA_DIR | grep '1p5' | cut -f 2))

declare -A LANGUAGES=(
    ["swa"]=sw
    ["afr"]=af
    ["arz"]=arz
    ["por"]=pt
    ["hau"]=ha
    ["ibo"]=ig
    ["som"]=so
    ["nya"]=ny
    ["zul"]=zu
    ["xho"]=xh
    ["sna"]=sn
    ["mlg"]=mg
    ["sot"]=st
    ["amh"]=am
    ["orm"]=om
    ["eng"]=en
    ["fra"]=fr
    ["tir"]=ti
    ["kin"]=rw
    ["yor"]=yo
)

for data in ${datasets[@]}
do
    split=$(dirname $data)
    split=${split##*/}

    language=${data##*/}
    language=${language/'.jsonl'/''}

    wiki_link="${LANGUAGES[$language]}.wikipedia.org"

    cat $data | grep $wiki_link > $WIKI_OUTPUT_DIR/$split/$language.jsonl
    cat $data | grep -v $wiki_link > $NOT_WIKI_OUTPUT_DIR/$split/$language.jsonl
done