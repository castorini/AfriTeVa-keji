import os
from functools import partial
from string import Template
from typing import Final

import seqio
import tensorflow as tf
from dotenv import load_dotenv
from t5.data.preprocessors import span_corruption, summarize
from t5.data.utils import rate_num_examples
from t5.evaluation.metrics import accuracy, bleu 

from teva.metrics import chrf, weighted_multiclass_f1
from teva.preprocessors import create_news_classification_example, jsonline_to_dict, line_to_dict, translate
from teva.utils import get_dataset_statistics, get_labels
from teva.vocab import DEFAULT_VOCAB

load_dotenv()

BUCKET_DIR=os.getenv("DATA_GCP_BUCKET_DIR", "data/")
STATISTICS_PATH = Template(BUCKET_DIR + "AwarawaV2${corpus}Passages/stats")
DATASET_PATH = Template(BUCKET_DIR + "AwarawaV2${corpus}Passages/${split}/${language}.txt")

DEFAULT_TEMPERATURE: Final = 1.0  # TODO: @theyorubayesian @ToluClassics
DEFAULT_MIX_RATE = partial(
    rate_num_examples, temperature=DEFAULT_TEMPERATURE
)

DEFAULT_OUTPUT_FEATURES: Final = {
    "inputs": seqio.Feature(vocabulary=DEFAULT_VOCAB, add_eos=True, required=False),
    "targets": seqio.Feature(vocabulary=DEFAULT_VOCAB, add_eos=True)
}

PRETRAINING_LANGUAGES: Final = [ 
    "afr", "amh", "arz", "eng_1p5", 
    "fra_1p5", "hau", "ibo", "kin", 
    "mlg", "nya", "orm", "por",
    "sna", "som", "sot", "swa",
    "tir", "xho", "yor", "zul"
]

CORPORA: Final = ["wiki", "news"] 
lm_tasks = []

for corpus in CORPORA:
    corpus_tasks = []

    dataset_statistics = get_dataset_statistics(STATISTICS_PATH.substitute(corpus=corpus.capitalize()))
    for lang in PRETRAINING_LANGUAGES:
        if corpus == "news" and lang in ["arz"]:
            continue

        lang_config_name = f"{lang}_{corpus}"

        seqio.TaskRegistry.add(
            lang_config_name,
            source=seqio.TextLineDataSource(
                split_to_filepattern={
                    "train": DATASET_PATH.substitute(
                        corpus=corpus.capitalize(),
                        split="train",
                        language=lang
                    ),
                    "validation": DATASET_PATH.substitute(
                        corpus=corpus.capitalize(), 
                        split="eval",
                        language=lang
                    )
                },
                num_input_examples=dataset_statistics[lang]
            ),
            preprocessors=[
                line_to_dict,
                seqio.preprocessors.tokenize,
                span_corruption,
                seqio.preprocessors.append_eos_after_trim
            ],
            output_features=DEFAULT_OUTPUT_FEATURES,
            metric_fns=[]
        )

        corpus_tasks.append(lang_config_name)
    
    seqio.MixtureRegistry.add(corpus, corpus_tasks, default_rate=DEFAULT_MIX_RATE)
    lm_tasks += corpus_tasks

seqio.MixtureRegistry.add("news_and_wiki", lm_tasks, default_rate=DEFAULT_MIX_RATE)

# ---------------------------------------------------------------------------------
# Evaluation Tasks
# ---------------------------------------------------------------------------------

# --------------------------
# MasakhaNews Classification
# --------------------------
MASAKHANEWS_DATASET_PATH = Template(BUCKET_DIR + "masakhanews/${language}/${split}.jsonl")
LABELS_PATH = Template(BUCKET_DIR + "masakhanews/${language}/labels.txt")

MASAKHANEWS_LANGUAGES: Final = [
    "amh", "eng", "fra", "hau",
    "ibo", "lin", "lug", "orm", 
    "pcm", "run", "sna", "som", 
    "swa", "tir", "xho", "yor"
]
masakhanews_dataset_statistics = get_dataset_statistics(f"{BUCKET_DIR}masakhanews/stats")
# TODO: @theyorubayesian
# This is a conversion from SplitStatistics to CorpusStatistics. Formalize it as a method.
masakhanews_dataset_statistics = {
    language: {
        split: masakhanews_dataset_statistics[split][language]
        for split in ["train", "validation", "test"]
    } for language in MASAKHANEWS_LANGUAGES
}

masakhanews_tasks = []

JSONLINE_SPECS = {
    field: tf.TensorSpec(tf.TensorShape([]), tf.string, name=field)
    for field in ["category", "headline", "text", "url"]
}
parse_jsonline = partial(jsonline_to_dict, specs=JSONLINE_SPECS)

for language in MASAKHANEWS_LANGUAGES:
    lang_config_name = f"{language}_masakhanews"
    labels = get_labels(LABELS_PATH.substitute(language=language))
    
    weighted_f1 = weighted_multiclass_f1(labels)

    seqio.TaskRegistry.add(
        name=lang_config_name,
        source=seqio.TextLineDataSource(
            split_to_filepattern={
                    "train": MASAKHANEWS_DATASET_PATH.substitute(
                        split="train",
                        language=language
                    ),
                    "validation": MASAKHANEWS_DATASET_PATH.substitute(
                        split="dev",
                        language=language
                    ),
                    "test": MASAKHANEWS_DATASET_PATH.substitute(
                        split="test",
                        language=language
                    )
                },
            num_input_examples=masakhanews_dataset_statistics[language]
        ),
        preprocessors=[
            parse_jsonline,
            create_news_classification_example,
            seqio.preprocessors.tokenize,
            seqio.preprocessors.append_eos_after_trim
        ],
        output_features=DEFAULT_OUTPUT_FEATURES,
        metric_fns=[accuracy, weighted_f1]
    )
    masakhanews_tasks.append(lang_config_name)

seqio.MixtureRegistry.add("masakhanews", masakhanews_tasks, default_rate=DEFAULT_MIX_RATE)

# -----------
# Translation
# -----------
LAFAND_DATASET_PATH = Template(BUCKET_DIR + "lafand/${pivot}-${language}/${split}.json")
EN_XX_TRANSLATION = os.getenv("LAFAND_TRANSLATION_DIRECTION", "en-xx") in ["en-xx", "fr-exx"]
LAFAND_FR_PIVOT_LANGUAGES = []
LAFAND_EN_PIVOT_LANGUAGES = [
    "hau", "pcm", "swa", "ibo", "yor", "zul"
]
LANGUAGE_CODE_MAP = {
    "hau": "Hausa", "pcm": "Pidgin", "swa": "Swahili", 
    "ibo": "Igbo", "yor": "Yoruba", "zul": "Zulu", 
    "en": "English", "fr": "French"
}
LAFAND_LANGUAGES = [*LAFAND_EN_PIVOT_LANGUAGES, *LAFAND_FR_PIVOT_LANGUAGES]
lafand_dataset_statistics = get_dataset_statistics(f"{BUCKET_DIR}lafand/stats")
lafand_dataset_statistics = {
    language: {
        split: lafand_dataset_statistics[split][language]
        for split in ["train", "validation", "test"]
        if language in lafand_dataset_statistics[split]     # Some languages do not have train
    } for language in LAFAND_LANGUAGES
}

lafand_tasks = []

for language in LAFAND_LANGUAGES:
    pivot = "en" if language in LAFAND_EN_PIVOT_LANGUAGES else "fr"
    task_name = f"{pivot}_{language}_lafand_mt" if EN_XX_TRANSLATION else f"{language}_{pivot}_lafand_mt"
    src_code = pivot if EN_XX_TRANSLATION else language
    tgt_code = language if EN_XX_TRANSLATION else pivot
    source_language = LANGUAGE_CODE_MAP[pivot] if EN_XX_TRANSLATION else language
    target_language = LANGUAGE_CODE_MAP[language] if EN_XX_TRANSLATION else LANGUAGE_CODE_MAP[pivot]
    prefix = f"Translate {source_language} to {target_language}: "

    JSONLINE_SPECS = {"translation": {
        language: tf.TensorSpec(tf.TensorShape([]), tf.string, name=language)
        for language in [pivot, language]
    }}
    parse_jsonline = partial(jsonline_to_dict, specs=JSONLINE_SPECS, return_key="translation")

    seqio.TaskRegistry.add(
        name=task_name,
        source=seqio.TextLineDataSource(
            split_to_filepattern={
                    "train": LAFAND_DATASET_PATH.substitute(
                        pivot=pivot,
                        split="train",
                        language=language
                    ),
                    "validation": LAFAND_DATASET_PATH.substitute(
                        pivot=pivot,
                        split="dev",
                        language=language
                    ),
                    "test": LAFAND_DATASET_PATH.substitute(
                        pivot=pivot,
                        split="test",
                        language=language
                    )
                },
            num_input_examples=lafand_dataset_statistics[language]
        ),
        preprocessors=[
            parse_jsonline,
            partial(translate, prefix=prefix, src_code=src_code, tgt_code=tgt_code),
            seqio.preprocessors.tokenize,
            seqio.preprocessors.append_eos_after_trim
        ],
        output_features=DEFAULT_OUTPUT_FEATURES,
        metric_fns=[accuracy, weighted_f1]
    )
    lafand_tasks.append(lang_config_name)

seqio.MixtureRegistry.add("lafand_mt", lafand_tasks, default_rate=DEFAULT_MIX_RATE)

# -------------------
# XLSUM Summarization
# -------------------
XLM_LANGUAGES = [
    "amharic", "english", "french", "hausa", 
    "igbo", "oromo", "pidgin", "portuguese",
    "somali", "swahili", "tigrinya", "yoruba", 
]

# xsum_dataset_statistics = get_dataset_statistics() # TODO: @theyorubayesian

for language in XLM_LANGUAGES:
    lang_config_name = f"{language.capitalize()}XSUM"

    # seqio.TaskRegistry.add(
    #     lang_config_name,
    #     source=seqio.TextLineDataSource(
            
    #     )
    # )