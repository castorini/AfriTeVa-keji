import os
from collections import OrderedDict
from functools import partial
from string import Template
from typing import Final
from typing import List

import seqio
import tensorflow as tf
from dotenv import load_dotenv
from t5.data.preprocessors import span_corruption, summarize
from t5.data.utils import rate_num_examples
from t5.evaluation.metrics import accuracy, bleu, rouge, squad as squad_metrics

from teva.metrics import chrf, weighted_multiclass_f1
from teva.preprocessors import (
    afriqa,
    create_news_classification_example, 
    jsonline_to_dict, 
    line_to_dict,
    squad,
    take_subset,
    translate
)
from teva.postprocessors import squad_postprocessor
from teva.utils import get_dataset_statistics, get_labels
from teva.vocab import DEFAULT_VOCAB

load_dotenv()

BUCKET_DIR=os.getenv("DATA_GCP_BUCKET_DIR", "data/")
if BUCKET_DIR.endswith("/"):
    BUCKET_DIR += "/"

DEFAULT_TEMPERATURE: Final = 1.0  # TODO: @theyorubayesian @ToluClassics
DEFAULT_MIX_RATE = partial(
    rate_num_examples, temperature=DEFAULT_TEMPERATURE
)

DEFAULT_OUTPUT_FEATURES: Final = {
    "inputs": seqio.Feature(vocabulary=DEFAULT_VOCAB, add_eos=True, required=False),
    "targets": seqio.Feature(vocabulary=DEFAULT_VOCAB, add_eos=True)
}


class TaskNotFoundException(Exception):
    message_template = Template("${task} not found in list of tasks (${tasks})")

    def __init__(self, task: str, tasks: List[str]) -> None:
        self.message = self.message_template.substitute(task=task, tasks=" , ".join(tasks))


def add_pretraining_task():
    STATISTICS_PATH = Template(BUCKET_DIR + "AwarawaV2${corpus}Passages/stats")
    DATASET_PATH = Template(BUCKET_DIR + "AwarawaV2${corpus}Passages/${split}/${language}.txt")
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
def add_masakhanews_task():
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

    MASAKHANEWS_JSONLINE_SPECS = {
        field: tf.TensorSpec(tf.TensorShape([]), tf.string, name=field)
        for field in ["category", "headline", "text", "url"]
    }
    parse_masakhanews_jsonline = partial(jsonline_to_dict, specs=MASAKHANEWS_JSONLINE_SPECS)

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
                parse_masakhanews_jsonline,
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
# For inference: beam search - size: 5, length penalty: 0.6
def add_lafand_task():
    LAFAND_DATASET_PATH = Template(BUCKET_DIR + "lafand/${pivot}-${language}/${split}.json")
    LAFAND_FR_PIVOT_LANGUAGES = []
    LAFAND_EN_PIVOT_LANGUAGES = [
        "hau", "pcm", "swa", "ibo", "yor", "zul", "tsn", "twi", # TODO: Include xho, zul
    ]
    LANGUAGE_CODE_MAP = {
        "hau": "Hausa", "pcm": "Pidgin", "swa": "Swahili", 
        "ibo": "Igbo", "yor": "Yoruba", "zul": "Zulu", 
        "en": "English", "fr": "French", "twi": "Twi", "tsn": "Tswana"
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

    lafand_en_xx_tasks = []
    lafand_xx_en_tasks = []

    for language in LAFAND_LANGUAGES:
        pivot = "en" if language in LAFAND_EN_PIVOT_LANGUAGES else "fr"

        lafand_en_xx_task_name = f"{pivot}_{language}_lafand_mt"
        en_xx_prefix = f"Translate {LANGUAGE_CODE_MAP[pivot]} to {LANGUAGE_CODE_MAP[language]}: "

        lafand_xx_en_task_name = f"{language}_{pivot}_lafand_mt"
        xx_en_prefix = f"Translate {LANGUAGE_CODE_MAP[language]} to {LANGUAGE_CODE_MAP[pivot]}: "

        JSONLINE_SPECS = {"translation": {
            _language: tf.TensorSpec(tf.TensorShape([]), tf.string, name=_language)
            for _language in [pivot, language]
        }}
        parse_lafand_jsonline = partial(jsonline_to_dict, specs=JSONLINE_SPECS, return_key="translation")
        
        lafand_source = seqio.TextLineDataSource(
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
        )

        seqio.TaskRegistry.add(
            name=lafand_en_xx_task_name,
            source=lafand_source,
            preprocessors=[
                parse_lafand_jsonline,
                partial(translate, prefix=en_xx_prefix, src_code=pivot, tgt_code=language),
                seqio.preprocessors.tokenize,
                seqio.preprocessors.append_eos_after_trim
            ],
            output_features=DEFAULT_OUTPUT_FEATURES,
            metric_fns=[bleu, chrf]
        )
        lafand_en_xx_tasks.append(lafand_en_xx_task_name)

        seqio.TaskRegistry.add(
            name=lafand_xx_en_task_name,
            source=lafand_source,
            preprocessors=[
                parse_lafand_jsonline,
                partial(translate, prefix=xx_en_prefix, src_code=language, tgt_code=pivot),
                seqio.preprocessors.tokenize,
                seqio.preprocessors.append_eos_after_trim
            ],
            output_features=DEFAULT_OUTPUT_FEATURES,
            metric_fns=[bleu, chrf]
        )
        lafand_xx_en_tasks.append(lafand_xx_en_task_name)

    seqio.MixtureRegistry.add("lafand_mt_en_xx", lafand_en_xx_tasks, default_rate=DEFAULT_MIX_RATE)
    seqio.MixtureRegistry.add("lafand_mt_xx_en", lafand_xx_en_tasks, default_rate=DEFAULT_MIX_RATE)
    seqio.MixtureRegistry.add("lafand_mt", [*lafand_xx_en_tasks, *lafand_en_xx_tasks], default_rate=DEFAULT_MIX_RATE)

# -------------------
# XLSUM Summarization
# -------------------
# For inference: beam search - size: 4, length penalty: 0.6
# Batch size: 256
def add_xlsum_task():
    XLSUM_DATASET_PATH = Template(BUCKET_DIR + "xlsum/${language}/${split}.json")

    XLSUM_LANGUAGES = [
        "amharic", "arabic", "english", "french", "hausa",
        "igbo", "kirundi", "oromo", "pidgin", "portuguese",
        "somali", "swahili", "tigrinya", "yoruba",
    ]
    xlsum_dataset_statistics = get_dataset_statistics(f"{BUCKET_DIR}xlsum/stats")
    xlsum_dataset_statistics = {
        language: {
            split: xlsum_dataset_statistics[split][language]
            for split in ["train", "validation", "test"]
        } for language in XLSUM_LANGUAGES
    }

    XLSUM_JSONLINE_SPECS = {
        field: tf.TensorSpec(tf.TensorShape([]), tf.string, name=field)
        for field in ["id", "url", "title", "summary", "text"]
    }
    parse_xlsum_jsonline = partial(jsonline_to_dict, specs=XLSUM_JSONLINE_SPECS)

    xlsum_tasks = []

    for language in XLSUM_LANGUAGES:
        xlsum_task_name = f"{language}_xlsum"

        seqio.TaskRegistry.add(
            xlsum_task_name,
            source=seqio.TextLineDataSource(
                split_to_filepattern={
                        "train": XLSUM_DATASET_PATH.substitute(
                            split="train",
                            language=language
                        ),
                        "validation": XLSUM_DATASET_PATH.substitute(
                            split="validation",
                            language=language
                        ),
                        "test": XLSUM_DATASET_PATH.substitute(
                            split="test",
                            language=language
                        )
                    },
                num_input_examples=xlsum_dataset_statistics[language]
            ),
            preprocessors=[
                parse_xlsum_jsonline,
                partial(summarize, article_key="text", summary_key="summary"),
                seqio.preprocessors.tokenize,
                seqio.preprocessors.append_eos_after_trim
            ],
            output_features=DEFAULT_OUTPUT_FEATURES,
            metric_fns=[rouge]
        )
        xlsum_tasks.append(xlsum_task_name)

    seqio.MixtureRegistry.add("xlsum", xlsum_tasks, default_rate=DEFAULT_MIX_RATE)

# -------
# SQuADV2
# -------
def add_squad_task():
    SQUAD_DATASET_PATH = Template(BUCKET_DIR + "squad_v2/${split}.jsonl")
    squad_dataset_statistics = {
        split: value["squad_v2"]
        for split, value in get_dataset_statistics(f"{BUCKET_DIR}squad_v2/stats").items()
    }

    ANSWER_SPEC = {"answers": {
        "text": tf.TensorSpec(tf.TensorShape([None]), tf.string, name="text"),
        "answer_start": tf.TensorSpec(tf.TensorShape([None]), tf.int32, name="answer_start"),
    }}
    OTHER_SPECS = {
        field: tf.TensorSpec(tf.TensorShape([]), tf.string, name=field)
        for field in ["id", "title", "context", "question"]
    }
    SQUAD_SPECS = {**OTHER_SPECS, **ANSWER_SPEC}
    parse_squad_jsonline = partial(jsonline_to_dict, specs=SQUAD_SPECS)

    seqio.TaskRegistry.add(
        name="squad_v2",
        source=seqio.TextLineDataSource(
            split_to_filepattern={
                "train": SQUAD_DATASET_PATH.substitute(split="train"),
                "validation": SQUAD_DATASET_PATH.substitute(split="val"),
            },
            num_input_examples=squad_dataset_statistics
        ),
        preprocessors=[
            parse_squad_jsonline,
            squad,
            seqio.preprocessors.tokenize,
            seqio.preprocessors.append_eos_after_trim
        ],
        postprocess_fn=squad_postprocessor,
        output_features=DEFAULT_OUTPUT_FEATURES,
        metric_fns=[squad_metrics]
    )

# -------
# AfriQA
# -------
def add_afriqa_task():
    AFRIQA_DATASET_PATH = Template(BUCKET_DIR + "afriqa/gold_passages/${language}/gold_span_passages.afriqa.${language}.${pivot}.${split}.json")
    EN_PIVOT_LANGUAGES = ["bem", "hau", "ibo", "kin", "twi", "zul"]
    FR_PIVOT_LANGUAGES = ["fon"]
    LANGUAGES = [*EN_PIVOT_LANGUAGES, *FR_PIVOT_LANGUAGES]

    afriqa_dataset_statistics = get_dataset_statistics(BUCKET_DIR + "afriqa/gold_passages/stats")
    afriqa_dataset_statistics = {
        language: {
            split: afriqa_dataset_statistics[split][language]
            for split in ["train", "validation", "test"]
        } for language in LANGUAGES
    }

    ANSWER_SPEC = {"answer_pivot": {
        "answer_start": tf.TensorSpec([None], tf.int32, name="answer_start"),
        "text": tf.TensorSpec([None], tf.string, name="text")
    }}
    OTHER_SPECS = {
        field: tf.TensorSpec([], tf.string, name=field) 
        for field in [
            "context", "id", "question_lang", 
            "question_translated", "title", "answer_lang"
        ]
    }
    AFRIQA_SPEC = {**OTHER_SPECS, **ANSWER_SPEC}
    parse_afriqa_jsonline = partial(jsonline_to_dict, specs=AFRIQA_SPEC)

    afriqa_tasks = []
    for language in LANGUAGES:
        task_name = f"{language}_afriqa"
        pivot = "fr" if language in FR_PIVOT_LANGUAGES else "en"

        seqio.TaskRegistry.add(
            name=task_name,
            source=seqio.TextLineDataSource(
                split_to_filepattern={
                    "train": AFRIQA_DATASET_PATH.substitute(split="train", language=language, pivot=pivot),
                    "validation": AFRIQA_DATASET_PATH.substitute(split="dev", language=language, pivot=pivot),
                    "test": AFRIQA_DATASET_PATH.substitute(split="test", language=language, pivot=pivot),
                },
                num_input_examples=afriqa_dataset_statistics[language]
            ),
            preprocessors=[
                parse_afriqa_jsonline,
                afriqa,
                seqio.preprocessors.tokenize,
                seqio.preprocessors.append_eos_after_trim
            ],
            postprocess_fn=squad_postprocessor,
            output_features=DEFAULT_OUTPUT_FEATURES,
            metric_fns=[squad_metrics]
        )
        afriqa_tasks.append(task_name)
    
    seqio.MixtureRegistry.add("afriqa", afriqa_tasks, default_rate=DEFAULT_MIX_RATE)

# ---
# Aya
# ---
def add_aya_task():
    DATASET_PATH = Template(BUCKET_DIR + "aya/${split}/${language}.jsonl")
    AYA_DATASET_LANGUAGES = [
        "afrikaans", "algerian_arabic", "amharic", "egyptian_arabic", "english",
        "french", "hausa", "igbo", "kinyarwanda", "mozambican_portuguese",
        "nyanja", "plateau_malagasy", "portuguese", "shona", "somali", "swahili",
        "shona", "southern_sotho", "xhosa", "yoruba", "zulu",
        # "moroccan_arabic", "tunisian_arabic",                 # These are alt arabic forms we could support
        # "bemba", "central_kanuri", "fon", "twi", "wolof"      # These are African languages not in WURA
    ]

    AYA_DATASET_STATISTICS = get_dataset_statistics(BUCKET_DIR + "aya/statistics.jsonl")
    
    TEXT_SPEC = {
        field: tf.TensorSpec([], tf.string, name=field) 
        for field in ['inputs', 'targets', 'dataset_name', 'sub_dataset_name', 'task_type', 'language', 'script', 'split']
    }

    ID_SPEC = {
        field: tf.TensorSpec([None], tf.int32, name=field)
        for field in ["id", "template_id"]
    }

    AYA_SPEC = {**TEXT_SPEC, **ID_SPEC}

    parse_aya_jsonline = partial(jsonline_to_dict, specs=AYA_SPEC)

    aya_tasks = []
    for language in AYA_DATASET_LANGUAGES:
        task_name = f"{language}_aya"

        seqio.TaskRegistry.add(
            name=task_name,
            source=seqio.TextLineDataSource(
                split_to_filepattern={
                    "train": DATASET_PATH.substitute(split="train", language=language),
                    "validation": DATASET_PATH.substitute(split="validation", language=language),
                    "test": DATASET_PATH.substiture(split="test", language=language)
                },
                num_input_examples=AYA_DATASET_STATISTICS[language]
            ),
            preprocessors=[
                parse_aya_jsonline,
                partial(take_subset, keys=["inputs", "targets"]),
                seqio.preprocessors.tokenize,
                seqio.preprocessors.append_eos_after_trim
            ],
            output_features=DEFAULT_OUTPUT_FEATURES,
            metric_fns=[]
        )

        aya_tasks.append(task_name)

    seqio.MixtureRegistry.add("aya", aya_tasks, default_rate=DEFAULT_MIX_RATE)

# ------
# MAIN
# ------
# TODO: Allow definition of task mixtures through YAML.
task_factory = {
    "news_and_wiki": add_pretraining_task,
    "masakhanews": add_masakhanews_task,
    "lafand_mt": add_lafand_task,
    "xlsum": add_xlsum_task,
    "squad_v2": add_squad_task,
    "afriqa": add_afriqa_task,
    "aya": add_aya_task,
}
task_factory = OrderedDict(sorted(task_factory.items()))

tasks = os.getenv("TASKS_TO_LOAD", "all")

if tasks == "all":
    """
    When finetuning ArawaT5, we finetune on SQuAD & zero-shot to AfriQA gold passages
    Here we create two tasks: one with AfriQA included (used for evaluation) 
    & another w/o AfriQA included used for multitask finetuning
    """
    arawa_multitask_ft_tasks = []
    eval_tasks = []
    for task, func in task_factory.items():
        func()
        
        if task != "news_and_wiki":
            eval_tasks.append(task)

            if task != "afriqa":
                arawa_multitask_ft_tasks.append(task)
    
    seqio.MixtureRegistry.add("_".join(arawa_multitask_ft_tasks), arawa_multitask_ft_tasks, default_rate=DEFAULT_MIX_RATE)
    seqio.MixtureRegistry.add("_".join(eval_tasks), eval_tasks, default_rate=DEFAULT_MIX_RATE)
else:
    tasks = sorted(tasks.split(","))

    eval_tasks = []
    for task in tasks:
        if task in task_factory:
            task_factory[task]()

            if task != "news_and_wiki":
                eval_tasks.append(task)
        else:
            raise TaskNotFoundException(task=task, tasks=list(task_factory.keys()))
    
    if len(eval_tasks) > 1:
        seqio.MixtureRegistry.add("_".join(eval_tasks), eval_tasks, default_rate=DEFAULT_MIX_RATE)
