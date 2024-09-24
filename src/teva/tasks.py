import os
import re
from collections import OrderedDict
from dataclasses import asdict, dataclass
from functools import partial
from string import Template
from typing import Final, List, Optional

import gin
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
if not BUCKET_DIR.endswith("/"):
    BUCKET_DIR += "/"

DEFAULT_OUTPUT_FEATURES: Final = {
    "inputs": seqio.Feature(vocabulary=DEFAULT_VOCAB, add_eos=True, required=False),
    "targets": seqio.Feature(vocabulary=DEFAULT_VOCAB, add_eos=True)
}

pattern = re.compile(r'[^\w\d\.\:_#()]')

# Function to normalize a task name
normalize = lambda name: pattern.sub('_', name).replace("(", "").replace(")", "")


class TaskNotFoundException(Exception):
    message_template = Template("${task} not found in list of tasks (${tasks})")

    def __init__(self, task: str, tasks: List[str]) -> None:
        self.message = self.message_template.substitute(task=task, tasks=" , ".join(tasks))


@gin.register
@dataclass
class MixtureRateConfig:
    scale: float = 1.0
    temperature: float = 1.0
    maximum: Optional[float] = None


def get_rate(
    scale: float = 1.0,
    temperature: float = 1.0,
    maximum: int | float | None  = None,
) -> callable:
    return partial(
        rate_num_examples,
        scale=scale,
        temperature = temperature,
        maximum=maximum
    )

# TODO: Should this consider the rate for each task?
def rate_num_examples_for_mixtures(
    task: seqio.Mixture,
    maximum: Optional[int] = None,
    scale: float = 1.0,
    temperature: float = 1.0,
    fallback_to_num_input_examples: bool = True,
    split: str = "train",
) -> float:
    ret = 0

    for t in task.tasks:
        try:
            if t.cache_dir or not fallback_to_num_input_examples:
                ret += t.get_cached_stats(split)["examples"]
            else:
                ret += t.num_input_examples(split)
        except (ValueError, KeyError):
            # Some tasks may not have a train split
            continue

    ret *= scale
    if maximum:
        if isinstance(maximum, float):
            maximum *= ret
        ret = min(ret, maximum)
    if temperature != 1.0:
        ret = ret ** (1.0 / temperature)
    return ret


def get_mixture_rate(
    num_examples: int,
    scale: float = 1.0,
    temperature: float = 1.0,
    maximum: int | float | None  = None
) -> float:
    num_examples *= scale
    if maximum:
        num_examples = min(num_examples, maximum)
    if temperature != 1.0:
        num_examples = num_examples ** (1.0 / temperature)
    
    return num_examples


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
        
        seqio.MixtureRegistry.add(corpus, corpus_tasks, default_rate=rate_num_examples)
        lm_tasks += corpus_tasks

    seqio.MixtureRegistry.add("news_and_wiki", lm_tasks, default_rate=rate_num_examples)

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

    seqio.MixtureRegistry.add("masakhanews", masakhanews_tasks, default_rate=rate_num_examples)

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

    seqio.MixtureRegistry.add("lafand_mt_en_xx", lafand_en_xx_tasks, default_rate=rate_num_examples)
    seqio.MixtureRegistry.add("lafand_mt_xx_en", lafand_xx_en_tasks, default_rate=rate_num_examples)
    seqio.MixtureRegistry.add("lafand_mt", [*lafand_xx_en_tasks, *lafand_en_xx_tasks], default_rate=rate_num_examples)

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

    seqio.MixtureRegistry.add("xlsum", xlsum_tasks, default_rate=rate_num_examples)

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
    
    seqio.MixtureRegistry.add("afriqa", afriqa_tasks, default_rate=rate_num_examples)

# ---
# Aya
# ---
AYA_DATASET_LANGUAGES = [
    "afrikaans", "amharic", "egyptian_arabic", "english",
    "french", "hausa", "igbo", "kinyarwanda", "nyanja",
    "plateau_malagasy", "portuguese", "shona", "somali",
    "swahili", "southern_sotho", "xhosa", "yoruba", "zulu"
    "algerian_arabic", "moroccan_arabic", "tunisian_arabic",  # These are alt arabic forms we could support
    "mozambican_portuguese",                                  # Alt portuguese forms
    "bemba", "central_kanuri", "fon", "twi", "wolof"          # These are African languages not in WURA
]

AYA_TRANSLATED_DATASETS = (
    "adversarial_qa_(t)","cnn-daily-mail_(t)", "dolly-v2_(t)",
    "flan-coqa_(t)", "flan-cot-submix_(t)", "flan-gem-wiki-lingua_(t)",
    "flan-lambada_(t)", "flan-unified-qa_(t)", "hotpotqa_(t)",
    "joke-explaination-inst_(t)", "mintaka-inst_(t)", "mlqa-en_(t)",
    "nq-open_(t)", "paws-wiki_(t)", "piqa_(t)", "soda-inst_(t)",
    "wiki_qa_(t)", "wiki-split-inst_(t)", "xlel_wd-inst_(t)"
)

AYA_TEMPLATED_DATASETS=[
    "afriqa-inst", "afrisenti-inst", "amharic_qa",
    "joke-explaination-inst", "masakhanews-inst",
    "mintaka-inst", "ntx-llm-inst", "nusax-senti-inst",
    "scirepeval-biomimicry-inst", "soda-inst", "uner-llm-inst",
    "wiki-split-inst", "xlel_wd-inst", "xwikis-inst"
]

AYA_DATASET_STATISTICS = get_dataset_statistics(
            BUCKET_DIR + "aya/statistics.json")


# TODO: @theyorubayesian - improve mixture num_examples by summing over tasks
def get_aya_rate(
    task: str,
    scale: float = 1.0,
    temperature: float = 1.0,
    language: Optional[str] = None,
    maximum: int | float | None = None
) -> float:
    
    if task == "human":
        num_examples = sum(AYA_DATASET_STATISTICS["aya-dataset"]["train"].values())
    elif task == "translated":
        num_examples = sum(
                sum(AYA_DATASET_STATISTICS[t]["train"].values())
                if "train" in AYA_DATASET_STATISTICS[t] else 0
                for t in AYA_TRANSLATED_DATASETS
        )
    elif task == "templated":
        num_examples = sum(
            sum(AYA_DATASET_STATISTICS[t]["train"].values())
            if "train" in AYA_DATASET_STATISTICS[t] else 0
            for t in AYA_TEMPLATED_DATASETS
        )
    else:
        assert language is not None

        for dataset_list in (AYA_TEMPLATED_DATASETS, AYA_TRANSLATED_DATASETS):
            if task in dataset_list:
                maximum *= AYA_DATASET_STATISTICS[task][language]["train"]
                return get_rate(scale, temperature, maximum)
        else:
            raise ValueError
    
    if isinstance(maximum, float):
        maximum *= num_examples
    
    return get_mixture_rate(num_examples, scale, temperature, maximum)


# configurable at language level
def create_aya_dataset_mixture(
    languages: list[str],
    dataset_name: str,
    suffix: Optional[str] = None,
    mixture_rate: dict[str, MixtureRateConfig]= rate_num_examples
) -> Optional[seqio.Mixture]:
    DATASET_PATH = Template(BUCKET_DIR + f"aya/{dataset_name}" + "/${split}/${language}.jsonl")
    prefix = [dataset_name, suffix][bool(suffix)]

    TEXT_SPEC = {
        field: tf.TensorSpec([], tf.string, name=field) 
        for field in [
            'inputs', 'targets', 'dataset_name', 
            'sub_dataset_name', 'task_type', 'language', 
            'script', 'split'
        ]
    }

    ID_SPEC = {
        field: tf.TensorSpec([None], tf.int32, name=field)
        for field in ["id", "template_id"]
    }

    AYA_SPEC = {**TEXT_SPEC, **ID_SPEC}

    parse_aya_jsonline = partial(jsonline_to_dict, specs=AYA_SPEC)

    aya_tasks = []
    for language in languages:
        sources = {
            split: DATASET_PATH.substitute(split="train", language=language)
            for split in ["train", "validation", "test"]
            if tf.io.gfile.exists(DATASET_PATH.substitute(split=split, language=language))
        }
        if not sources:
            continue

        task_name = normalize(f"{language}_{prefix}_aya")

        seqio.TaskRegistry.add(
            name=task_name,
            source=seqio.TextLineDataSource(
                split_to_filepattern=sources,
                num_input_examples={
                    source: AYA_DATASET_STATISTICS[dataset_name][source][language]
                    for source in sources
                }
            ),
            preprocessors=[
                parse_aya_jsonline,
                partial(take_subset, keys=["inputs", "targets", "dataset_name", "task_type", "language", "script"]),
                seqio.preprocessors.tokenize,
                seqio.preprocessors.append_eos_after_trim
            ],
            output_features=DEFAULT_OUTPUT_FEATURES,
            metric_fns=[]
        )

        aya_tasks.append(task_name)
    
    # if len(aya_tasks) > 1:
    mixture = seqio.MixtureRegistry.add(
        name=f"{prefix}_aya", 
        tasks=aya_tasks,
        default_rate=mixture_rate
    )
    return mixture


@gin.register
def add_aya_human_task(languages: Optional[list[str]] = None) -> seqio.Mixture:
    # Are these languages we're interested in or those that are human
    LANGUAGES=[
        "amharic", "egyptian_arabic", "english",
        "french", "hausa", "igbo", "nyanja",
        "plateau_malagasy", "portuguese", "shona",
        "somali", "swahili", "xhosa", "yoruba", "zulu",
        # African languages not in wura
        "moroccan_arabic", "wolof"
    ]

    if languages:
        assert all(lang in LANGUAGES for lang in languages), "Some languages passed are not in `aya-dataset`"
        LANGUAGES = languages
    
    aya_human_mixture = create_aya_dataset_mixture(
        LANGUAGES, "aya-dataset", mixture_rate=rate_num_examples, suffix="human")
    return aya_human_mixture


@gin.register
def add_aya_translated_task(**mixture_rate_cfg_map) -> seqio.Mixture:
    sub_mixtures = []
    
    for dataset_name in AYA_TRANSLATED_DATASETS:
        mixture = create_aya_dataset_mixture(
            AYA_DATASET_LANGUAGES,
            dataset_name=dataset_name,
            mixture_rate=rate_num_examples
        )
        mixture_rate_cfg = mixture_rate_cfg_map.get(
            f"{normalize(dataset_name)}_mixture_cfg", MixtureRateConfig())
        
        sub_mixtures.append((mixture, rate_num_examples_for_mixtures(
            mixture, **asdict(mixture_rate_cfg))
        ))
    
    translated_aya = seqio.MixtureRegistry.add(
        "translated_aya", sub_mixtures
    )
    return translated_aya


# Configurable at task level
@gin.register
def add_aya_templated_task(**mixture_rate_cfg_map) -> seqio.Mixture:
    sub_mixtures = []
    
    for dataset_name in AYA_TEMPLATED_DATASETS:
        mixture = create_aya_dataset_mixture(
            AYA_DATASET_LANGUAGES,
            dataset_name=dataset_name,
            mixture_rate=rate_num_examples
        )
        mixture_rate_cfg = mixture_rate_cfg_map.get(
            f"{normalize(dataset_name)}_mixture_cfg", MixtureRateConfig())
        
        sub_mixtures.append((mixture, rate_num_examples_for_mixtures(
            mixture, **asdict(mixture_rate_cfg))
        ))
    
    templated_aya = seqio.MixtureRegistry.add(
        "templated_aya", sub_mixtures
    )
    return templated_aya


@gin.register
def add_aya_task(
    human_mixture_cfg: MixtureRateConfig = MixtureRateConfig(),
    translated_mixture_cfg: MixtureRateConfig = MixtureRateConfig(),
    templated_mixture_cfg: MixtureRateConfig = MixtureRateConfig()
) -> seqio.Mixture:
    aya_human_mixture = add_aya_human_task()
    aya_translated_mixture = add_aya_translated_task()
    aya_templated_mixture = add_aya_templated_task()

    # In SeqIO, submixtures must carry float rates not funcs
    return seqio.MixtureRegistry.add(
        "aya",
        [
            (aya_human_mixture, rate_num_examples_for_mixtures(
                aya_human_mixture, **asdict(human_mixture_cfg))),
            (aya_templated_mixture, rate_num_examples_for_mixtures(
                aya_templated_mixture, **asdict(templated_mixture_cfg))),
            (aya_translated_mixture, rate_num_examples_for_mixtures(
                aya_translated_mixture, **asdict(translated_mixture_cfg)))
        ],
    )


default_task_factory = {
    "news_and_wiki": add_pretraining_task,
    "masakhanews": add_masakhanews_task,
    "lafand_mt": add_lafand_task,
    "xlsum": add_xlsum_task,
    "squad_v2": add_squad_task,
    "afriqa": add_afriqa_task,
    "aya_human": add_aya_human_task,
    "aya_templated": add_aya_templated_task,
    "aya_translated": add_aya_translated_task,
    "aya": add_aya_task,
}


def main(
    tasks: str,
    **configured_task_factory
):
    default_task_factory.update(configured_task_factory)
    task_factory = OrderedDict(sorted(default_task_factory.items()))

    if tasks == "all":
        """
        When finetuning AfriTeVa V2, we finetune on SQuAD & zero-shot to AfriQA gold passages
        Here we create two tasks: one with AfriQA included (used for evaluation) 
        & another w/o AfriQA included used for multitask finetuning
        """
        multitask_ft_tasks = []
        eval_tasks = []
        for task, func in task_factory.items():
            func()
            
            if task != "news_and_wiki":
                eval_tasks.append(task)

                if task != "afriqa":
                    multitask_ft_tasks.append(task)
        
        seqio.MixtureRegistry.add("_".join(multitask_ft_tasks), multitask_ft_tasks, default_rate=rate_num_examples)
        seqio.MixtureRegistry.add("_".join(eval_tasks), eval_tasks, default_rate=rate_num_examples)
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
            seqio.MixtureRegistry.add("_".join(eval_tasks), eval_tasks, default_rate=rate_num_examples)
