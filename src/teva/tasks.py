import copy
import enum
import os
from dataclasses import asdict
from functools import partial
from string import Template
from typing import Final, Literal, Optional, Sequence

import gin
import seqio
import tensorflow as tf
from dotenv import load_dotenv
from t5.data.preprocessors import span_corruption, summarize
from t5.data.utils import rate_num_examples
from t5.evaluation.metrics import accuracy, bleu, rouge, squad as squad_metrics

from teva.constants import *
from teva.metrics import chrf, weighted_multiclass_f1
from teva.mixture_utils import *
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
from teva.utils import (
    get_dataset_statistics,
    get_labels,
    get_language_from_code,
    mix_exists,
    normalize,
    TaskNotFoundException,
)
from teva.vocab import DEFAULT_VOCAB

load_dotenv()

DATA_DIR=os.getenv("DATA_DIR", "data")

DEFAULT_OUTPUT_FEATURES: Final = {
    "inputs": seqio.Feature(vocabulary=DEFAULT_VOCAB, add_eos=True, required=False),
    "targets": seqio.Feature(vocabulary=DEFAULT_VOCAB, add_eos=True)
}

@gin.constants_from_enum
@enum.unique
class TevaTasks(enum.Enum):
    WURA = "wura"
    IFT = "instruction_finetuning"
    SFT = "supervised_finetuning"
    EVAL = "eval"
    # EVAL
    MASAKHANEWS = "masakhanews"
    LAFAND = "lafand"
    XLSUM = "xlsum"
    SQUAD = "squad"
    AFRIQA = "afriqa"
    # NAIJARC = "naijarc"
    # BELEBELE = "belebele"
    # SIB = "sib"
    # IFT
    HUMAN_AYA = "aya-dataset"
    TRANSLATED_AYA = "translated_aya"
    TEMPLATED_AYA = "templated_aya"
    AYA_COLLECTION = "aya_collection"
    XP3X = "xp3x"
    OCTOPACK_OSST = "octopack_osst"
    OIG_SMALL_CHIP2 = "oig_small_chip2"
    TASKSOURCE_INSTRUCT = "tasksource_instruct"
    FLAN_NIV2_SUBMIX = "flan_niv2_submix"
    FLAN2021_SUBMIX = "flan2021_submix"
    FLAN_COT_SUBMIX = "flan_cot_submix"
    FLAN_DIALOG_SUBMIX = "flan_dialog_submix"
    FLAN_T0_SUBMIX = "flan_t0_submix"
    FLAN_COLLECTION = "flan_collection"
    DPI_TEMPLATED = "dpi_templated"
    TEMPLATED_IFT = "templated_ift"

    @classmethod
    def get_supervised_ft_tasks(cls) -> frozenset["TevaTasks"]:
        return frozenset([
            cls.LAFAND, cls.MASAKHANEWS, cls.XLSUM, cls.SQUAD
        ])
    
    @classmethod
    def get_evaluation_tasks(cls) -> frozenset["TevaTasks"]:
        return frozenset([
            cls.LAFAND, cls.MASAKHANEWS, cls.XLSUM, cls.AFRIQA
        ])
    
    @classmethod
    def get_flan_collection_tasks(cls) -> frozenset["TevaTasks"]:
        return frozenset([
            cls.FLAN2021_SUBMIX,
            cls.FLAN_COT_SUBMIX,
            cls.FLAN_DIALOG_SUBMIX,
            cls.FLAN_NIV2_SUBMIX,
            cls.FLAN_T0_SUBMIX
        ])
    
    @classmethod
    def get_dpi_templated_tasks(cls) -> frozenset["TevaTasks"]:
        return frozenset([
            cls.TASKSOURCE_INSTRUCT,
            cls.OIG_SMALL_CHIP2,
            *cls.get_flan_collection_tasks()
        ])
    
    @classmethod
    def get_aya_collection_tasks(cls) -> frozenset["TevaTasks"]:
        return frozenset([cls.HUMAN_AYA, cls.TEMPLATED_AYA, cls.TRANSLATED_AYA])
    
    @classmethod
    def get_templated_instruction_tasks(cls, flatten_dpi: bool = False) -> frozenset["TevaTasks"]:
        tasks = [cls.XP3X, cls.TEMPLATED_AYA]
        
        if flatten_dpi:
            tasks += cls.get_dpi_templated_tasks()
        else:
            tasks.append(cls.DPI_TEMPLATED)

        return frozenset(tasks)
    
    @classmethod
    def get_instruction_tasks(cls) -> frozenset["TevaTasks"]:
        return cls.get_templated_instruction_tasks(flatten_dpi=True) | cls.get_aya_collection_tasks()


def add_wura_task():
    STATISTICS_PATH = Template(os.path.join(DATA_DIR, "wura/${corpus}-passages/stats"))
    DATASET_PATH = Template(os.path.join(DATA_DIR, "wura/${corpus}-passages/${split}/${language}.txt"))

    CORPORA: Final = ["wiki", "news"] 
    lm_tasks = []

    for corpus in CORPORA:
        corpus_tasks = []

        dataset_statistics = get_dataset_statistics(STATISTICS_PATH.substitute(corpus=corpus.capitalize()))
        for lang in WURA_LANGUAGES:
            if corpus == "news" and lang in ["arz"]:
                continue

            lang_config_name = f"{lang}_wura_{corpus}"

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

    seqio.MixtureRegistry.add("wura", lm_tasks, default_rate=rate_num_examples)

# ---------------------------------------------------------------------------------
# Evaluation Tasks
# ---------------------------------------------------------------------------------

# --------------------------
# MasakhaNews Classification
# --------------------------
def add_masakhanews_task():
    MASAKHANEWS_DATASET_PATH = Template(os.path.join(DATA_DIR, "masakhanews/${language}/${split}.jsonl"))
    LABELS_PATH = Template(os.path.join(DATA_DIR, "masakhanews/${language}/labels.txt"))
    
    masakhanews_dataset_statistics = get_dataset_statistics(os.path.join(DATA_DIR, "masakhanews/stats"))
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
    LAFAND_DATASET_PATH = Template(os.path.join(DATA_DIR, "lafand/${pivot}-${language}/${split}.json"))
    
    lafand_dataset_statistics = get_dataset_statistics(os.path.join(DATA_DIR, "lafand/stats"))
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
        en_xx_prefix = f"Translate {get_language_from_code(pivot)} to {get_language_from_code(language)}: "

        lafand_xx_en_task_name = f"{language}_{pivot}_lafand_mt"
        xx_en_prefix = f"Translate {get_language_from_code(language)} to {get_language_from_code(pivot)}: "

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
    XLSUM_DATASET_PATH = Template(os.path.join(DATA_DIR, "xlsum/${language}/${split}.json"))
    xlsum_dataset_statistics = get_dataset_statistics(os.path.join(DATA_DIR, "xlsum/stats"))
    
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
    SQUAD_DATASET_PATH = Template(os.path.join(DATA_DIR, "squad_v2/${split}.jsonl"))
    squad_dataset_statistics = {
        split: value["squad_v2"]
        for split, value in get_dataset_statistics(os.path.join(DATA_DIR, "squad_v2/stats")).items()
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
    AFRIQA_DATASET_PATH = Template(
        os.path.join(
            DATA_DIR, 
            "afriqa/gold_passages/${language}/gold_span_passages.afriqa.${language}.${pivot}.${split}.json"
        )
    )

    afriqa_dataset_statistics = get_dataset_statistics(os.path.join(DATA_DIR, "afriqa/gold_passages/stats"))
    afriqa_dataset_statistics = {
        language: {
            split: afriqa_dataset_statistics[split][language]
            for split in ["train", "validation", "test"]
        } for language in AFRIQA_LANGUAGES
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
    for language in AFRIQA_LANGUAGES:
        task_name = f"{language}_afriqa"
        pivot = "fr" if language in AFRIQA_FR_PIVOT_LANGUAGES else "en"
        
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

def add_evaluation_tasks():
    ...

# ---
# Aya
# ---
@gin.register
def create_aya_dataset_mixture(
    languages: Sequence[str],
    dataset_name: str,
    suffix: Optional[str] = None,
    **mixture_rate_cfg_map: MixtureRateConfig
) -> Optional[seqio.Mixture]:
    DATASET_PATH = Template(os.path.join(DATA_DIR, f"aya/{dataset_name}", "${split}/${language}.jsonl"))
    prefix = [dataset_name, suffix][bool(suffix)]

    AYA_DATASET_STATISTICS = get_dataset_statistics(os.path.join(DATA_DIR, "aya/statistics.json"))
    
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

        if not mix_exists(task_name):
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

        mixture_rate_cfg = mixture_rate_cfg_map.get(
            f"{normalize(dataset_name)}_mixture_cfg", MixtureRateConfig())
        mixture_rate = get_rate(**asdict(mixture_rate_cfg))

        aya_tasks.append((task_name, mixture_rate))
    
    # if len(aya_tasks) > 1:
    if not mix_exists(f"{prefix}_aya"):
        mixture = seqio.MixtureRegistry.add(
            name=f"{prefix}_aya", 
            tasks=aya_tasks,
            default_rate=rate_num_examples
        )
    else:
        mixture = seqio.MixtureRegistry.get(f"{prefix}_aya")
    
    return mixture


@gin.register
def add_aya_human_task(languages: Sequence[str] = AYA_HUMAN_LANGUAGES) -> seqio.Mixture:
    assert AYA_HUMAN_LANGUAGES.issuperset(languages)
    aya_human_mixture = create_aya_dataset_mixture(
        languages, "aya-dataset", mixture_rate=rate_num_examples, suffix="human")
    return aya_human_mixture


@gin.register
def add_aya_translated_task(
    datasets: Sequence[str] = AYA_TEMPLATED_DATASETS,
    **mixture_rate_cfg_map: MixtureRateConfig
) -> seqio.Mixture:
    assert AYA_TEMPLATED_DATASETS.issuperset(datasets)
    sub_mixtures = []
    
    for dataset_name in datasets:
        mixture = create_aya_dataset_mixture(
            AYA_LANGUAGES,
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


@gin.register
def add_aya_templated_task(
    datasets: Sequence[str] = AYA_TEMPLATED_DATASETS,
    **mixture_rate_cfg_map: MixtureRateConfig
) -> seqio.Mixture:
    assert AYA_TEMPLATED_DATASETS.issuperset(datasets)
    sub_mixtures = []
    
    for dataset_name in datasets:
        mixture = create_aya_dataset_mixture(
            AYA_LANGUAGES,
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
def add_aya_collection_task(
    human_mixture_cfg: MixtureRateConfig = MixtureRateConfig(),
    translated_mixture_cfg: MixtureRateConfig = MixtureRateConfig(),
    templated_mixture_cfg: MixtureRateConfig = MixtureRateConfig()
) -> seqio.Mixture:
    aya_human_mixture = add_aya_human_task()
    aya_translated_mixture = add_aya_translated_task()
    aya_templated_mixture = add_aya_templated_task()

    # In SeqIO, submixtures must carry float rates not funcs
    return seqio.MixtureRegistry.add(
        "aya_collection",
        [
            (aya_human_mixture, rate_num_examples_for_mixtures(
                aya_human_mixture, **asdict(human_mixture_cfg))),
            (aya_templated_mixture, rate_num_examples_for_mixtures(
                aya_templated_mixture, **asdict(templated_mixture_cfg))),
            (aya_translated_mixture, rate_num_examples_for_mixtures(
                aya_translated_mixture, **asdict(translated_mixture_cfg)))
        ],
    )


@gin.register
def add_xp3x_task(
    languages: Sequence[str] = XP3X_LANGUAGE_CODES,
    **mixture_rate_cfg_map: MixtureRateConfig
):
    assert XP3X_LANGUAGE_CODES.issuperset(languages)

    XP3X_DATASET_PATH = Template(os.path.join(DATA_DIR, "xP3x/${language}/*.jsonl"))
    xp3x_dataset_statistics = get_dataset_statistics(os.path.join(DATA_DIR, "xP3x/stats"))

    XP3X_JSONLINE_SPECS = {
        field: tf.TensorSpec(tf.TensorShape([]), tf.string, name=field)
        for field in ["inputs", "language", "split", "template", "dataset", "config"]
    }
    parse_xp3x_jsonline = partial(jsonline_to_dict, specs=XP3X_JSONLINE_SPECS)

    xP3x_tasks = []

    for language in languages:
        task_name = f"{language}_xp3x"

        xp3x_source = seqio.TextLineDataSource(
            split_to_filepattern={
                "train": XP3X_DATASET_PATH.substitute(language=language)
            },
            num_input_examples=xp3x_dataset_statistics[language]
        )
        
        if mix_exists(task_name):
            continue

        seqio.TaskRegistry.add(
            name=task_name,
            source=xp3x_source,
            preprocessors=[
                parse_xp3x_jsonline,
                partial(take_subset, keys=["inputs", "targets", "config"]),
                seqio.preprocessors.tokenize,
                seqio.preprocessors.append_eos_after_trim
            ],
            output_features=DEFAULT_OUTPUT_FEATURES,
            metric_fns=[]
        )

        mixture_rate_cfg = mixture_rate_cfg_map.get(
            f"{language}_mixture_cfg", MixtureRateConfig())
        mixture_rate = get_rate(**asdict(mixture_rate_cfg))

        xP3x_tasks.append((task_name, mixture_rate))
    
    if mix_exists("xP3x"):
        mixture = seqio.MixtureRegistry.get("xP3x")
    else:
        mixture = seqio.MixtureRegistry.add(
            name="xP3x",
            tasks=xP3x_tasks,
            default_rate=rate_num_examples
        )
    return mixture


@gin.register
def add_octopack_osst():
    ...


def add_oig_small_chip2():
    # TODO: @theyorubayesian - Confirm dataset path
    if mix_exists("oig-small-chip2"):
        return
    
    OIG_DATASET_PATH = os.path.join(DATA_DIR, "OIG-small-chip2/train*.jsonl")
    oig_dataset_statistics = get_dataset_statistics(os.path.join(DATA_DIR, "OIG-small-chip2/stats"))

    OIG_JSONLINE_SPECS = {
        field: tf.TensorSpec(tf.TensorShape([]), tf.string, name=field)
        for field in ["user", "chip2"]
    }
    parse_oig_jsonline = partial(jsonline_to_dict, specs=OIG_JSONLINE_SPECS)

    seqio.TaskRegistry.add(
        "oig-small-chip2",
        source=seqio.TextLineDataSource(
            split_to_filepattern={
                "train": OIG_DATASET_PATH
            },
            num_input_examples=oig_dataset_statistics
        ),
        preprocessors=[
            parse_oig_jsonline,
            partial(
                seqio.preprocessors.rekey,
                key_map={"inputs": "user", "targets": "chip2"}
            ),
            seqio.preprocessors.tokenize,
            seqio.preprocessors.append_eos_after_trim
        ],
        output_features=DEFAULT_OUTPUT_FEATURES,
        metric_fns=[]
    )


def add_tasksource_instruct():
    # if mix_exists(f"{dataset_name.value}_submix"):
        # return
    ...


def create_flan_collection_submix_task(
    dataset_name: FlanTask,
    flan_collection_statistics: FlanCollectionStatistics | None = None
) -> seqio.Mixture:
    if mix_exists(f"{dataset_name.value}_submix"):
        return seqio.MixtureRegistry.get(f"{dataset_name.value}_submix")
    
    FLAN_COLLECTION_DATASET_PATH = os.path.join(DATA_DIR, f"flan_collection/{dataset_name.value}/train*.jsonl")
    
    if flan_collection_statistics is None:
        flan_collection_statistics = get_dataset_statistics(os.path.join(DATA_DIR, "flan_collection/stats"))
    
    submix_statistics = {"train": flan_collection_statistics[dataset_name.value]}
    TEXT_SPEC = {
        field: tf.TensorSpec([], tf.int32, name=field)
        for field in ["inputs", "targets", "task_source", "task_name", "template_type"]
    }

    parse_flan_jsonline = partial(jsonline_to_dict, specs=TEXT_SPEC)

    mixture = seqio.TaskRegistry.add(
        name=f"{dataset_name.value}_submix",
        source=seqio.TextLineDataSource(
            split_to_filepattern={"train": FLAN_COLLECTION_DATASET_PATH},
            num_input_examples=submix_statistics,
        ),
        preprocessors=[
            parse_flan_jsonline,
            partial(take_subset, keys=["inputs", "targets"]),
            seqio.preprocessors.tokenize,
            seqio.preprocessors.append_eos_after_trim
        ],
        output_features=DEFAULT_OUTPUT_FEATURES,
        metric_fns=[]
    )
    return mixture


def add_niv2_submix_filtered(flan_collection_statistics: FlanCollectionStatistics | None = None) -> seqio.Mixture:
    return create_flan_collection_submix_task(
        dataset_name=FlanTask.NIV2,
        flan_collection_statistics=flan_collection_statistics
    )


def add_flan2021_submix_filtered(flan_collection_statistics: FlanCollectionStatistics | None = None) -> seqio.Mixture:
    return create_flan_collection_submix_task(
        dataset_name=FlanTask.FLAN2021,
        flan_collection_statistics=flan_collection_statistics
    )


def add_cot_submix_filtered(flan_collection_statistics: FlanCollectionStatistics | None = None) -> seqio.Mixture:
    return create_flan_collection_submix_task(
        dataset_name=FlanTask.COT,
        flan_collection_statistics=flan_collection_statistics
    )


def add_t0_submix(flan_collection_statistics: FlanCollectionStatistics | None = None) -> seqio.Mixture:
    return create_flan_collection_submix_task(
        dataset_name=FlanTask.T0,
        flan_collection_statistics=flan_collection_statistics
    )


def add_dialog_submix(flan_collection_statistics: FlanCollectionStatistics | None = None) -> seqio.Mixture:
    return create_flan_collection_submix_task(
        dataset_name=FlanTask.DIALOG,
        flan_collection_statistics=flan_collection_statistics
    )


def add_dpi_templated_tasks(**mixture_rate_cfg_map: MixtureRateConfig):
    """
    This is a mixture of the following tasks/mixtures:
        * Octopack OSST
        * OpenInstruction Generalist
        * Filtered Flan Collection (NIV2, COT, FLAN2021, T0, Dialog)
        * TaskSource Instruct?
    """
    sub_mixtures = []

    for task in TevaTasks.get_dpi_templated_tasks():
        task_func = default_task_factory[task]


@gin.register
def add_templated_instruction_ft_tasks():
    """
    This is a mixture of the following mixtures:
        * Aya Templated
        * xP3x
        * Data Provenance Initiative
    """
    sub_mixtures = []

    for task in TevaTasks.get_templated_instruction_tasks():
        ...



@gin.register
def add_instruction_ft_tasks(**mixture_rate_cfg_map: MixtureRateConfig):
    """
    This is a mixture of the following mixtures:
        * TemplatedIFT
        * AyaHuman
        * AyaTranslated
    """


default_task_factory: dict[TevaTasks, callable] = {
    TevaTasks.WURA: add_wura_task,
    TevaTasks.EVAL: add_evaluation_tasks,
    TevaTasks.IFT: add_instruction_ft_tasks,
    # TevaTasks.SFT: add_supervised_ft_tasks,   # TODO: @theyorubayesian
    TevaTasks.MASAKHANEWS: add_masakhanews_task,
    TevaTasks.LAFAND: add_lafand_task,
    TevaTasks.XLSUM: add_xlsum_task,
    TevaTasks.SQUAD: add_squad_task,
    TevaTasks.AFRIQA: add_afriqa_task,
    TevaTasks.HUMAN_AYA: add_aya_human_task,
    TevaTasks.TEMPLATED_AYA: add_aya_templated_task,
    TevaTasks.TRANSLATED_AYA: add_aya_translated_task,
    TevaTasks.AYA_COLLECTION: add_aya_collection_task,
    TevaTasks.XP3X: add_xp3x_task,
    TevaTasks.FLAN2021_SUBMIX: add_flan2021_submix_filtered,
    TevaTasks.FLAN_COT_SUBMIX: add_cot_submix_filtered,
    TevaTasks.FLAN_DIALOG_SUBMIX: add_dialog_submix,
    TevaTasks.FLAN_NIV2_SUBMIX: add_niv2_submix_filtered,
    TevaTasks.FLAN_T0_SUBMIX: add_t0_submix,
    TevaTasks.OIG_SMALL_CHIP2: add_oig_small_chip2,
    TevaTasks.OCTOPACK_OSST: add_octopack_osst,
    TevaTasks.DPI_TEMPLATED: add_dpi_templated_tasks,
    TevaTasks.TEMPLATED_IFT: add_templated_instruction_ft_tasks
}


def setup_tasks(
    tasks: Sequence[TevaTasks] | Literal["all"],
    **configured_task_factory: callable
):
    """
    `configured_task_factory` provides a way to update the task factory dict with 
    gin configured versions of the task functions
    """
    factory = copy.deepcopy(default_task_factory)
    factory.update(configured_task_factory)

    if tasks == "all":
        factory[TevaTasks.WURA]()
        factory[TevaTasks.EVAL]()
        factory[TevaTasks.IFT]()
    else:
        # TODO: @theyorubayesian - Dynamically creating a mixture of tasks has not been tested
        # What are the implications of setting default mixture rate to 1.0?
        selected_sft_tasks = []
        selected_eval_tasks = []
        selected_ift_tasks = []

        all_eval_tasks = TevaTasks.get_evaluation_tasks()
        all_ift_tasks = TevaTasks.get_instruction_tasks()
        all_sft_tasks = TevaTasks.get_supervised_ft_tasks()

        for task in tasks:
            factory[task]()

            if task in all_eval_tasks:
                selected_eval_tasks.append(task)
            elif task in all_ift_tasks:
                selected_ift_tasks.append(task)

            if task in all_sft_tasks:
                selected_sft_tasks.append(task)
        
        if len(selected_sft_tasks) > 1:
            seqio.MixtureRegistry.add("teva_sft", selected_sft_tasks, default_rate=1.0)
        
        if len(selected_eval_tasks) > 1:
            seqio.MixtureRegistry.add("teva_evaluation", selected_eval_tasks, default_rate=1.0)

        if len(selected_ift_tasks) > 1:
            seqio.MixtureRegistry.add("teva_ift", selected_ift_tasks, default_rate=1.0)

    print(f"Registered tasks: \n\n{seqio.MixtureRegistry.names()}")
