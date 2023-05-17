import os
from functools import partial
from string import Template
from typing import Final

import seqio
from dotenv import load_dotenv
from t5.data.preprocessors import span_corruption
from t5.data.utils import rate_num_examples

from teva.vocab import DEFAULT_VOCAB
from teva.utils import line_to_dict

load_dotenv()

BUCKET_DIR=os.getenv("DATA_GCP_BUCKET_DIR")
DATASET_PATH = Template(BUCKET_DIR + "${corpus}Passages/${split}/${language}.txt")

DEFAULT_TEMPERATURE: Final = 1.0  # TODO: @theyorubayesian @ToluClassics
DEFAULT_MIX_RATE = partial(
    rate_num_examples, temperature=DEFAULT_TEMPERATURE
)

DEFAULT_OUTPUT_FEATURES: Final = {
    "inputs": seqio.Feature(vocabulary=DEFAULT_VOCAB, add_eos=True, required=False),
    "targets": seqio.Feature(vocabulary=DEFAULT_VOCAB, add_eos=True)
}

LANGS: Final = [ 
    "afr", "amh", "arz", "eng", 
    "fra", "hau", "ibo", "kin", 
    "mlg", "nya", "orm", "por",
    "sna", "som", "sot", "swa",
    "tir", "xho", "yor", "zul"
]

CORPORA: Final = ["wiki", "news"] 
lm_tasks = []

for corpus in CORPORA:
    corpus_tasks = []
    for lang in LANGS:
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
                }
            ),
            preprocessors=[
                line_to_dict,
                seqio.preprocessors.tokenize,
                seqio.CacheDatasetPlaceholder(),    # TODO: @theyorubayesian @ToluClassics
                span_corruption,
                seqio.preprocessors.append_eos_after_trim
            ],
            output_features=DEFAULT_OUTPUT_FEATURES,
            metric_fns=[]
        )

        corpus_tasks.append(lang_config_name)
    
    seqio.MixtureRegistry.add(corpus, corpus_tasks, default_rate=DEFAULT_MIX_RATE)
    lm_tasks += corpus_tasks

seqio.MixtureRegistry.add("_".join(CORPORA), lm_tasks, default_rate=DEFAULT_MIX_RATE)

# TODO: Finetune & Evaluate tasks
