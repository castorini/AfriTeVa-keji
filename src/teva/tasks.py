from functools import partial

import seqio
from t5.data.preprocessors import span_corruption
from t5.data.utils import rate_num_examples

from teva.vocab import DEFAULT_VOCAB

DEFAULT_TEMPERATURE = 1.0  # TODO: @theyorubayesian @ToluClassics

DEFAULT_MIX_RATE = partial(
    rate_num_examples, temperature=DEFAULT_TEMPERATURE
)

DEFAULT_OUTPUT_FEATURES = {
    "inputs": seqio.Feature(vocabulary=DEFAULT_VOCAB, add_eos=True, required=False),
    "targets": seqio.Feature(vocabulary=DEFAULT_VOCAB, add_eos=True)
}

LANGS = [
    "afr", "amh", "arz", "eng", 
    "fra", "hau", "ibo", "kin", 
    "mlg", "nya", "orm", "por",
    "sna", "som", "sot", "swa",
    "tir", "xho", "yor", "zul"
]

CORPORA = ["wiki", "news"] 

# NOTE: This will require splitting our corpus into lang x corpus configurations
for corpus in CORPORA:
    corpus_tasks = []
    for lang in LANGS:
        lang_config_name = f"{lang}_{corpus}"
        seqio.TaskRegistry.add(
            lang_config_name,
            source=seqio.TfdsDataSource(
                tfds_name="arawa:1.0.0",    # TODO: @theyorubayesian @ToluClassics
                splits={
                    "train": lang_config_name,
                    "validation": f"{lang_config_name}-validation"
                }
            ),
            preprocessors=[
                partial(
                    seqio.preprocessors.rekey,
                    key_map={"inputs": None, "targets": "text"},
                ),
                seqio.preprocessors.tokenize,
                seqio.CacheDatasetPlaceholder(),
                span_corruption,
                seqio.preprocessors.append_eos_after_trim
            ],
            output_features=DEFAULT_OUTPUT_FEATURES,
            metric_fns=[]
        )

        corpus_tasks.append(lang_config_name)
    
    seqio.MixtureRegistry.add(corpus, corpus_tasks, default_rate=DEFAULT_MIX_RATE)

# TODO: Finetune & Evaluate tasks