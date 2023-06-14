from typing import List, Literal, TypedDict, Union

import numpy as np
import tensorflow as tf
import tensorflow_io as tfio
from seqio.utils import map_over_dataset
from t5.evaluation.metrics import sklearn_metrics_wrapper


class DatasetStatistics(TypedDict, total=False):
    dev: int
    eval: int
    train: int
    test: int
    val: int
    validation: int


class CorpusStatistics(TypedDict):
    language: DatasetStatistics


class ClassificationInput(TypedDict):
    headline: str
    category: str
    text: str
    url: str


@map_over_dataset
def line_to_dict(line: str) -> TypedDict("example", targets=str, inputs=str):
    return {"targets": line, "inputs": ""}

@map_over_dataset
def jsonline_to_dict(line: str, specs):
    return tfio.experimental.serialization.decode_json(line, specs=specs)


@map_over_dataset
def create_news_classification_example(
    example: ClassificationInput,
    config: Literal["headline", "headline_and_text", "text"] = "text",
    prompt: str = "classify:"
) -> TypedDict("example", targets=str, inputs=str):
    return {
        "inputs": tf.strings.join(
            inputs=[
                prompt,
                example['headline'] if config == 'headline_only' 
                else example['text'] if config == 'text' 
                else example['text'] + example['headline']],
            separator=" "
        ),
        "targets": example["category"]
    }


@map_over_dataset
def translate(example, src_language, tgt_language):
    prefix = f"translate {src_language} to {tgt_language}: "
    return {
        'inputs': tf.strings.join([prefix, example[src_language]]),
        'targets': example[tgt_language],
    }


def get_labels(labels_file: str) -> List[str]:
    with tf.io.gfile.GFile(labels_file) as f:
        return f.read().splitlines()


def get_dataset_statistics(file: str) -> CorpusStatistics:
    stats: CorpusStatistics = {}

    with tf.io.gfile.GFile(file) as f:
        for line in f:
            num_input_examples, language_and_split = line.split()
            split, file = language_and_split.split("/")
            language = file.split(".")[0]

            if split in ["dev", "eval", "val", "validation"]:
                split = "validation"
            
            if language in stats:
                stats[language][split] = int(num_input_examples)
            else:
                stats[language] = {split: int(num_input_examples)}
    
    return stats

# -------
# Metrics
# -------
def weighted_multiclass_f1(labels, **metric_fn_kwargs):
    """Computes the unweighted average of the F1 per class."""
    return sklearn_metrics_wrapper(
        "f1_score",
        metric_dict_str="weighted_%dclass_f1" % len(labels),
        metric_post_process_fn=lambda x: 100 * x,
        labels=labels,
        average="weighted",
        **metric_fn_kwargs
    )
