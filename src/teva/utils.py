from typing import TypedDict

import tensorflow as tf
from seqio.utils import map_over_dataset


class DatasetStatistics(TypedDict, total=False):
    dev: int
    eval: int
    train: int
    test: int
    val: int
    validation: int


class CorpusStatistics(TypedDict):
    language: DatasetStatistics


@map_over_dataset
def line_to_dict(line: str) -> TypedDict("example", targets=str, inputs=str):
    return {"targets": line, "inputs": ""}


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
