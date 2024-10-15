import json
import re
from string import Template
from typing import Dict, List, NewType, TypedDict, Union

import pycountry
import seqio
import tensorflow as tf

Language = NewType('Language', str)

# Function to normalize a task name
pattern = re.compile(r'[^\w\d\.\:_#()]')
normalize = lambda name: pattern.sub('_', name).replace("(", "").replace(")", "")


class SplitStatistics(TypedDict, total=False):
    train: int
    test: int
    validation: int


class CorpusStatisticsByLanguage(TypedDict):
    language: SplitStatistics


class CorpusStatisticsBySplit(TypedDict):
    train: Dict[Language, int]
    test: Dict[Language, int]
    validation: Dict[Language, int]

CorpusStatistics = Union[CorpusStatisticsByLanguage, CorpusStatisticsBySplit]


class TaskNotFoundException(Exception):
    message_template = Template("${task} not found in list of tasks (${tasks})")

    def __init__(self, task: str, tasks: List[str]) -> None:
        self.message = self.message_template.substitute(task=task, tasks=" , ".join(tasks))


def task_or_mix_exists(mix_name: str) -> bool:
    return mix_name in seqio.MixtureRegistry.names() \
        or mix_name in seqio.TaskRegistry.names()


def get_labels(labels_file: str) -> List[str]:
    with tf.io.gfile.GFile(labels_file) as f:
        return f.read().splitlines()


def get_dataset_statistics(file: str):
    stats = {}

    with tf.io.gfile.GFile(file) as f:
        content = f.read()
        
        try:
            stats = json.loads(content)
        except json.JSONDecodeError:
            for line in content.split("\n"):
                num_input_examples, language_and_split = line.split()
                split, file = language_and_split.split("/")
                language = file.split(".")[0]

                if language in ["dev", "eval", "val", "validation"]:
                    language = "validation"
                
                if split in ["dev", "eval", "val", "validation"]:
                    split = "validation"
                
                if language in stats:
                    stats[language][split] = int(num_input_examples)
                else:
                    stats[language] = {split: int(num_input_examples)}
    
    return stats


def get_language_from_code(code: str) -> str:
    language_tuple = pycountry.languages.get(**{f"alpha_{len(code)}": code})
    return language_tuple.name
