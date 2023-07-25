from typing import Dict, Literal, List, TypedDict, Union

import tensorflow as tf
import tensorflow_io as tfio
from t5.data.preprocessors import _string_join
from seqio.utils import map_over_dataset


JsonSpec = Dict[str, Union[tf.TensorSpec, "JsonSpec"]]


class AfriQAInput(TypedDict):
    id: str
    title: str
    context: str
    question_lang: set
    question_translated: str
    answer_lang: str
    answer_pivot: TypedDict("answer_pivot", {"text": List[str], "answer_start": List[int]})


class ClassificationInput(TypedDict):
    headline: str
    category: str
    text: str
    url: str


class SQuADInput(TypedDict):
    id: str
    title: str
    context: str
    question: str
    answers: TypedDict("answers", {"text": List[str], "answer_start": List[int]})


class TTTExample(TypedDict):
    inputs: Union[str, tf.Tensor]
    targets: Union[str, tf.Tensor]


@map_over_dataset
def line_to_dict(line: str) -> TTTExample:
    return {"targets": line, "inputs": ""}


@map_over_dataset
def jsonline_to_dict(line: str, specs: JsonSpec, return_key: str = None) -> dict:
    decoded_json =  tfio.experimental.serialization.decode_json(line, specs=specs)
    if return_key:
        return decoded_json[return_key]
    return decoded_json


@map_over_dataset
def create_news_classification_example(
    example: ClassificationInput,
    config: Literal["headline", "headline_and_text", "text"] = "text",
    prompt: str = "classify:"
) -> TTTExample:
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
def translate(example, prefix, src_code, tgt_code) -> TTTExample:
    return {
        'inputs': tf.strings.join([prefix, example[src_code]]),
        'targets': example[tgt_code],
    }


@map_over_dataset
def squad(example: SQuADInput) -> TTTExample:
    q = tf.strings.strip(example["question"])
    c = tf.strings.strip(example["context"])

    output = {
        "inputs": _string_join(["question:", q, "context:", c]),
        "targets": tf.cond(
            tf.equal(tf.size(example["answers"]["text"]), 0),  
            lambda: "",
            lambda: example["answers"]["text"][0],
        )
    }
    return output


@map_over_dataset
def afriqa(example: AfriQAInput, use_translated_question: bool = False) -> TTTExample:
    q = tf.strings.strip(example["question_translated" if use_translated_question else "question_lang"])
    c = tf.strings.strip(example["context"])

    # TODO: @theyorubayesian - What do we set when there is no answer?
    output = {
        "inputs": _string_join(["question:", q, "context:", c]),
        "targets": tf.cond(
            tf.equal(tf.size(example["answer_pivot"]["answer_start"]), 0),
            lambda: "",
            lambda: tf.cond(
                tf.equal(example["answer_pivot"]["answer_start"][0], -1),
                lambda: "",
                lambda: example["answer_pivot"]["text"][0]
            ),
        )
    }
    return output
