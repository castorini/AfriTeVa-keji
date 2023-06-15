from typing import Dict, Literal, TypedDict, Union

import tensorflow as tf
import tensorflow_io as tfio
from seqio.utils import map_over_dataset


JsonSpec = Dict[str, Union[tf.TensorSpec, "JsonSpec"]]

class ClassificationInput(TypedDict):
    headline: str
    category: str
    text: str
    url: str

class TTTExample(TypedDict):
    inputs: Union[str, tf.Tensor]
    targets: Union[str, tf.Tensor]


@map_over_dataset
def line_to_dict(line: str) -> TypedDict("example", targets=str, inputs=str):
    return {"targets": line, "inputs": ""}


@map_over_dataset
def jsonline_to_dict(line: str, specs: JsonSpec, return_key: str = None):
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
