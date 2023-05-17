from typing import TypedDict

from seqio.utils import map_over_dataset


@map_over_dataset
def line_to_dict(line: str) -> TypedDict("example", targets=str, inputs=str):
    return {"targets": line, "inputs": ""}
