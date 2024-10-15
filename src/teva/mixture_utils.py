from dataclasses import dataclass
from functools import partial
from typing import Optional

import gin
import seqio
from t5.data.utils import rate_num_examples


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
