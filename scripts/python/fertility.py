import argparse
import string
from typing import Generator
from typing import Type

import transformers
from transformers import PreTrainedTokenizer

RESULT_TEMPLATE = string.Template(
    """
    Total number of words: $n_words
    Monolingual Tokenizer - Number of tokens: $n_mono_tokens
    Multilingual Tokenizer - Number of tokens: $n_multi_tokens

    Monolingual Fertility - $n_mono_fertility
    Multilingual Fertility - $n_multi_fertility
    """
)


def setup_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sample Sentences from monolingual corpora to train tokenizer")
    parser.add_argument("--input-files", type=str, help="Path text files")
    parser.add_argument("--mono-tokenizer-cls")
    parser.add_argument("--mono-tokenizer-name-or-path", type=str, required=True, help="Number of sentences in each file input-file")
    parser.add_argument("--multi-tokenizer-cls")
    parser.add_argument("--multi-tokenizer-name-or-path", type=str, required=True, help="path to store sampled sentences")
    
    args = parser.parse_args()
    return args


def get_tokenizer(cls: str, tokenizer_name_or_path: str) -> PreTrainedTokenizer:
    tokenizer_cls: Type[PreTrainedTokenizer] = getattr(transformers, cls)
    tokenizer: PreTrainedTokenizer = tokenizer_cls.from_pretrained(tokenizer_name_or_path)
    return tokenizer


def get_iterator(input_file: str) -> Generator(str, None, None):
    with open(input_file, "r") as f:
        for line in f:
            yield line.translate(
                str.maketrans({key: " {0} " for key in [".", ",", "!"]})
            )


def main():
    args = setup_args()

    mono_tokenizer = get_tokenizer(args.mono_tokenizer_cls, args.mono_tokenizer_name_or_path)
    multi_tokenizer = get_tokenizer(args.multi_tokenizer_cls, args.multi_tokenizer_name_or_path)

    n_words = 0
    n_mono_tokens = 0
    n_multi_tokens = 0
    for sentence in get_iterator(args.input_file):
        n_words += len(sentence.split())

        n_mono_tokens += len(mono_tokenizer.tokenize(sentence))
        n_multi_tokens += len(multi_tokenizer.tokenize(sentence))

    result = RESULT_TEMPLATE.substitute(
        n_words=n_words,
        n_mono_tokens=n_mono_tokens,
        n_multi_tokens=n_multi_tokens,
        n_mono_fertility=n_mono_tokens / n_words,
        n_multi_fertility=n_multi_tokens / n_words
    )

    print(result)
