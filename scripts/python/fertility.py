import argparse
import json
import string
from collections import defaultdict
from pathlib import Path
from pprint import pprint
from typing import Generator
from typing import List
from typing import TextIO
from typing import Tuple
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
    parser.add_argument("--input-file", type=argparse.FileType("r"), help="Path text files")
    parser.add_argument("--mono-tokenizer-cls")
    parser.add_argument("--mono-tokenizer-name-or-path", type=str, help="Number of sentences in each file input-file")
    parser.add_argument("--multi-tokenizer-cls")
    parser.add_argument("--multi-tokenizer-name-or-path", type=str, help="path to store sampled sentences")
    parser.add_argument("--output-dir", type=str, help="Output directory for computed statistics")

    args = parser.parse_args()
    return args


def get_tokenizer(cls: str, tokenizer_name_or_path: str) -> PreTrainedTokenizer:
    tokenizer_cls: Type[PreTrainedTokenizer] = getattr(transformers, cls)
    tokenizer: PreTrainedTokenizer = tokenizer_cls.from_pretrained(tokenizer_name_or_path)
    return tokenizer


# def get_iterator(input_file: str) -> Generator(str, None, None):
#     with open(input_file, "r") as f:
#         for line in f:
#             yield line.translate(
#                 str.maketrans({key: " {0} " for key in [".", ",", "!"]})
#             )


def read_sentences(stream: TextIO) -> Generator[List[str], None, None]:
    sent = []

    for line in stream:
        if not line.strip():
            if sent:    
                yield sent
            sent = []
        else:
            sent.append(line.rstrip("\n"))


def main():
    args = setup_args()
    mono_tokenizer = None
    multi_tokenizer = None

    if args.mono_tokenizer_name_or_path:
        mono_tokenizer = get_tokenizer(args.mono_tokenizer_cls, args.mono_tokenizer_name_or_path)
        mono_sentence_lengths = defaultdict(int)
        mono_token_lengths = defaultdict(int)

    if args.multi_tokenizer_name_or_path:
        multi_tokenizer = get_tokenizer(args.multi_tokenizer_cls, args.multi_tokenizer_name_or_path)
        multi_sentence_lengths = defaultdict(int)
        multi_token_lengths = defaultdict(int)

    counter = defaultdict(int)
    sentence_lengths = defaultdict(int)
    token_lengths = defaultdict(int)
    
    for sentence in read_sentences(args.input_file):
        mono_tokens = []
        multi_tokens = []

        for word in sentence:
            token = word.split(" ")[0]
            token_lengths[len(token)] += 1

            if mono_tokenizer:
                mono_tokens.extend(mono_tokenizer.tokenize(token))

            if multi_tokenizer:
                multi_tokens.extend(multi_tokenizer.tokenize(token))

        counter["num_sentences"] += 1
        counter["num_tokens"] += len(sentence)

        sentence_lengths[len(sentence)] += 1 

        if mono_tokenizer:
            counter["num_mono_tokens"] += len(multi_tokens)

            for t in mono_tokens:
                if t.startswith("_"):
                    t = t[1:]
                else:
                    counter["num_mono_continuation_tokens"] += 1
                mono_token_lengths[len(t)] += 1
            mono_sentence_lengths[len(mono_tokens)] += 1
            counter["mono_sentence_maxlen"] = max(len(mono_tokens), counter["mono_sentence_maxlen"])
        
        if multi_tokenizer:
            counter["num_multi_tokens"] += len(multi_tokens)

            for t in multi_tokens:
                if t.startswith("▁"):   # NOTE: This isn't underscore. Its ord is 9601. 
                    t = t[1:]
                else:
                    counter["num_multi_continuation_tokens"] += 1
                multi_token_lengths[len(t)] += 1

            multi_sentence_lengths[len(multi_tokens)] += 1
            counter["multi_sentence_maxlen"] = max(len(multi_tokens), counter["multi_sentence_maxlen"])

        counter["sentence_maxlen"] = max(len(sentence), counter["sentence_maxlen"])
    
    statistics = {
        "num_tokens": counter["num_tokens"],
        "num_sentences": counter["num_sentences"],
        "sentence_maxlen": counter["sentence_maxlen"],
    }

    output_dir = Path(args.output_dir)

    stats_file = output_dir / "summary_stats.json"
    token_lengths_file = output_dir / "token_lengths.json"
    sentence_lengths_file = output_dir / "sentence_lengths.json"

    json.dump(dict(token_lengths), token_lengths_file.open("w"), indent=4)
    json.dump(dict(sentence_lengths), sentence_lengths_file.open("w"), indent=4)

    if mono_tokenizer:
        mono_token_lengths_file = output_dir / "mono_token_lengths.json"
        mono_sentence_lengths_file = output_dir / "mono_sentence_lengths.json"

        statistics["mono_fertility"] = counter["num_mono_tokens"] / counter["num_tokens"]
        statistics["num_mono_tokens"] = counter["num_mono_tokens"]
        statistics["mono_sentence_maxlen"] = counter["mono_sentence_maxlen"]
        statistics["num_mono_continuation_tokens"] = counter["num_mono_continuation_tokens"]

        json.dump(dict(mono_token_lengths), mono_token_lengths_file.open("w"), indent=4)
        json.dump(dict(mono_sentence_lengths), mono_sentence_lengths_file.open("w"), indent=4)        
    
    if multi_tokenizer:
        multi_token_lengths_file = output_dir / "multi_token_lengths.json"
        multi_sentence_lengths_file = output_dir / "multi_sentence_lengths.json"

        statistics["multi_fertility"] = counter["num_multi_tokens"] / counter["num_tokens"]
        statistics["num_multi_tokens"] = counter["num_multi_tokens"]
        statistics["multi_sentence_maxlen"] = counter["multi_sentence_maxlen"]
        statistics["num_multi_continuation_tokens"] = counter["num_multi_continuation_tokens"]

        json.dump(dict(multi_token_lengths), multi_token_lengths_file.open("w"), indent=4)    
        json.dump(dict(multi_sentence_lengths), multi_sentence_lengths_file.open("w"), indent=4)

    pprint(statistics)
    json.dump(statistics, stats_file.open("w"), indent=4)


if __name__ == "__main__":
    main()
