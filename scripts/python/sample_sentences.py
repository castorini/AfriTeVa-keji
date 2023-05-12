import argparse
import random
import logging
from pathlib import Path
from pprint import pprint
from typing import Dict
from typing import TextIO

from tqdm import tqdm

logger = logging.getLogger(__name__)


def setup_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sample Sentences from monolingual corpora to train tokenizer")
    parser.add_argument(
        "--input-files",
        type=str,
        nargs="+",
        help="Path to one of more text files"
    )
    parser.add_argument("--n-sentences", type=int, nargs="+", help="Number of sentences in each file input-file")
    parser.add_argument("--output-file", type=str, required=True, help="path to store sampled sentences")
    parser.add_argument("--alpha", type=float, default=0.3, help="multinomial alpha")
    parser.add_argument("--seed", type=int, default=10, help="random seed")

    args = parser.parse_args()

    # ------------
    # Sanity Check
    # ------------
    assert len(args.input_files) == len(args.n_sentences), \
        "`input-files` and `n_sententeces must be of same length"
    # ------------

    return args


def calc_num_sampled_sentences(
    lang_num_lines: Dict[str, int], 
    alpha: float
) -> Dict[str, int]:
    """
    Calculate number of lines to sample per language
    See https://arxiv.org/abs/1901.07291 : Section 3.1 Shared sub-word vocabulary
    """
    lang_prob: Dict[str, float] = {}

    total_n_sentences = sum(lang_num_lines.values())
    lang_prob = {
        language: n_sentences / total_n_sentences
        for language, n_sentences in lang_num_lines.items()
    }

    total_distr = sum([k**alpha for k in lang_prob.values()])

    sampling_prob = {k: v**alpha / total_distr for k, v in lang_prob.items()}
    pprint(f"Sampling probability: {sampling_prob}")
    
    n_sampled_sentences = {
        language: round(sampling_prob[language] * lang_num_lines[language])
        for language in lang_num_lines.keys()
    }

    return n_sampled_sentences


def sample_sentences(n_sample: int, n_total: int, input_file: TextIO, output_file: TextIO):
    j = 0
    sampled_lines = sorted(random.sample(range(n_total), n_sample))
    
    for idx, line in enumerate(input_file):
        if (j < n_sample) and (idx == sampled_lines[j]):
            output_file.write(line)
            j += 1
        elif (j >= n_sample):  # Early exit
            break


def main():
    args = setup_args()

    random.seed(args.seed)
    print("***** Sampling Sentences for Tokenizer Training *****")

    n_sentences_map = {
        file: n_sentence
        for file, n_sentence in zip(args.input_files, args.n_sentences)
    }

    n_sampled_sentences = calc_num_sampled_sentences(n_sentences_map, args.alpha)

    with open(args.output_file, "w") as outfile:
        for file, n_sentences_to_sample in tqdm(n_sampled_sentences.items()):
            print(f"Number of sampled sentences for {file} = {n_sentences_to_sample}")
            lang_file = open(file, "r")
            sample_sentences(n_sentences_to_sample, n_sentences_map[file], lang_file, outfile)


if __name__ == "__main__":
    main()
