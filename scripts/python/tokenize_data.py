import argparse
import logging
from collections import Counter
from contextlib import nullcontext
from pathlib import Path
from typing import Generator
from typing import Optional
from typing import TextIO
from typing import Tuple
from typing import Type

import pandas as pd
import transformers
from nltk import download
from nltk.tokenize import sent_tokenize
from numpy import isnan
from transformers import PreTrainedTokenizer

MAX_LENGTH = 512
MIN_NUM_TOKENS = 5

logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - PID: %(process)d -  %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
logger = logging.getLogger(__name__)


def setup_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--data-path", type=str, help="Path to data file to tokenize")
    parser.add_argument("--language", type=str, help="ISO Code for language")
    parser.add_argument("--output-file", type=str, help="Path to output file")
    parser.add_argument("--write-output", action="store_true", help="If True, write output to file")
    parser.add_argument(
        "--chunk",
        default="subword", 
        choices=["subword", "sentence", "sized-chunks"],
        help="Chunk dataset into sentences or subwords. Tokenizer arguments must be passed when chunk is `subword`")
    parser.add_argument("--chunk-size", type=int, default=512, help="Create chunks of specified size")
    parser.add_argument("--allow-doc-overlap", action="store_true", help="If True, chunks from different documents can overlap")

    tokenizer_args = parser.add_argument_group("Tokenizer arguments")
    tokenizer_args.add_argument("--tokenizer-class", default="T5Tokenizer")
    tokenizer_args.add_argument("--tokenizer-name-or-path", default="EleutherAI/gpt-neo-1.3B")
    tokenizer_args.add_argument("--min-num-tokens", type=int, default=MIN_NUM_TOKENS, help="Min number of tokens per line in dataset.")
    tokenizer_args.add_argument("--max-length", type=int, default=MAX_LENGTH, help="Model max length for tokenizer")

    args = parser.parse_args()
    return args


def get_tokenizer(tokenizer_class: str, tokenizer_name_or_path, max_length: int) -> PreTrainedTokenizer:
    tokenizer_class: Type[PreTrainedTokenizer] = getattr(transformers, tokenizer_class)
    tokenizer: PreTrainedTokenizer = tokenizer_class.from_pretrained(tokenizer_name_or_path)
    tokenizer.model_max_length = max_length

    logger.info(f"Tokenizer loaded from {tokenizer_name_or_path}")
    logger.info(f"Number of tokens in tokenizer: {tokenizer.vocab_size}")
    return tokenizer


def get_iterator(filename: str) -> Generator[str, None, None]:
    file = Path(filename)
    extension = file.suffix

    if extension == ".jsonl":
        f = pd.read_json(filename, lines=True)
    elif extension == ".tsv":
        f = pd.read_csv(filename, sep="\t")
    elif extension == ".csv":
        f = pd.read_csv(filename)
    elif extension == ".txt":
        with open(filename, "r") as f:
            for line in f:
                yield line
        return
    else:
        raise ValueError("Only `.jsonl`, `csv` and `.tsv` are handled")
    
    for _, line in f.iterrows():
        if line["headline"] == line["content"]:             # Empty Wikipedia articles
            yield line["headline"]
        elif isnan(line["headline"]):                       # Headline may be null
             yield line["content"]
        elif line["headline"] and line["content"]:
            if line["content"].startswith(line["headline"]):    # Content may start with headline 
                yield line["content"]
            else:
                yield " ".join([line["headline"], line["content"]])
        elif line["headline"]:
            yield line["headline"]
        elif line["content"]:
            yield line["content"]
        else:
            continue


def tokenize_dataset(
    data: Path,
    tokenizer: PreTrainedTokenizer,
    min_num_tokens: int,
    write_output: bool = True,
    output_file: Path = None
) -> Tuple[int, Counter]:
    n_total_tokens = 0
    token_counter = Counter()
    
    for line in data.open("r", encoding="utf-8"):
        if (len(line.split()) > min_num_tokens and not line.isspace()):
            tokens = tokenizer.tokenize(line)
            n_total_tokens += len(tokens)
            token_counter.update(tokens)

    tokens = sorted(
        token_counter.items(), 
        key=lambda item: (item[1], item[0]),
        reverse=True
    )
    
    if write_output:
        with open(output_file, "w") as out:
            out.writelines([f'{token}\t{count}\n' for token, count in tokens])

    logger.info(f"Number of unique tokens in {data.name}: {len(tokens)}")
    logger.info(f"Total number of tokens in {data.name}: {n_total_tokens}")
    return n_total_tokens, token_counter


def sentencize_dataset(
    data: Path,
    min_num_tokens: int,
    write_output: bool = True,
    output_file: Path = None,
    *args,
    **kwargs
) -> int:
    n_sentences = 0
    
    with output_file.open("a", encoding="utf-8") if write_output else nullcontext() as f:

        # for line in data.open("r", encoding="utf-8"):
        for line in get_iterator(data.as_posix()):
            if (len(line.split()) > min_num_tokens and not line.isspace()):
                sentences = sent_tokenize(line)
                n_sentences += len(sentences)

                if f:
                    _ = [f.write(sentence + "\n") for sentence in sentences]

    return n_sentences


def _check_alpha_content(s: str) -> bool:
    numeric_content = sum(k.isdigit() for k in s) / len(s)
    return numeric_content < 0.2
  

def _clean_string(s: str) -> str:
    cs = s.replace("<nowiki>", "") \
        .replace("</nowiki>", "") \
        .replace('()', '')
    
    return cs


def validate(s: str) -> Optional[str]:
    words = s.split()

    if len(set(words)) <= 3:
        print("Skipping sentence: Not enough unique words")
        return None
    
    if not _check_alpha_content(s):
        print("Skipping sentence: Not enough alpha content")
        return None
    
    validated_string = " ".join([word.strip() for word in words if not word.isspace()])
    return _clean_string(validated_string)


def _write(f: TextIO, s: str) -> None:
    global last_s
    
    clean_string = validate(s)

    if clean_string:
        if not clean_string == last_s:
            f.write(clean_string + "\n")
            last_s = clean_string
        else:
            print("Skipping sentence: Duplicate")
            print(f"Sentence A:\n{last_s}\nSentence B:\n{clean_string}\n")


def chunk_dataset(
    data: Path,
    tokenizer: PreTrainedTokenizer,
    chunk_size: int,
    output_file: Path,
    language: str = "english",
    allow_overlap: float = 0.4,
    within_document: bool = True,
    *args,
    **kwargs
) -> None:
    """
    With the assumption that each line is a document, 
    we greedily chunk each document into the specified chunk-size,
    backtracking whenever 
    """
    with output_file.open("w", encoding="utf-8") as f:
        for line in get_iterator(data.as_posix()):
            sentences = sent_tokenize(line, language=language)
            token_lengths = [len(tokenizer.tokenize(sentence)) for sentence in sentences]
            
            start = 0
            subtotal = 0
            for i, length in enumerate(token_lengths):
                subtotal += length
                if subtotal >= chunk_size:
                    if subtotal / chunk_size >= 1.2:
                        end = i
                    else:
                        end = i+1

                    _write(f, " ".join(sentences[start:end]))

                    start = end
                    subtotal = 0
                
                if i+1 == len(token_lengths):
                    # If `allow_overlap` > 0, add the last sentence from the previous chunk 
                    # iff the sentence does not take up more than defined fraction of `chunk_size`
                    if (allow_overlap > 0) and ((token_lengths[start-1] / chunk_size) <= allow_overlap):
                        start -= 1
                    
                    _write(f, " ".join(sentences[start:]))


chunk_function_factory = {
    "subword": tokenize_dataset,
    "sentence": sentencize_dataset,
    "sized-chunks": chunk_dataset
}


def main():
    args = setup_args()

    global last_s

    last_s = ""

    data = Path(args.data_path)
    output_file = Path(args.output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    if args.chunk in ["sentence", "sized-chunks"]:
        download("punkt")
    
    tokenizer = None
    if args.chunk in ["subword", "sized-chunks"]:
        tokenizer = get_tokenizer(args.tokenizer_class, args.tokenizer_name_or_path, args.max_length)
    
    within_document = ~args.allow_doc_overlap
    chunk_func = chunk_function_factory[args.chunk]
    output = chunk_func(
        data=data,
        tokenizer=tokenizer,
        min_num_tokens=args.min_num_tokens,
        write_output=args.write_output,
        output_file=output_file,
        chunk_size=args.chunk_size,
        within_document=within_document
    )


if __name__ == "__main__":
    main()
