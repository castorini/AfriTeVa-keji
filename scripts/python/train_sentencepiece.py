import argparse

import sentencepiece as spm

SPECIAL_TOKENS_ID = {
    "BOS": -1,
    "PAD": 0,
    "EOS": 1,
    "UNK": 2,
}
NORMALIZATION_RULE_NAME = "nmt_nfkc"


def setup_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True, help="Path of input file")
    parser.add_argument("--model_prefix", type=str, required=True, help="Name of model prefix")
    parser.add_argument("--vocab_size", type=int, default=8004, help="Vocabulary size of subword model")
    parser.add_argument("--max_sent_len", type=int, default=100000, help="Max sentence length in bytes")
    parser.add_argument("--character_coverage", type=float, default=0.9995, help="Character coverage")
    parser.add_argument("--lowercase", action="store_true", help="Lowercase while learning subwords")
    parser.add_argument("--split_digits", type=bool, default=True, help="split all digits (0-9) into separate pieces")
    parser.add_argument("--byte_fallback", type=bool, default=True, help="decompose unknown pieces into UTF-8 byte pieces")
    parser.add_argument("--split_by_whitespace", type=bool, default=True, help="Split pieces by whitespace")

    args = parser.parse_args()
    return args


def train_tokenizer(args: argparse.Namespace):
    # This normalization rule takes care of case folding 
    # (https://github.com/google/sentencepiece/blob/master/doc/normalization.md)
    normalization_rule_name = "nmt_nfkc_cf" if args.lowercase else NORMALIZATION_RULE_NAME
    spm.SentencePieceTrainer.Train(
        input=args.input,
        model_prefix=args.model_prefix,
        vocab_size=args.vocab_size,
        normalization_rule_name=normalization_rule_name,
        character_coverage=args.character_coverage,
        max_sentence_length=args.max_sent_len,
        byte_fallback=args.byte_fallback,
        split_by_whitespace=args.split_by_whitespace,
        split_digits=args.split_digits,
        bos_id=SPECIAL_TOKENS_ID["BOS"],
        eos_id=SPECIAL_TOKENS_ID["EOS"],
        pad_id=SPECIAL_TOKENS_ID["PAD"],
        unk_id=SPECIAL_TOKENS_ID["UNK"],
        train_extremely_large_corpus=True
    )


if __name__ == "__main__":
    args = setup_args()
    train_tokenizer(args)
