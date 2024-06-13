import numpy as np
from evaluate import EvaluationModule, load
from transformers import EvalPrediction, PreTrainedTokenizer

from teva.torch.translation.arguments import DataTrainingArguments


def preprocess_function(
    examples: list[dict],
    data_args: DataTrainingArguments,
    tokenizer: PreTrainedTokenizer,
) -> dict[str, list[str]]:
    prefix = data_args.source_prefix if data_args.source_prefix is not None else ""

    inputs = [ex[data_args.source_lang] for ex in examples["translation"]]
    inputs = [prefix + inp for inp in inputs]

    targets = [ex[data_args.target_lang] for ex in examples["translation"]]

    padding = "max_length" if data_args.pad_to_max_length else False

    model_inputs = tokenizer(
        inputs,
        max_length=data_args.max_source_length,
        padding=padding,
        truncation=True,
    )
    
    labels = tokenizer(
        targets, max_length=data_args.max_target_length, padding=padding, truncation=True
    )

    # If we are padding here, replace all tokenizer.pad_token_id in the labels by -100 when we want to ignore
    # padding in the loss.
    if padding == "max_length" and data_args.ignore_pad_token_for_loss:
        labels["input_ids"] = [
            [(l if l != tokenizer.pad_token_id else -100) for l in label]
            for label in labels["input_ids"]
        ]

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs


def postprocess_text(preds: list[str], labels: list[str]) -> tuple[list[str]]:
    preds = [pred.strip() for pred in preds]
    labels = [[label.strip()] for label in labels]

    return preds, labels


def get_metrics() -> dict[str, EvaluationModule]:
    metrics_dict = {
        "sacrebleu": load("sacrebleu"),
        "chrf": load("chrf")
    }
    return metrics_dict


def compute_metrics(
    eval_preds: EvalPrediction,
    metrics: dict[str, EvaluationModule],
    tokenizer: PreTrainedTokenizer
) -> dict[str, float]:
    preds, labels = eval_preds
    if isinstance(preds, tuple):
        preds = preds[0]
    
    preds = np.where(preds != -100, preds, tokenizer.pad_token_id)
    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)

    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    # Some simple post-processing
    decoded_preds, decoded_labels = postprocess_text(decoded_preds, decoded_labels)
    
    results = {
        "bleu": metrics["sacrebleu"].compute(predictions=decoded_preds, references=decoded_labels)["score"],
        "chrf": metrics["chrf"].compute(predictions=decoded_preds, references=decoded_labels)["score"]
    }

    prediction_lens = [
        np.count_nonzero(pred != tokenizer.pad_token_id) for pred in preds
    ]
    results["average_generation_length"] = np.mean(prediction_lens)

    return results
