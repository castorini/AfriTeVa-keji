from typing import Literal

from sklearn import metrics
from transformers import AutoTokenizer, EvalPrediction

from teva.torch.classification.arguments import DataTrainingArguments

MASAKHANEWS_LABELS = ["business", "entertainment", "health", "politics", "religion", "sports", "technology"]
MASAKHANEWS_LABELS_MAP = {idx: label for idx, label in enumerate(MASAKHANEWS_LABELS)}


def preprocess_function(
    examples: list[dict],
    data_args: DataTrainingArguments,
    tokenizer: AutoTokenizer,
):
    prefix = data_args.source_prefix if data_args.source_prefix is not None else ""
    padding = "max_length" if data_args.pad_to_max_length else False

    inputs = [prefix + (text.lower() if data_args.lowercase else text) for text in examples[data_args.task_config]]
    model_inputs = tokenizer(inputs, max_length=data_args.max_source_length, padding=padding, truncation=True)

    targets = [MASAKHANEWS_LABELS_MAP[label] for label in examples["label"]]
    labels = tokenizer(text_target=targets, max_length=2, padding="max_length")

    # If we are padding here, replace all tokenizer.pad_token_id in the labels by -100 when we want to ignore
    # padding in the loss.
    if padding == "max_length" and data_args.ignore_pad_token_for_loss:
        labels["input_ids"] = [
            [(l if l != tokenizer.pad_token_id else -100) for l in label] for label in labels["input_ids"]
        ]

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs


def compute_classification_metrics(p: EvalPrediction, tokenizer: AutoTokenizer):
    predictions, labels = p

    decoded_predictions = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    classification_report = metrics.classification_report(decoded_labels, decoded_predictions)
    accuracy = metrics.accuracy_score(decoded_labels, decoded_predictions)
    weighted_f1 = metrics.f1_score(decoded_labels, decoded_predictions, average="weighted")
    weighted_precision = metrics.precision_score(decoded_labels, decoded_predictions, average="weighted")
    weighted_recall = metrics.recall_score(decoded_labels, decoded_predictions, average="weighted")

    results = {
        "accuracy": accuracy,
        "classification_report": classification_report,
        "weighted_f1": weighted_f1,
        "weighted_precision": weighted_precision,
        "weighted_recall": weighted_recall
    }
    return results
