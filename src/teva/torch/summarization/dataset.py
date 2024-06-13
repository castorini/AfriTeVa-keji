import nltk
import numpy as np
from datasets import load_dataset, DatasetDict as HFDatasetDict
from evaluate import load, EvaluationModule
from transformers import EvalPrediction, PreTrainedTokenizer

from teva.torch.dataset import DataMixture
from teva.torch.summarization.arguments import DataTrainingArguments


def get_metrics():
    nltk.download("punkt")
    metric = load("rouge")
    return metric


def preprocess_function(
    examples: list[dict],
    data_args: DataTrainingArguments,
    tokenizer: PreTrainedTokenizer,
) -> dict[str, list[str]]:
    prefix = data_args.source_prefix if data_args.source_prefix is not None else ""

    inputs = [ex for ex in examples[data_args.text_column]]
    inputs = [prefix + inp for inp in inputs]

    targets = [ex for ex in examples[data_args.summary_column]]

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


def postprocess_text(preds, labels):
    preds = [pred.strip() for pred in preds]
    labels = [label.strip() for label in labels]

    # rougeLSum expects newline after each sentence
    preds = ["\n".join(nltk.sent_tokenize(pred)) for pred in preds]
    labels = ["\n".join(nltk.sent_tokenize(label)) for label in labels]

    return preds, labels


def compute_metrics(
    eval_preds: EvalPrediction,
    metric: EvaluationModule,
    tokenizer: PreTrainedTokenizer):
    preds, labels = eval_preds
    if isinstance(preds, tuple):
        preds = preds[0]
    
    # Replace -100s used for padding as we can't decode them
    preds = np.where(preds != -100, preds, tokenizer.pad_token_id)
    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)

    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    # Some simple post-processing
    decoded_preds, decoded_labels = postprocess_text(decoded_preds, decoded_labels)

    result = metric.compute(predictions=decoded_preds, references=decoded_labels, use_stemmer=True)
    result = {k: round(v * 100, 4) for k, v in result.items()}

    prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in preds]
    result["average_generation_length"] = np.mean(prediction_lens)

    return result


def dataset_provider(data_args: DataTrainingArguments):
    configs = data_args.dataset_config_name.split(",")
    if len(configs) > 1:
        ds = {
            config: load_dataset(data_args.dataset_name, config)
            for config in configs
        }

        train_ds = DataMixture(
            datasets = {
                config: ds[config]["train"]
                for config in ds
            }
        )

        dataset_dict = {
            "train": train_ds,
            "validation": HFDatasetDict({
                config: ds[config]["validation"]
                for config in ds
            }),
            "test": HFDatasetDict({
                config: ds[config]["test"]
                for config in ds
            })
        }
        return dataset_dict

    return load_dataset(data_args.dataset_name, data_args.dataset_config_name)
