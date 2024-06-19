import copy
import functools
import logging
import os
import sys
from typing import Callable

import numpy as np
import torch
import transformers
from datasets import load_dataset
from transformers import (
    AutoConfig,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    GenerationConfig,
    HfArgumentParser,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    set_seed,
)
from transformers.trainer_utils import is_main_process
from transformers.training_args import ParallelMode

from teva.torch.arguments import DataTrainingArguments, ModelArguments
from teva.torch.utils import check_output_dir

logger = logging.getLogger(__name__)


def main(
    preprocess_function: Callable,
    compute_metrics_function: Callable,
    training_arguments: type[Seq2SeqTrainingArguments] = Seq2SeqTrainingArguments,
    model_arguments: type[ModelArguments] = ModelArguments,
    data_arguments: type[DataTrainingArguments] = DataTrainingArguments,
    dataset_provider = None
):
    parser = HfArgumentParser((model_arguments, data_arguments, training_arguments))

    # HF expects local_rank but torch.distributed.launch passed local-rank
    if sys.argv[1].startswith("--local-rank"):
        sys.argv[1] = f"--local_rank={sys.argv[1].split('=')[-1]}"
    
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    
    training_args: Seq2SeqTrainingArguments

    is_multi_config_training = len(data_args.dataset_config_name.split(",")) > 1
    logger.info("Multi-dataset-configuration training enabled")
    if is_multi_config_training:
        assert dataset_provider is not None
    
    check_output_dir(training_args)

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    logger.warning(
        "Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
        training_args.local_rank,
        training_args.device,
        training_args.n_gpu,
        bool(training_args.parallel_mode == ParallelMode.DISTRIBUTED),
        training_args.fp16,
    )
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # Set the verbosity to info of the Transformers logger (on main process only):
    if is_main_process(training_args.local_rank):
        transformers.utils.logging.set_verbosity_info()
    
    logger.info("Training/evaluation parameters %s", training_args)

    set_seed(training_args.seed)

    if dataset_provider is None:
        ds = load_dataset(data_args.dataset_name, data_args.dataset_config_name)
    else:
        ds = dataset_provider(data_args)

    config = AutoConfig.from_pretrained(
        model_args.config_name if model_args.config_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        token=model_args.token,
        trust_remote_code=model_args.trust_remote_code,
    )
    
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        use_fast=model_args.use_fast_tokenizer,
        revision=model_args.model_revision,
        token=model_args.token,
        trust_remote_code=model_args.trust_remote_code,
    )
    model = AutoModelForSeq2SeqLM.from_pretrained(
        model_args.model_name_or_path,
        from_tf=".ckpt" in model_args.model_name_or_path,
        config=config,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        token=model_args.token,
        trust_remote_code=model_args.trust_remote_code,
    )

    # TODO: @theyorubayesian - Figure out a better way to pass Generation Arguments
    # Note: Only num_beams and max_length can be passed using TrainingArguments
    # Also see: https://github.com/huggingface/transformers/issues/25917
    generation_config = None
    if hasattr(training_args, "length_penalty"):
        generation_config: GenerationConfig = copy.deepcopy(model.generation_config)
        generation_config.num_beams = training_args.generation_num_beams
        generation_config.length_penalty = training_args.length_penalty
        generation_config.max_length = training_args.generation_max_length
        training_args.generation_config=generation_config

    if not any([training_args.do_train, training_args.do_eval, training_args.do_predict]):
        logger.info("There is nothing to do. Please pass `do_train`, `do_eval` and/or `do_predict`.")
        return

    if training_args.do_train:
        if "train" not in ds:
            raise ValueError("--do_train requires a train dataset")
        try:
            column_names = ds["train"].column_names
        except AttributeError:
            # ds["train"] is a DataMixture
            sample_config = data_args.dataset_config_name.split(",")[0]
            column_names = ds["train"].datasets[sample_config].column_names
    
    if training_args.do_eval:
        if "validation" not in ds:
            raise ValueError("--do_eval requires a validation dataset")
        try:
            column_names = ds["validation"].column_names
            if isinstance(column_names, dict):
                column_names = list(column_names.values())[0]
        except AttributeError:
            sample_config = data_args.dataset_config_name.split(",")[0]
            column_names = ds["validation"][sample_config].column_names
        
    
    if training_args.do_predict:
        if "test" not in ds:
            raise ValueError("--do_predict requires a test dataset")
        try:
            column_names = ds["test"].column_names
            if isinstance(column_names, dict):
                column_names = list(column_names.values())[0]
        except AttributeError:
            sample_config = data_args.dataset_config_name.split(",")[0]
            column_names = ds["test"][sample_config].column_names

    if training_args.do_train:
        train_dataset = ds["train"]
        
        if data_args.max_train_samples is not None:
            max_train_samples = min(len(train_dataset), data_args.max_train_samples)
            train_dataset = train_dataset.select(range(max_train_samples))

        with training_args.main_process_first("Train dataset pre-processing"):
            train_dataset = train_dataset.map(
                functools.partial(preprocess_function, tokenizer=tokenizer, data_args=data_args),
                batched=True,
                num_proc=data_args.preprocessing_num_workers,
                remove_columns=column_names,
                load_from_cache_file=not data_args.overwrite_cache,
                desc="Tokenizing train dataset"
            )
    
    if training_args.do_eval:
        eval_dataset = ds["validation"]

        if data_args.max_eval_samples is not None:
            max_eval_samples = min(len(eval_dataset), data_args.max_eval_samples)
            eval_dataset = eval_dataset.select(range(max_eval_samples))
        
        with training_args.main_process_first(desc="Validation dataset pre-processing"):
            eval_dataset = eval_dataset.map(
                functools.partial(preprocess_function, tokenizer=tokenizer, data_args=data_args),
                batched=True,
                num_proc=data_args.preprocessing_num_workers,
                remove_columns=column_names,
                load_from_cache_file=not data_args.overwrite_cache,
                desc="Tokenizing validation dataset",
            )

    if training_args.do_predict:
        predict_dataset = ds["test"]

        if data_args.max_predict_samples is not None:
            max_predict_samples = min(len(predict_dataset), data_args.max_predict_samples)
            predict_dataset = predict_dataset.select(range(max_predict_samples))
        
        with training_args.main_process_first(desc="Prediction dataset pre-processing"):
            predict_dataset = predict_dataset.map(
                functools.partial(preprocess_function, tokenizer=tokenizer, data_args=data_args),
                batched=True,
                num_proc=data_args.preprocessing_num_workers,
                remove_columns=column_names,
                load_from_cache_file=not data_args.overwrite_cache,
                desc="Tokenizing prediction dataset",
            )
    
    label_pad_token_id = -100 if data_args.ignore_pad_token_for_loss else tokenizer.pad_token_id
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        label_pad_token_id=label_pad_token_id,
        pad_to_multiple_of=8 if training_args.fp16 else None,
    )

    training_args.generation_max_length = (
        training_args.generation_max_length
        if training_args.generation_max_length is not None
        else data_args.val_max_target_length
    )

    # Normally, the forward pass of the model returns loss and logits but we need tokens for the our metrics
    # predict_with_generate wraps generate() to handle this. 
    trainer_cls = Seq2SeqTrainer

    # TODO: Figure a better way to indicate multi-config training
    if is_multi_config_training:
        from teva.torch.summarization.trainer import S2STrainer as Trainer
        trainer_cls = Trainer
    
    trainer = trainer_cls(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=eval_dataset if training_args.do_eval else None,
        data_collator=data_collator,
        compute_metrics=(
            functools.partial(compute_metrics_function, tokenizer=tokenizer) 
            if training_args.predict_with_generate else None
        )
    )

    # Training
    if training_args.do_train:
        train_result = trainer.train()
        trainer.save_model()

        metrics = train_result.metrics
        max_train_samples = (
            data_args.max_train_samples if data_args.max_train_samples is not None else len(train_dataset)
        )
        metrics["train_samples"] = min(max_train_samples, len(train_dataset))

        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()

    # Evaluation
    results = {}
    if training_args.do_eval:
        logger.info("*** Evaluate ***")
        if isinstance(eval_dataset, dict):
            metrics = {}
            for eval_ds_name, eval_ds in eval_dataset.items():
                dataset_metrics = trainer.evaluate(eval_dataset=eval_ds, metric_key_prefix=f"eval_{eval_ds_name}")
                metrics.update(dataset_metrics)
        else:
            metrics = trainer.evaluate(metric_key_prefix="eval")
        max_eval_samples = data_args.max_eval_samples if data_args.max_eval_samples is not None else len(eval_dataset)
        metrics["eval_samples"] = min(max_eval_samples, len(eval_dataset))

        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

    if training_args.do_predict:
        if not isinstance(predict_dataset, dict):
            predict_dataset = {"predict": predict_dataset}
        
        if isinstance(predict_dataset, dict):
            predict_dataset = {f"predict_{ds_name}": ds for ds_name, ds in predict_dataset.items()}
        
        for predict_ds_name, predict_ds in predict_dataset.items():
            predict_results = trainer.predict(predict_ds, metric_key_prefix=predict_ds_name)
            
            max_predict_samples = (
                data_args.max_predict_samples if data_args.max_predict_samples is not None else len(predict_ds)
            )
            max_predict_samples = min(max_predict_samples, len(predict_ds))

            metrics = predict_results.metrics
            metrics["predict_samples"] = max_predict_samples
        
            trainer.log_metrics(predict_ds_name, metrics)
            trainer.save_metrics(predict_ds_name, metrics)

            if trainer.is_world_process_zero():
                if training_args.predict_with_generate:
                    predictions = predict_results.predictions
                    predictions = np.where(predictions != -100, predictions, tokenizer.pad_token_id)
                    predictions = tokenizer.batch_decode(
                        predictions, skip_special_tokens=True, clean_up_tokenization_spaces=True
                    )
                    predictions = [pred.strip() for pred in predictions]

                    output_prediction_file = os.path.join(
                        training_args.output_dir,
                        f"{predict_ds_name.replace('predict_', '')}_generated_predictions.txt"
                    )
                    
                    with open(output_prediction_file, "w") as writer:
                            writer.write("\n".join(predictions))
