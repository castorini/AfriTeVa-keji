import functools
import logging

from teva.torch.train import main
from teva.torch.arguments import ModelArguments
from teva.torch.summarization.arguments import DataTrainingArguments, SummarizationTrainingArguments
from teva.torch.summarization.dataset import compute_metrics, dataset_provider, get_metrics, preprocess_function

logger = logging.getLogger(__name__)


if __name__ == "__main__":
    main(
        dataset_provider=dataset_provider,
        preprocess_function=preprocess_function,
        compute_metrics_function=functools.partial(compute_metrics, metric=get_metrics()),
        training_arguments=SummarizationTrainingArguments,
        data_arguments=DataTrainingArguments,
        model_arguments=ModelArguments,
    )
