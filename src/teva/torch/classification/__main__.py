from teva.torch.train import main
from teva.torch.arguments import ModelArguments
from teva.torch.classification.arguments import DataTrainingArguments
from teva.torch.classification.dataset import compute_classification_metrics, preprocess_function


if __name__ == "__main__":
    main(
        data_arguments=DataTrainingArguments,
        model_arguments=ModelArguments,
        preprocess_function=preprocess_function,
        compute_metrics_function=compute_classification_metrics
    )
