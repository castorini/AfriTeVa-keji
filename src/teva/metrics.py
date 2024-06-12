import sacrebleu
from t5.evaluation.metrics import sklearn_metrics_wrapper


def chrf(targets, predictions):
    if isinstance(targets[0], list):
        targets = [[x for x in target] for target in targets]
    else:
        # Need to wrap targets in another list for corpus_bleu.
        targets = [targets]

    chrf_score = sacrebleu.corpus_chrf(
        predictions, targets,
    )
    return {"chrf": chrf_score.score}


def weighted_multiclass_f1(labels, **metric_fn_kwargs):
    """Computes the weighted average of the F1 per class."""
    return sklearn_metrics_wrapper(
        "f1_score",
        metric_dict_str="weighted_%dclass_f1" % len(labels),
        metric_post_process_fn=lambda x: 100 * x,
        labels=labels,
        average="weighted",
        **metric_fn_kwargs
    )
