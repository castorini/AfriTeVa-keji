import argparse
import json
import os
import statistics
from collections import defaultdict


def main():
    parser = argparse.ArgumentParser(description="Collate results across all seeds")
    parser.add_argument("--results-dir", default="runs/classification/afriteva_v2_base")
    parser.add_argument("--results-file", default="predict_results.json")
    parser.add_argument("--language", type=str)
    parser.add_argument("--n-seeds", type=int, default=5)
    parser.add_argument("--output-file", default="runs/classification/results/afriteva_v2_base/amh.json")

    args = parser.parse_args()
    factory = lambda: dict(values=[])
    results = defaultdict(factory)

    keys = ["predict_accuracy", "predict_weighted_f1", "predict_loss", "predict_weighted_precision", "predict_weighted_recall"]
    
    for i in range(1, args.n_seeds+1):
        seed_dir = args.language + f"_{i}"
        
        fp = open(os.path.join(args.results_dir, seed_dir, args.results_file), "r")
        result = json.load(fp)

        for key in keys:
            results[key]["values"].append(result[key])
    
    for k in keys:
        mean = statistics.mean(results[k]["values"])
        std = statistics.stdev(results[k]["values"])
        results[k]["summary"] = [mean, std]

    with open(args.output_file, "w") as f:
        json.dump(results, f, indent=4)


if __name__ == "__main__":
    main()
