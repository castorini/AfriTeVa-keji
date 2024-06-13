import logging
from collections import OrderedDict
from typing import Iterator

import numpy as np
import torch
from datasets import Dataset as HFDataset, DatasetDict as HFDatasetDict
from torch.utils.data import (
    Dataset as TorchDataset,
    Sampler,
    RandomSampler,
    DistributedSampler,
    WeightedRandomSampler
)

logger = logging.getLogger(__name__)


class DataMixture(TorchDataset):
    def __init__(self, datasets: dict[str, HFDataset]):
        self.datasets = HFDatasetDict(datasets)
    
    def __len__(self):
        return sum(len(d) for d in self.datasets.values())

    def __getitem__(self, index) -> dict[str, torch.tensor]:
        language, idx = index
        item = self.datasets[language][idx]
        return item
    
    def map(self, *args, **kwargs):
        self.datasets = self.datasets.map(*args, **kwargs)
        return self


class MixtureSampler(Sampler):
    def __init__(
        self,
        batch_size: int,
        datasets: DataMixture,
        weights: list[float] = None,
        sampling_factor: int = 1.0
    ):
        self.datasets = datasets.datasets
        
        sampler_class = RandomSampler
        sampler_args = {"replacement": False}
        if torch.distributed.is_initialized():
            logger.info("Training is distributed. Switching to DistributedSampler.")
            sampler_class = DistributedSampler
            sampler_args = {"drop_last": True}

        self.samplers = {
            name: sampler_class(dataset, **sampler_args)
            for name, dataset in self.datasets.items()
        }
        self.configs = list(datasets.datasets.keys())

        if weights:
            assert len(datasets) == len(weights), \
                "You need to provide the same number of `datasets` and `weights`"
            self.weights = weights
        else:
            self.sampling_factor = sampling_factor
            self._set_dataset_weights()
        
        self.batch_size = batch_size

        self._batch = []
        self._batch_language = None
        self._create_language_sampler()

    def __iter__(self) -> Iterator[tuple[str, int]]:
        self._create_dataset_iters()
        while True:
            if not self._batch:
                # This raises StopIteration if all samples have been exhausted
                try:
                    self._batch = self._get_batch()
                except StopIteration:
                    break
            
            if self._batch:
                yield self._batch.pop()
    
    def __len__(self) -> int:
        return sum(len(sampler) for sampler in self.samplers.values())
    
    def _create_language_sampler(self):
        self.batch_language_sampler = WeightedRandomSampler(self.weights, len(self.weights), replacement=True)
        self.batch_language_iter = iter(self.batch_language_sampler)
    
    def _create_dataset_iters(self):
        self.iters = {
            name: iter(sampler)
            for name, sampler in self.samplers.items()
        }
    
    def _get_batch_language(self):
        if not self.iters:
            # We can create an infinite sampler by recreating samplers
            # self._create_config_samplers()
            raise StopIteration
        
        language = None
        if (self._batch_language is None) or (not self._batch):
            while language not in self.iters:
                try:
                    language = self.configs[next(self.batch_language_iter)]
                except StopIteration:
                    self._create_language_sampler()
                    language = self._get_batch_language()
        
        self._batch_language = language
        return self._batch_language

    def _get_batch(self):
        batch = []
        batch_language = self._get_batch_language()

        try:
            while len(batch) < self.batch_size:
                idx: int = next(self.iters[batch_language])
                batch.append((batch_language, idx))
        except StopIteration:
            print(f"Exhausted samples for language: {batch_language}")
            
            _ = self.iters.pop(batch_language)
            batch = self._get_batch()
        
        return batch

    def _set_dataset_weights(self):
        if self.sampling_factor == 0:
            # Use a uniform distribution
            self.weights = [
                1 / len(d) for _, d in self.datasets.values()
            ]
            return
        
        self.datasets = OrderedDict(self.datasets.items())

        total_num_examples = len(self)
        probs = np.array(
            [
                (len(d)/total_num_examples) ** self.sampling_factor
                for d in self.datasets.values()
            ]
        )

        self.weights = list(probs / probs.sum())

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

        for _, sampler in self.samplers.items():
            if hasattr(sampler, "set_epoch"):
                sampler.set_epoch(epoch)
    