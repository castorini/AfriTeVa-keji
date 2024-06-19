import logging
from collections import OrderedDict

import numpy as np
import torch
from datasets import (
    Dataset as HFDataset,
    IterableDataset as IterableHFDataset,
    DatasetDict as HFDatasetDict,
    IterableDatasetDict as IterableHFDatasetDict
)
from torch.utils.data import (
    Dataset as TorchDataset,
    IterableDataset as IterableTorchDataset,
    Sampler,
    RandomSampler
)

logger = logging.getLogger(__name__)


def _maybe_add_torch_dataset_parent_class(cls, torch_dataset_class: str):
    """Add torch.utils.data dataset class as a parent class if 'torch' is available"""
    try:
        import torch.utils.data
        torch_dataset_cls = getattr(torch.utils.data, torch_dataset_class)
        cls.__bases__ += (torch_dataset_cls,)
        # print(cls.__bases__)
    except ImportError:
        return
    except AttributeError:
        raise


class SamplerMixin:
    def __init__(
        self,
        datasets,
        batch_size: int,
        weights: list[float] = None,
        sampling_factor: int = 1.0
    ):
        if weights:
            assert len(datasets) == len(weights), \
                "You need to provide the same number of `datasets` and `weights`"
            self.weights = weights
        else:
            self.sampling_factor = sampling_factor
            self._set_dataset_weights()

        self.batch_size = batch_size
        self.datasets = OrderedDict(self.datasets.items())
        self.configs = list(self.datasets.keys())
        
        self._batch = []
        self._batch_language = None
        
    def __iter__(self):
        self._create_dataset_iters()
        while True:
            if not self._batch:
                # This raises StopIteration if all samples have been exhausted
                # self._batch = self._get_batch()
                try:
                    self._batch = self._get_batch()
                except StopIteration:
                    break
            
            if self._batch:
                yield self._batch.pop()
    
    def _create_dataset_iters(self):
        self._iters = {
            name: iter(sampler_or_dataset)
            for name, sampler_or_dataset in self._sampler_or_dataset_dict.items()
        }

    def _get_batch_language(self):
        if not self._iters:  
            raise StopIteration
        
        if (self._batch_language is None) or (not self._batch):
            language = None
            while language not in self._iters:
                language_idx = np.argmax(np.random.multinomial(1, self.weights))
                language = self.configs[language_idx]
            
            self._batch_language = language
        return self._batch_language

    def _get_batch(self):
        batch = []
        batch_language = self._get_batch_language()

        while len(batch) < self.batch_size:  
            try:  
                while len(batch) < self.batch_size:  
                    idx: int = next(self._iters[batch_language])  
                    batch.append((batch_language, idx))  
            except StopIteration:
                logger.info(f"Exhausted samples for language: {batch_language}")  
                
                _ = self._iters.pop(batch_language)
                batch = []
                batch_language = self._get_batch_language()
        return batch

    def _set_dataset_weights(self):
        if self.sampling_factor == 0:
            # Use a uniform distribution
            self.weights = [
                1 / len(d) for _, d in self.datasets.values()
            ]
            return

        total_num_examples = len(self)
        probs = np.array(
            [
                (len(d)/total_num_examples) ** self.sampling_factor
                for d in self.datasets.values()
            ]
        )

        self.weights = list(probs / probs.sum())
    
    def set_epoch(self, epoch: int) -> None:
        for _, ds in self.datasets.items():
            ds.set_epoch(epoch)


class DataMixture(TorchDataset):
    def __init__(self, datasets: dict[str, HFDataset]):
        # Ensure all datasets are of same class
        assert all(isinstance(ds, HFDataset) for ds in datasets.values())
        self.datasets = HFDatasetDict(datasets)

        # _maybe_add_torch_dataset_parent_class(self.__class__, "Dataset")
    
    def __len__(self):
        return sum(len(d) for d in self.datasets.values())

    def __getitem__(self, index) -> dict[str, torch.tensor]:
        language, idx = index
        item = self.datasets[language][idx]
        return item
    
    def map(self, *args, **kwargs):
        self.datasets = self.datasets.map(*args, **kwargs)
        return self
    
    def to_iterable_data_mixture(
        self,
        batch_size: int,
        num_shards: int = None,
        weights: list[float] = None,
        sampling_factor: float = 1.0
    ):
        return IterableDataMixture(
            self.datasets,
            batch_size=batch_size,
            num_shards=num_shards,
            weights=weights,
            sampling_factor=sampling_factor
        )


class IterableDataMixture(SamplerMixin, IterableTorchDataset):
    def __init__(
        self, 
        datasets: dict[str, IterableHFDataset],
        batch_size: int,
        weights: list[float] = None,
        sampling_factor: int = 1.0
    ):
        # Ensure all datasets are of same class
        assert all(isinstance(ds, IterableHFDataset) for ds in datasets.values())
        self.datasets = IterableHFDatasetDict(datasets)
        self._sampler_or_dataset_dict = self.datasets

        SamplerMixin.__init__(self, datasets, batch_size, weights, sampling_factor)
        # _maybe_add_torch_dataset_parent_class(self.__class__, "IterableDataset")
    
    def __iter__(self):
        return SamplerMixin.__iter__(self)
    
    def map(self, *args, **kwargs):
        self.datasets = self.datasets.map(*args, **kwargs)
        return self


class MixtureSampler(Sampler, SamplerMixin):
    def __init__(
        self,
        batch_size: int,
        datasets: DataMixture,
        weights: list[float] = None,
        sampling_factor: int = 1.0
    ):
        self.datasets = datasets.datasets

        self.samplers = {
            name: RandomSampler(dataset, replacement=False)
            for name, dataset in self.datasets.items()
        }
        self._sampler_or_dataset_dict = self.samplers

        SamplerMixin.__init__(self, datasets, batch_size, weights, sampling_factor)
    
    def __iter__(self):
        return SamplerMixin.__iter__(self)
    
    def __len__(self) -> int:
        return sum(len(sampler) for sampler in self.samplers.values())
    
    def set_epoch(self, epoch: int) -> None:
        # Seems to only be used for DistributedSampler
        # To ensure randomness across epochs
        # https://discuss.pytorch.org/t/why-is-sampler-set-epoch-epoch-needed-for-distributedsampler/149672
        self.epoch = epoch

        for _, sampler in self.samplers.items():
            if hasattr(sampler, "set_epoch"):
                sampler.set_epoch(epoch)
