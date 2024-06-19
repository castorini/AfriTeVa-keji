import random
from typing import Optional

import numpy as np
import torch.utils.data
from datasets import Dataset as HFDataset
from torch.utils.data import DataLoader
from transformers import Seq2SeqTrainer
from transformers.trainer_utils import has_length

from teva.torch.dataset import MixtureSampler


def get_worker_shard(
    datasets: dict[str, HFDataset],
    worker_id: int,
    num_workers: int,
    seed: int
) -> dict[str, HFDataset]:
    _datasets = {}
    for config, dataset in datasets.items():
        _datasets[config] = dataset.shard(
            num_shards=num_workers, index=worker_id, keep_in_memory=False)
    
    return _datasets


def worker_init_fn(worker_id: int) -> None:
    np.random.seed(np.random.get_state()[1][0] + worker_id + random.randint(1, 1000))
    worker_info = torch.utils.data.get_worker_info()
    worker_info.dataset.datasets = get_worker_shard(
        worker_id=worker_id, seed=worker_info.dataset.data_seed)        


class S2STrainer(Seq2SeqTrainer):
    def _get_train_sampler(self) -> Optional[torch.utils.data.Sampler]:
        if self.train_dataset is None or not has_length(self.train_dataset):
            return None
        
        # Note that train_batch_size is only used to ensure we 
        # sample the entire batch from a single dataset config
        # MixtureSampler yields one index at a time and is 
        # wrapped by a BatchSampler provided by torch.utils.data.DataLoader
        # In Distributed Training, BatchSampler is sharded so that each process
        # receives a different index of the train_dataset
        train_sampler = MixtureSampler(
            batch_size=self._train_batch_size,
            datasets=self.train_dataset,
            weights=None,
            sampling_factor=self.args.sampling_factor
        )
        return train_sampler

    def get_train_dataloder(self) -> DataLoader:
        """
        This implementation changes the worker_init_fn
        """
        if self.train_dataset is None:
            raise ValueError("Trainer: training requires a train_dataset.")

        train_dataset = self.train_dataset
        data_collator = self.data_collator
        data_collator = self._get_collator_with_removed_columns(data_collator, description="training")

        dataloader_params = {
            "batch_size": self._train_batch_size,
            "collate_fn": data_collator,
            "num_workers": self.args.dataloader_num_workers,
            "pin_memory": self.args.dataloader_pin_memory,
            "persistent_workers": self.args.dataloader_persistent_workers,
        }

        if not isinstance(train_dataset, torch.utils.data.IterableDataset):
            dataloader_params["sampler"] = self._get_train_sampler()
            dataloader_params["drop_last"] = self.args.dataloader_drop_last
            dataloader_params["worker_init_fn"] = worker_init_fn
            dataloader_params["prefetch_factor"] = self.args.dataloader_prefetch_factor

        return self.accelerator.prepare(DataLoader(train_dataset, **dataloader_params))
