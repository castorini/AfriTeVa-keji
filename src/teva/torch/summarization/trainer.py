import random

import numpy as np
import torch
from datasets import Dataset as HFDataset
from torch.utils.data import DataLoader
from transformers import Seq2SeqTrainer

from teva.torch.dataset import MixtureSampler


def get_worker_shard(datasets: dict[str, HFDataset], worker_id: int, num_workers: int, seed: int):
    _datasets = {}
    for config, dataset in datasets.items():
        _datasets[config] = dataset.shard(num_shards=num_workers, index=worker_id, keep_in_memory=False)
    
    return _datasets


def worker_init_fn(worker_id: int) -> None:
    np.random.seed(np.random.get_state()[1][0] + worker_id + random.randint(1, 1000))
    worker_info = torch.utils.data.get_worker_info()
    worker_info.dataset.datasets = get_worker_shard(worker_id=worker_id, seed=worker_info.dataset.data_seed)        


class S2STrainer(Seq2SeqTrainer):
    def get_train_dataloader(self) -> DataLoader:
        train_sampler = MixtureSampler(
            batch_size=self.args.train_batch_size,
            datasets=self.train_dataset,
            weights=None,
            sampling_factor=self.args.sampling_factor
        )

        return DataLoader(
            dataset=self.train_dataset,
            batch_size=self.args.train_batch_size,
            sampler=train_sampler,
            collate_fn=self.data_collator,
            pin_memory=self.args.dataloader_pin_memory,
            worker_init_fn=worker_init_fn,
            num_workers=self.args.dataloader_num_workers,
            persistent_workers=self.args.dataloader_persistent_workers,
            prefetch_factor=self.args.dataloader_prefetch_factor
        )
