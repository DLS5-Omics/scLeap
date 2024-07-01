import os
import random
import itertools

import numpy as np
import torch
from torch.utils.data import DataLoader

from common import config as cfg
from .dataset import TrainDataset


class TrainSampler:
    def __init__(self):
        pass

    def __iter__(self):
        while True:
            yield np.random.randint(0, 998244353)


class TrainLoader(DataLoader):
    def __init__(self):
        self._dataset = TrainDataset()
        self._sampler = TrainSampler()
        super().__init__(
            self._dataset,
            batch_size=None,
            sampler=self._sampler,
            num_workers=6,
            # num_workers=0,
            pin_memory=True,
        )
