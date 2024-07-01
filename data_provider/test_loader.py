import os
import random
import itertools

import numpy as np
import torch
from torch.utils.data import DataLoader

from common import config as cfg
from .dataset import TestDataset


class TestSampler:
    def __init__(self, end_iter):
        self.end_iter = end_iter
        pass

    def __iter__(self):
        i = 0
        while True:
            if i == self.end_iter - 1:
                return
            else:
                yield i
                i += 1


class TestLoader(DataLoader):
    def __init__(self):
        self._dataset = TestDataset()
        self._sampler = TestSampler(end_iter=len(self._dataset))
        super().__init__(
            self._dataset,
            batch_size=None,
            sampler=self._sampler,
            num_workers=1,
            pin_memory=True,
        )

    def __len__(self):
        return len(self._dataset)
