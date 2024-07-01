#!/usr/bin/env python
import sys
from pathlib import Path

from tqdm import tqdm
import torch

sys.path.append(str(Path(__file__).parent.parent.resolve()))
from data_provider.train_loader import TrainLoader
from data_provider.validation_loader import ValidationLoader


if __name__ == "__main__":

    def stats(x, name):
        print(name, x.norm(), x.min(), x.max())

    dp = ValidationLoader()
    print(len(dp))
    for i, x in tqdm(enumerate(dp)):
        print([(k, v.shape) for k, v in x.items()])
        continue
        stats(x["downsample"][0] - x["expression"][0], "delta")
        stats(x["downsample"][0], "downsample")
        stats(x["expression"][0], "expression")
