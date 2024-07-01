#!/usr/bin/env python
import os
import sys
import math
from pathlib import Path

import click
import numpy as np
import pandas as pd
from tqdm import tqdm
import scipy.stats


def load_csv(path):
    df = pd.read_csv(path)
    return df.to_numpy()[:, 1:].astype(float)


def normalize(x):
    eps = 1e-12
    x = np.log1p(x / (x.sum(axis=0) + eps) * 10000)
    return x


@click.command()
@click.argument("csv1")
@click.argument("csv2")
def main(csv1, csv2):
    x = normalize(load_csv(csv1))
    y = normalize(load_csv(csv2))
    sp, nrmse = [], []
    cnt = 0
    for i in range(x.shape[0]):
        t = scipy.stats.spearmanr(x[i], y[i])[0]
        r = np.mean((x[i] - y[i]) ** 2) / (x[i].max() - x[i].min())
        if not math.isnan(t):
            sp.append(t)
        else:
            cnt += 1

        if not math.isnan(r):
            nrmse.append(r)
    print(np.mean(sp), np.mean(nrmse), cnt)


if __name__ == "__main__":
    main()
