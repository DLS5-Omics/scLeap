#!/usr/bin/env python
import os
from collections import defaultdict
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

import click
import torch
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np
from tabulate import tabulate

from common import config as cfg
from model.finetune_model import FinetuneModel as model_fn
from data_provider.train_loader import TrainLoader
from data_provider.validation_loader import ValidationLoader
import scanpy
import anndata
import pandas as pd



def get_checkpoint_path(step):
    model_path = os.path.join(cfg.model_dir, "checkpoint-step-%s.pth" % step)
    return model_path


def load_model(step):
    model = model_fn()
    checkpoint_path = get_checkpoint_path(step)
    checkpoint = torch.load(checkpoint_path, map_location=torch.device("cpu"))
    parsed_dict = {}
    for k, v in checkpoint["state_dict"].items():
        if k.startswith("module."):
            k = k[7:]
        parsed_dict[k] = v
    model.load_state_dict(parsed_dict)
    return model


def _load_fn(model):
    data_list, label_list = [], []
    cell_name = []
    
    vl = ValidationLoader()
    dataset = vl._dataset
    perturb_list = dataset._perturb_all
    cat_len = len(perturb_list.cat.categories.tolist())
    
    out_array = np.empty((0, cat_len))
    
    file_path = "results.csv"

    for i, data in enumerate(tqdm(vl)):
        
        if torch.cuda.is_available():
            data = {k: v.cuda() for k, v in data.items()}
        
        with torch.no_grad():
            output = model(data)

        data_list.append(data["perturb"].item())
        _, pred = output["pred"][:,:cat_len].topk(1, 1, True, True)
        cell_name.append(dataset._cell_name[i])
        out_array = np.vstack((out_array, output["pred"][0][:cat_len].cpu().numpy()))
        label_list.append(pred)
        
    df = pd.DataFrame(out_array)
    df.columns = perturb_list.cat.categories.tolist()
    df.index = cell_name
    df.to_csv(file_path)
    out_array = np.empty((0, cat_len))


@click.command()
@click.option(
    "-c", "--checkpoint", help="Step to evaluate (e.g. 100000)", required=True
)
def evaluate(checkpoint):
    """
    Test checkpoint
    """
    model = load_model(checkpoint)
    model = model.eval()

    if torch.cuda.is_available():
        model = model.cuda()

    _load_fn(model)


if __name__ == "__main__":
    evaluate()
