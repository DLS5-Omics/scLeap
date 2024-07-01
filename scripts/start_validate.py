#!/usr/bin/env python
import os
from collections import defaultdict
import sys
from pathlib import Path
from sklearn.metrics import f1_score

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


def _validate_fn(model):
    b_losses, b_metrics = defaultdict(list), defaultdict(list)
    data_list, out_list = [], []
    for i, data in enumerate(tqdm(ValidationLoader())):
        if torch.cuda.is_available():
            data = {k: v.cuda() for k, v in data.items()}
        with torch.no_grad():
            output = model(data)
            losses = {k: v for k, v in output.items() if k.endswith("loss")}
            metrics = {
                k: v
                for k, v in output.items()
                if k.endswith("_acc") or k.startswith("token")
            }
            data_list.append(data["perturb"].item())
            _, pred = output["pred"].topk(1, 1, True, True)
            out_list.append(pred.item())
            for k, v in losses.items():
                b_losses[k].append(v.item())
            for k, v in metrics.items():
                b_metrics[k].append(v)
        if i % 100 == 99:

            loss_str = ", ".join(
                ["%s: %.6f" % (k, sum(v) / len(v)) for k, v in b_losses.items()]
            )
            metric_str = ", ".join(
                ["%s: %.6f" % (k, sum(v) / len(v)) for k, v in b_metrics.items()]
            )
            f1 = f1_score(data_list, out_list, average='weighted')
            log_str = f"{loss_str}, {metric_str}"
            print(log_str, "| f1 score:", f1)


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

    _validate_fn(model)


if __name__ == "__main__":
    evaluate()
