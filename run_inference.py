#!/usr/bin/env python
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import click
import torch
import torch.nn.functional as F
from tqdm import tqdm


from common import config as cfg
from model.main_model import MainModel as model_fn


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


def _inference_fn(model, ds, ds_norm, u_token, v_token):
    device = next(model.parameters()).device

    ds = torch.tensor(ds).long().to(device)
    ds_norm = torch.tensor(ds_norm).float().to(device)
    u_token = torch.tensor(u_token).long().to(device)
    v_token = torch.tensor(v_token).long().to(device)
    ratio = torch.tensor([20]).long().to(device)

    with torch.no_grad():
        out = model(
                {"ds": ds, "ds_norm": ds_norm, "u_token": u_token, "v_token": v_token, "ratio": ratio},
            False,
        )
    u_pred_bin = torch.softmax(out["u_pred_bin"], dim=-1).cpu().numpy()
    v_pred_bin = torch.softmax(out["v_pred_bin"], dim=-1).cpu().numpy()
    cell_embs = out["cell_embeddings"].cpu().numpy()
    bins = cfg.bins
    bins = (bins[1:] + bins[:-1]) / 2
    u_pred = np.sum(u_pred_bin * bins, axis=-1)
    v_pred = np.sum(v_pred_bin * bins, axis=-1)

    u_pred = out["u_pred"].cpu().numpy()
    v_pred = out["v_pred"].cpu().numpy()
    u_pred[u_pred < 0] = 0
    v_pred[v_pred < 0] = 0

    return u_pred, v_pred, cell_embs


@click.command()
@click.option(
    "-c", "--checkpoint", help="Step to evaluate (e.g. 100000)", required=True
)
@click.option("-i", "--csv", required=True)
@click.option("-o", "--output", required=True)
@click.option("-oemb", "--output_emb", required=True)
@click.option("--use-gpu/--no-use-gpu", default=False)
def inference(checkpoint, csv, output, output_emb, use_gpu):
    """
    Test checkpoint
    """
    model = load_model(checkpoint)

    model = model.eval()
    if use_gpu and torch.cuda.is_available():
        model = model.cuda()

    df = pd.read_csv(csv)
    ref_gene_token = {}
    for i, x in enumerate(open(cfg.dataset_dir / "gene_name.txt")):
        ref_gene_token[x.strip()] = i

    row2gene = [40000]
    gene_idx, gene_token = [], []
    for i in range(df.shape[0]):
        row2gene.append(ref_gene_token.get(df.iloc[i, 0], -1))
    row2gene = np.array(row2gene)
    print("Known gene-id num:", (row2gene != -1).sum() - 1, "out of", row2gene.shape[0] - 1)

    bs, num_u, num_v = 128, 512, 200000

    def normalize(x):
        eps = 1e-12
        x = np.log1p(x / (x.sum(axis=0) + eps) * 10000)
        return x

    rawcount = df.iloc[:, 1:].to_numpy().astype(int)
    rawcount_norm = normalize(rawcount)

    n_gene, n_cell = rawcount.shape[0], rawcount.shape[1]
    embeddings = []

    for j in tqdm(range(0, n_cell, bs)):
        jst, jed = j, j + bs
        jed = min(jed, n_cell)
        u_row_idx, v_row_idx = [], []

        for i in range(jst, jed):
            u = np.arange(n_gene)[(row2gene[1:] != -1) & (rawcount[:, i] > 0)]
            v = np.arange(n_gene)[(row2gene[1:] != -1) & (rawcount[:, i] == 0)]
            u_row_idx.append(u[:num_u])
            v_row_idx.append(v)

        def _pad_fn(a, value):
            max_shape = np.max([_.shape for _ in a], axis=0)
            na = []
            for x in a:
                pad_shape = [(0, l2 - l1) for l1, l2 in zip(x.shape, max_shape)]
                na.append(np.pad(x, pad_shape, mode="constant", constant_values=value))
            return np.stack(na)

        u_row_idx = _pad_fn(u_row_idx, -1)
        v_row_idx = _pad_fn(v_row_idx, -1)
        ds = np.zeros(u_row_idx.shape)
        ds_norm = np.zeros(u_row_idx.shape)
        for r in range(jst, jed):
            for c in range(u_row_idx.shape[1]):
                if u_row_idx[r - jst, c] != -1:
                    ds[r - jst, c] = rawcount[u_row_idx[r - jst, c]][r]
                    ds_norm[r - jst, c] = rawcount_norm[u_row_idx[r - jst, c]][r]
        u_token = row2gene[u_row_idx + 1]
        u_pred, cell_embs = None, None

        for i in range(0, v_row_idx.shape[1], num_v):
            ist, ied = i, i + num_v
            ied = min(ied, v_row_idx.shape[1])

            sub_v_row_idx = v_row_idx[:, ist:ied]
            v_token = row2gene[sub_v_row_idx + 1]

            u_pred, v_pred, cell_embs = _inference_fn(
                model, ds, ds_norm, u_token, v_token
            )
            for r in range(jst, jed):
                for c in range(sub_v_row_idx.shape[1]):
                    if sub_v_row_idx[r - jst, c] != -1:
                        rawcount_norm[sub_v_row_idx[r - jst, c]][r] = v_pred[r - jst, c]

        for r in range(jst, jed):
            for c in range(u_row_idx.shape[1]):
                if u_row_idx[r - jst, c] != -1:
                    rawcount_norm[u_row_idx[r - jst, c]][r] = u_pred[r - jst, c]
        embeddings.append(cell_embs)

    imputed = np.exp(rawcount_norm) - 1
    imputed = np.around(imputed / (imputed.sum(axis=0) + 1e-12) * 10000)
    lines = [",".join(df.columns)]
    for i in range(df.shape[0]):
        line = [df.iloc[i, 0]] + ["%.0f" % _ for _ in imputed[i]]
        lines.append(",".join(line))

    with open(output, "w") as fp:
        fp.write("\n".join(lines))
    embeddings = np.concatenate(embeddings, axis=1)
    np.save(output_emb, embeddings)


if __name__ == "__main__":
    inference()
