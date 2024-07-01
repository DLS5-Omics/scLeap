#!/usr/bin/env python
import sys
from pathlib import Path

import torch

sys.path.append(str(Path(__file__).parent.parent.resolve()))
from model.finetune_model import FinetuneModel as model_fn
from data_provider.train_loader import TrainLoader


f = model_fn()
print(str(f))


dp = TrainLoader()

for data in dp:
    o = f(data)
    for k, v in o.items():
        print(k)
    print(o)
    sys.exit(1)
    loss = {"pair_loss": o["pair_loss"]}
    print(loss)
    # break
    sum(loss.values()).backward()
    for x in f.named_parameters():
        if x[1].requires_grad:
            if x[1].grad is None:
                print(x[0])
            if torch.max(x[1].grad) > 1:
                print(x[0], torch.max(x[1].grad), torch.min(x[1].grad))
    break
