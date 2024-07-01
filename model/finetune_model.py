import torch
from torch import nn
import torch.nn.functional as F


from common import config as cfg
from .base_model import BaseModel
from .pretrain_model import PretrainModel
from sklearn.metrics import f1_score


class FinetuneModel(BaseModel):
    def __init__(self):
        super(FinetuneModel, self).__init__()
        self.pretrain = PretrainModel(pretrain=False)
        # for x in self.pretrain.parameters():
        #     if x.requires_grad:
        #         x.requires_grad = False
        inplane = self.pretrain.d_model
        self.d_model = 256
        self.embed_in_degree = nn.Sequential(
            nn.Embedding(20000, self.d_model, padding_idx=0)
        )
        self.embed_out_degree = nn.Sequential(
            nn.Embedding(20000, self.d_model, padding_idx=0)
        )
        self.graph_attn_bias = nn.Embedding(10, 16)
        
        self.fc = nn.Sequential(
            nn.LayerNorm(inplane),
            nn.Dropout(0.1),
            nn.Linear(inplane, inplane * 4),
            nn.ReLU(),
            nn.Linear(inplane * 4, cfg.n_cls),
        )

    def forward(self, data):
        in_degree = data["in_degree"]
        out_degree = data["out_degree"]
        mask_attn = data["mask_attn"]
        cls_mask1 = torch.ones(
            (mask_attn.shape[0], 1, mask_attn.shape[2]),
            dtype=mask_attn.dtype,
            device=mask_attn.device,
        )*6
        mask_attn = torch.cat([cls_mask1, mask_attn], dim=1)
        cls_mask2 = torch.ones(
            (mask_attn.shape[0], mask_attn.shape[1], 1),
            dtype=mask_attn.dtype,
            device=mask_attn.device,
        )*6
        mask_attn = torch.cat([cls_mask2, mask_attn], dim=2)
        mask_attn[mask_attn == -1] = 5
        
        in_degree_embed =  self.embed_in_degree(in_degree)
        out_degree_embed =  self.embed_out_degree(out_degree)
        mask_attn_embed = self.graph_attn_bias(mask_attn).view(mask_attn.shape[0], 16, mask_attn.shape[1], mask_attn.shape[2])

        x = self.pretrain(data, in_degree_embed, out_degree_embed, mask_attn_embed)
        x = self.fc(x[:, 0])

        out = {"pred": x}
        out.update(self._compute_loss(out, data))
        out.update(self._compute_metric(out, data))
        return out

    def _compute_loss(self, out, data):
        ce_loss = F.cross_entropy(out["pred"], data["perturb"])
        losses = {"ce_loss": ce_loss}
        losses["update_loss"] = ce_loss
        return losses

    def _compute_metric(self, out, data):
        def accuracy(output, target, topk=(1,)):
            maxk = max(topk)
            batch_size = target.size(0)
            _, pred = output.topk(maxk, 1, True, True)
            pred = pred.t()
            correct = pred.eq(target.view(1, -1).expand_as(pred))
            res = []

            for k in topk:
                correct_k = correct[:k].reshape(-1).float().sum(0)
                res.append(correct_k.mul_(1.0 / batch_size))
            return res

        with torch.no_grad():
            metrics = {}
            topk = [1, 5, 10, 50, 100]
            acc = accuracy(out["pred"], data["perturb"], topk)
            for x, v in zip(topk, acc):
                metrics[f"top{x}_acc"] = v
            return metrics
