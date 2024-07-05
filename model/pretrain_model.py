import torch
from torch import nn
import torch.nn.functional as F

from common import config as cfg
from .base_model import BaseModel
from .attention import TransformerEncoder, TransformerDecoder
from .rawcount_encoding import RawCountEncoding


class PretrainModel(BaseModel):
    def __init__(self, pretrain=True):
        super(PretrainModel, self).__init__()
        self.pretrain = pretrain
        self.d_model = 256

        self.embed_exp = nn.Sequential(nn.Linear(1, self.d_model), nn.Dropout(0.1))
        self.embed_ratio = nn.Sequential(
            nn.Embedding(21, self.d_model), nn.Dropout(0.1)
        )
        self.embed_gene = nn.Sequential(
            nn.Embedding(40002, self.d_model, padding_idx=40000), nn.Dropout(0.1)
        )
        
        self.embed_rawcount = RawCountEncoding(
            num_buckets=2048, max_value=20000, out_dim=self.d_model, dropout=0.1
        )
        self.encoder = TransformerEncoder(
            n_layer=6,
            d_model=self.d_model,
            d_key=self.d_model,
            n_head=self.d_model // 16,
            dim_feedforward=self.d_model * 4,
            dropout=0.1
        )
        self.decoder = TransformerDecoder(
            n_layer=2,
            d_model=self.d_model,
            d_key=self.d_model,
            n_head=self.d_model // 16,
            dim_feedforward=self.d_model * 4,
            dropout=0.1,
        )
        self.fc = nn.Sequential(
            nn.LayerNorm(self.d_model),
            nn.Linear(self.d_model, self.d_model),
            nn.ReLU(),
            nn.Linear(self.d_model, len(cfg.bins)),
        )
        if not self.pretrain:
            for x in self.decoder.parameters():
                if x.requires_grad:
                    x.requires_grad = False
            for x in self.fc.parameters():
                if x.requires_grad:
                    x.requires_grad = False

    def forward(self, data, in_degree_embed, out_degree_embed, mask_attn_embed, compute_loss=True):
        src = data["ds_norm"].unsqueeze(-1)
        rawcount = data["ds"]
        u_token = data["u_token"]

        cls_id = torch.ones(
            (u_token.shape[0], 1),
            dtype=u_token.dtype,
            device=u_token.device,
        )
        cls_id[:] = 40001
        cls_emb = self.embed_gene(cls_id) + self.embed_ratio(data["ratio"]) ## shape [1,1,256]

        src = (
            self.embed_exp(src)
            + self.embed_gene(u_token)
            + self.embed_rawcount(rawcount)
            + in_degree_embed
            + out_degree_embed
        )

        src = torch.cat([cls_emb, src], dim=1)
        u_token = torch.cat([cls_id, u_token], dim=1)
        mask = u_token == 40000

        cell_embs = []
        src, emb = self.encoder(src, mask, mask_attn_embed)

        if not self.pretrain:
            return src
        cell_embs.extend(emb)
        v_token = data["v_token"]
        if compute_loss:
            tgt = self.embed_gene(v_token)
            src, tgt, emb = self.decoder(src, tgt, mask)
            cell_embs.extend(emb)
            src = self.fc(src)
            tgt = self.fc(tgt)
        else:
            tgts = []
            step = 128
            osrc = None
            for i in range(0, v_token.shape[1], step):
                tgt = self.embed_gene(v_token[:, i : i + step])
                osrc, tgt, emb = self.decoder(src, tgt, mask)
                if i == 0:
                    cell_embs.extend(emb)
                tgt = self.fc(tgt)
                tgts.append(tgt)
            src = self.fc(osrc)
            tgt = torch.cat(tgts, dim=1)
        out = {
            "u_pred_bin": src[:, 1:, 1:],
            "v_pred_bin": tgt[..., 1:],
            "u_pred": src[:, 1:, 0],
            "v_pred": tgt[..., 0],
            "cell_embeddings": torch.stack(cell_embs),
        }

        if compute_loss:
            out.update(self._compute_loss(out, data))
            out.update(self._compute_metric(out, data))
        return out

    def _compute_loss(self, out, data):
        u_mask = data["u_token"] == 40000
        u_ce_loss = F.cross_entropy(
            out["u_pred_bin"][~u_mask], data["u_gt_bin"][~u_mask]
        )

        v_mask = data["v_token"] == 40000
        v_ce_loss = F.cross_entropy(
            out["v_pred_bin"][~v_mask], data["v_gt_bin"][~v_mask]
        )
        u_mse_loss = F.mse_loss(out["u_pred"][~u_mask], data["u_gt"][~u_mask])
        v_mse_loss = F.mse_loss(out["v_pred"][~v_mask], data["v_gt"][~v_mask])

        losses = {
            "u_ce_loss": u_ce_loss,
            "v_ce_loss": v_ce_loss,
            "u_mse_loss": u_mse_loss,
            "v_mse_loss": v_mse_loss,
        }
        losses["update_loss"] = u_ce_loss + v_ce_loss + u_mse_loss + v_mse_loss
        return losses

    def _compute_metric(self, out, data):
        return {}
