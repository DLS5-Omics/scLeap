import math
import torch
import torch.nn as nn
from torch.nn import functional as F


class RawCountEncoding(nn.Module):
    def __init__(self, num_buckets=2048, max_value=20000, out_dim=256, dropout=0.1):
        super(RawCountEncoding, self).__init__()
        self.num_buckets = num_buckets
        self.max_value = max_value
        self.out_dim = out_dim
        self.fc = nn.Sequential(
            nn.Embedding(self.num_buckets, self.out_dim), nn.Dropout(dropout)
        )

    @staticmethod
    def _rawcount_bucket(rawcount, num_buckets, max_value):
        max_exact = num_buckets // 2
        is_small = rawcount < max_exact

        val_if_large = (
            max_exact
            + (
                torch.log(rawcount / max_exact)
                / math.log(max_value / max_exact)
                * (num_buckets - max_exact)
            ).long()
        )
        val_if_large = torch.min(
            val_if_large, torch.full_like(val_if_large, num_buckets - 1)
        )

        return torch.where(is_small, rawcount, val_if_large)

    def forward(self, rawcount):
        bucket = self._rawcount_bucket(
            rawcount,
            num_buckets=self.num_buckets,
            max_value=self.max_value,
        )
        x = self.fc(bucket)
        return x
