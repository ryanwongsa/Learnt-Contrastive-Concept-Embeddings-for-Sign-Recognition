import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

import numpy as np
import importlib
from models.model_utils.specaug import dropmask

class HeadModel(nn.Module):
    def __init__(
        self,
        in_dim,
        hidden_dim,
        num_classes,
        num_m,
        head_pool_type="avg",
        num_background=1,
        dropout=0.5,
        temperature=0.2,
        apply_dropmask=False,
        temporal_drop_width=4,
        channel_drop_width=64,
    ):
        super().__init__()

        if hidden_dim is not None:
            self.fc_hidden = nn.Linear(in_dim, hidden_dim)
        else:
            self.fc_hidden = nn.Identity()
            hidden_dim = in_dim

        vocab_embedding = torch.nn.Parameter(
            torch.empty(hidden_dim, num_classes + num_background, num_m)
        )
        torch.nn.init.xavier_uniform_(vocab_embedding, gain=1)
        vocab_embedding.requires_grad = True
        self.vocab_embedding = vocab_embedding

        self.num_classes = num_classes
        self.num_m = num_m
        self.head_pool_type = head_pool_type
        self.temperature = temperature

        self.bias = 0.0
        self.dropout = nn.Dropout(dropout)
        self.apply_dropmask = apply_dropmask
        self.temporal_drop_width = temporal_drop_width
        self.channel_drop_width = channel_drop_width


    def compute_embedding_v(self):
        v = self.vocab_embedding
        C, V, M = v.shape

        if self.head_pool_type == "max":
            v_out = F.adaptive_max_pool3d(v.unsqueeze(0), (C, V, 1)).squeeze(-1)
        elif self.head_pool_type == "avg":
            v_out = F.adaptive_avg_pool3d(v.unsqueeze(0), (C, V, 1)).squeeze(-1)
        elif self.head_pool_type == "avg+max":
            v_out = F.adaptive_avg_pool3d(v.unsqueeze(0), (C, V, 1)).squeeze(
                -1
            ) + F.adaptive_max_pool3d(v.unsqueeze(0), (C, V, 1)).squeeze(-1)

        vocab_embedding = torch.nn.functional.normalize(
            v_out.squeeze(0)[:, : self.num_classes], dim=0
        )
        return torch.mm(vocab_embedding.T, vocab_embedding)

    def logit_compare_embed(self, out, vocab):
        N, T, C = out.shape
        _, V, M = vocab.shape

        if self.training and self.apply_dropmask:
            out, vocab = dropmask(
                out.unsqueeze(1),
                vocab,
                channel_stripes_num=2,
                channel_drop_width=self.channel_drop_width,
                temporal_stripes_num=2,
                temporal_drop_width=self.temporal_drop_width,
            )
            out = out.squeeze(1)

        out = F.normalize(out, dim=-1)
        vocab = F.normalize(vocab, dim=0)

        fc_out = torch.bmm(
            out.reshape(N, T, C),
            vocab.reshape(C, V * M).unsqueeze(0).repeat(N, 1, 1),
        ).reshape(N, T, V, M)

        fc_out = fc_out / self.temperature + self.bias

        if self.head_pool_type == "max":
            fc_out = F.adaptive_max_pool3d(fc_out, (T, V, 1)).squeeze(-1)
        elif self.head_pool_type == "avg":
            fc_out = F.adaptive_avg_pool3d(fc_out, (T, V, 1)).squeeze(-1)
        elif self.head_pool_type == "avg+max":
            fc_out = F.adaptive_avg_pool3d(fc_out, (T, V, 1)).squeeze(
                -1
            ) + F.adaptive_max_pool3d(fc_out, (T, V, 1)).squeeze(-1)
        fc_out = fc_out.reshape(N, T, V)
        time_res = torch.softmax(fc_out, dim=-1)
        return time_res

    def forward(self, x):
        # b x t x c
        b, t, c = x.shape

        y = self.fc_hidden(self.dropout(x))

        time_res = self.logit_compare_embed(y, self.vocab_embedding)

        total_vocab_sum = time_res[:, :, : self.num_classes].sum(axis=(1, 2)) + 1e-8

        logits = (
            time_res[:, :, : self.num_classes].sum(axis=1)
        ) / total_vocab_sum.unsqueeze(-1)

        V = self.compute_embedding_v()

        return {
            "time_res": time_res,
            "logits": logits,
            "V": V,
            "vocab_embedding": self.vocab_embedding,
            "background": time_res[:, :, self.num_classes :].sum(axis=-1),
            "length_T": time_res.shape[1],
        }


if __name__ == "__main__":
    model_params = {
        "in_dim": 512,
        "hidden_dim": 512,
        "num_classes": 1000,
        "num_m": 5,
        "head_pool_type": "avg",
        "num_background": 2,
        "dropout": 0.5,
        "temperature": 0.2,
        "apply_dropmask": True,
        "temporal_drop_width": 2,
        "channel_drop_width": 64,
    }

    model = HeadModel(**model_params)

    x = torch.randn(8, 64, 512)

    y = model(x)

    print(y)
