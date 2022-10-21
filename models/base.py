import torch
import torch.nn as nn
import numpy as np
from . import model_dict


class BaseModel(nn.Module):
    def __init__(self, opt):
        super().__init__()
        self.encoder = model_dict[opt.backbone](avg_pool=True, drop_rate=0.0, dropblock_size=5)

    def forward(self, x_base, x_ref):
        # feature extraction
        base_embs = self.encoder(x_base)
        ref_embs = self.encoder(x_ref)

        logits = self._forward(base_embs, ref_embs)
        return logits

    def _forward(self, base_embs, ref_embs):
        raise NotImplementedError('Suppose to be implemented by subclass')