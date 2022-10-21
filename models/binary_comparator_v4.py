import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

from models.base_torch import BaseModel
"""using all pairs within batch"""

class Comparator(nn.Module):
    def __init__(self, hdim, n_cls=3):
        super().__init__()
        self.n_cls = n_cls
        self.hdim = hdim
        self.fc1 = nn.Linear(2*hdim, 2*hdim, bias=False)
        self.bn1 = nn.BatchNorm1d(2*hdim)
        self.relu1 = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(2*hdim, 2*hdim, bias=False)
        self.bn2 = nn.BatchNorm1d(2*hdim)
        self.relu2 = nn.ReLU(inplace=True)
        self.fc3 = nn.Linear(2*hdim, n_cls)

    def forward(self, embs, ref_embs=None):
        if ref_embs is not None:
            embs = torch.cat([embs, ref_embs], dim=-1)
            x = self.fc1(embs)
            x = self.bn1(x)
            x = self.relu1(x)
            x = self.fc2(x)
            x = self.bn2(x)
            x = self.relu2(x)
            out = self.fc3(x)

        else:
            n_embs = embs.shape[0]
            embs_base = embs.view(n_embs, 1, self.hdim).expand(n_embs, n_embs, self.hdim).contiguous().view(-1, self.hdim)
            embs_ref = embs.view(1, n_embs, self.hdim).expand(n_embs, n_embs, self.hdim).contiguous().view(-1, self.hdim)
            embs = torch.cat([embs_base, embs_ref], dim=-1)
            x = self.fc1(embs)
            x = self.bn1(x)
            x = self.relu1(x)
            x = self.fc2(x)
            x = self.bn2(x)
            x = self.relu2(x)
            out = self.fc3(x)
        return out


class BinaryV4(BaseModel):
    def __init__(self, opt):
        super().__init__(opt)
        if opt.backbone == 'resnet12':
            hdim = 640
        elif opt.backbone == 'resnet18':
            hdim = 640
        elif opt.backbone == 'vgg16':
            hdim = 512
        elif opt.backbone == 'vgg16fc':
            hdim = 4096
        else:
            raise ValueError('')
        #
        # comparators = []
        # for _ in range(opt.n_phases):
        #     comparators.append()
        # self.comparators = nn.ModuleList(comparators)
        self.comparator = Comparator(hdim, n_cls=2)

    def _forward(self, embs, ref_embs=None):
        if ref_embs is not None:
            logits = self.comparator(embs, ref_embs)
        else:
            logits = self.comparator(embs)
        return logits
#
#
# if __name__ =="__main__":
#
#