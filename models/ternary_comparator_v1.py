import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

from models.base import BaseModel


class Comparator(nn.Module):
    def __init__(self, hdim, n_cls=3):
        super().__init__()
        self.n_cls = n_cls

        self.fc1 = nn.Linear(2*hdim, 2*hdim, bias=False)
        self.bn1 = nn.BatchNorm1d(2*hdim)
        self.relu1 = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(2*hdim, 2*hdim, bias=False)
        self.bn2 = nn.BatchNorm1d(2*hdim)
        self.relu2 = nn.ReLU(inplace=True)
        self.fc3 = nn.Linear(2*hdim, n_cls)

    def forward(self, base_embs, ref_embs):
        embs = torch.cat([base_embs, ref_embs], dim=-1)
        x = self.fc1(embs)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.fc2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        out = self.fc3(x)
        return out


class TernaryV1(BaseModel):
    def __init__(self, opt):
        super().__init__(opt)
        if opt.backbone == 'resnet12':
            hdim = 640
        elif opt.backbone == 'resnet18':
            hdim = 640
        else:
            raise ValueError('')

        comparators = []
        for _ in range(opt.n_phases):
            comparators.append(Comparator(hdim, n_cls=3))
        self.comparators = nn.ModuleList(comparators)

    def _forward(self, base_embs, ref_embs):
        logits = []
        for comparator in self.comparators:
            logits.append(comparator(base_embs, ref_embs))
        return logits
