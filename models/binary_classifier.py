import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

from models.base_torch_projection_head import BaseModel


class Classifier(nn.Module):
    def __init__(self, hdim, n_cls=3):
        super().__init__()
        self.n_cls = n_cls

        self.fc1 = nn.Linear(hdim, hdim, bias=False)
        self.bn1 = nn.BatchNorm1d(hdim)
        self.relu1 = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(hdim, hdim, bias=False)
        self.bn2 = nn.BatchNorm1d(hdim)
        self.relu2 = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(p=0.3)
        self.fc3 = nn.Linear(hdim, n_cls)

    def forward(self, embs):
        x = self.fc1(embs)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.fc2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        x = self.dropout(x)
        out = self.fc3(x)
        return out


class BinaryClf(BaseModel):
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
        self.classifier = Classifier(hdim, n_cls=2)

    def _forward(self, base_embs, ref_embs=None):
        if ref_embs is None:
            logits = self.classifier(base_embs)
        return logits
#
#
# if __name__ =="__main__":
#
#