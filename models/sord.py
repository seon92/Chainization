import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

from models.base_torch_projection_head import BaseModel



class SORD(BaseModel):
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
        self.fc = nn.Linear(hdim, opt.num_classes, bias=True)


    def _forward(self, base_embs, ref_embs=None):
        logits = self.fc(base_embs)
        if ref_embs is None:
            return logits

#
#
# if __name__ =="__main__":
#
#