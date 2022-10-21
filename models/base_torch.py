import torch
import torch.nn as nn
import numpy as np
import torchvision.models as models


class BaseModel(nn.Module):
    def __init__(self, opt):
        super().__init__()
        # self.encoder = model_dict[opt.backbone](avg_pool=True, drop_rate=0.0, dropblock_size=5)
        if opt.backbone == 'resnet18':
            backbone = models.resnet18(pretrained=True)
            backbone.fc = nn.Identity()
            self.encoder = backbone
        elif opt.backbone == 'vgg16':
            backbone = models.vgg16_bn(pretrained=True)
            backbone.avgpool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
            backbone.classifier = nn.Identity()
            self.encoder = backbone
        elif opt.backbone == 'vgg16fc':
            backbone = models.vgg16_bn(pretrained=True)
            backbone.classifier[5] = nn.Identity()
            backbone.classifier[6] = nn.Identity()
            self.encoder = backbone
        else:
            raise ValueError(f'Not supported backbone architecture {opt.backbone}')

    def forward(self, x, x_ref=None):
        # feature extraction
        base_embs = self.encoder(x)
        if x_ref is not None:
            ref_embs = self.encoder(x_ref)
            logits = self._forward(base_embs, ref_embs)
        else:
            logits = self._forward(base_embs)
        return logits


    def _forward(self, base_embs, ref_embs):
        raise NotImplementedError('Suppose to be implemented by subclass')