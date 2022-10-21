from __future__ import print_function

import torch
from . import model_dict
from models.ternary_comparator_v1 import TernaryV1
from models.binary_comparator_v1 import BinaryV1
from models.binary_comparator_v2 import BinaryV2
from models.binary_comparator_v3_torchvision import BinaryV3
from models.binary_comparator_v4 import BinaryV4
from models.binary_comparator_v5 import BinaryV5
from models.binary_classifier import BinaryClf
from models.ternary_comparator_v3_torchvision import TernaryV3
from models.dra import DRA
from models.sord import SORD

def prepare_model(opt):
    model = eval(opt.model)(opt)

    # # load pre-trained model (no FC weights)
    # if args.init_weights is not None:
    #     model_dict = model.state_dict()
    #     pretrained_dict = torch.load(args.init_weights)['params']  # params, model for rfs pretrain model
    #     if args.backbone_class == 'ConvNet':
    #         pretrained_dict = {'encoder.' + k: v for k, v in pretrained_dict.items()}  # also do this for rfs model
    #     # pretrained_dict = {'encoder.' + k: v for k, v in pretrained_dict.items()}  # also do this for rfs model
    #
    #     pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    #     print(pretrained_dict.keys())
    #     model_dict.update(pretrained_dict)
    #     model.load_state_dict(model_dict)

    return model

