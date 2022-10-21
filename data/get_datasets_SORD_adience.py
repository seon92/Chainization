import random
from copy import deepcopy
from PIL import Image

import pickle
import numpy as np

import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import networkx as nx


def get_datasets(opt):
    loader_dict = dict()

    with open(opt.train_file, 'rb') as f:
        data = pickle.load(f)
        tr_imgs = data['data']
        tr_ages = data['age']

    with open(opt.test_file, 'rb') as f:
        data = pickle.load(f)
        te_imgs = data['data']
        te_ages = data['age']

    loader_dict['train_lb'] = DataLoader(AdienceVanilla(tr_imgs, tr_ages, opt.num_classes),
                                         batch_size=opt.batch_size, shuffle=True, drop_last=True, num_workers=opt.num_workers)

    loader_dict['test'] = DataLoader(AdienceVanilla(te_imgs, te_ages, opt.num_classes),
                                     batch_size=opt.batch_size, shuffle=False, drop_last=False, num_workers=opt.num_workers)

    return loader_dict



class AdienceVanilla(Dataset):
    def __init__(self, imgs, labels, n_cls, transform=None):
        super(Dataset, self).__init__()

        self.mean = [0.485, 0.456, 0.406]
        self.std = [0.229, 0.224, 0.225]
        self.normalize = transforms.Normalize(mean=self.mean, std=self.std)

        if transform is None:
            self.transform = transforms.Compose([
                lambda x: Image.fromarray(x),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                self.normalize
            ])
        else:
            self.transform = transform

        self.imgs = imgs

        self.labels = labels - labels.min()
        self.n_imgs = len(self.imgs)
        self.n_cls = n_cls

    def convert_one_hot(self, label):
        return torch.nn.functional.one_hot(torch.tensor(label), self.n_cls)


    def __getitem__(self, item):
        img = np.asarray(self.imgs[item]).astype('uint8')
        img = self.transform(img)
        target = self.labels[item]
        # one_hot = self.convert_one_hot(target)

        # return img, one_hot, target
        return img, target

    def __len__(self):
        return len(self.imgs)

