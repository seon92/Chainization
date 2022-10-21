import torch
import pandas as pd
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import numpy as np
from PIL import Image
import os
from copy import deepcopy
from itertools import combinations
import pickle

class AdienceVanilla(Dataset):
    def __init__(self, data_file, transform=None):
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
        with open(data_file, 'rb') as f:
            data = pickle.load(f)
            self.imgs = data['data']
            self.ages = data['age']
            self.labels = data['labels']
        self.n_imgs = len(self.imgs)

    def __getitem__(self, item):
        img = np.asarray(self.imgs[item]).astype('uint8')
        img = self.transform(img)

        return img, item

    def __len__(self):
        return len(self.imgs)


if __name__=="__main__":
    train_file = 'Uniform_N3_th[30, 42]_R0_G0_train_5000.pickle'
    test_file = 'Uniform_N3_th[30, 42]_R0_G0_test_1000.pickle'
    ds = AdienceVanilla(train_file, 4)
    from torch.utils.data import DataLoader

    train_loader = DataLoader(ds, batch_size=8, shuffle=False, drop_last=False, num_workers=0)
    print('data pipeline test')
