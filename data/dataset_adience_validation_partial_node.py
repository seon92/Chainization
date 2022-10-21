import torch
import pandas as pd
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import numpy as np
from PIL import Image
import os
import pickle
from copy import deepcopy


class AdienceReference(Dataset):
    def __init__(self, base_file, ref_file, sampling_ratio=0.05, use_age=False, transform=None):
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

        with open(ref_file, 'rb') as f:
            data = pickle.load(f)
            self.ref_imgs = data['data']
            self.ref_ages = data['age']
            self.ref_labels = data['labels']

        with open(base_file, 'rb') as f:
            data = pickle.load(f)
            self.base_imgs = data['data']
            self.base_ages = data['age']
            self.base_labels = data['labels']

        self.n_base_imgs = len(self.base_imgs)

        if use_age:
            self.base_labels = self.base_ages - min(self.base_ages)
            self.ref_labels = self.ref_ages - min(self.ref_ages)
        self.sample_refs_partial_nodes(sampling_ratio)
        self.n_ref_imgs = len(self.ref_imgs)

    def __getitem__(self, item):
        base_img = np.asarray(self.base_imgs[item]).astype('uint8')
        base_img = self.transform(base_img)

        ref_idx = np.random.choice(self.n_ref_imgs, 1)[0]
        ref_img = np.asarray(self.ref_imgs[ref_idx]).astype('uint8')
        ref_img = self.transform(ref_img)

        # order label generation
        order_labels, base_ranks, ref_ranks = self.get_order_labels(item, ref_idx)

        # gt ages
        base_age = self.base_ages[item]
        ref_age = self.ref_ages[ref_idx]
        return base_img, ref_img, order_labels, [base_age, ref_age], item

    def __len__(self):
        return len(self.base_imgs)

    def get_order_labels(self, base_idx, ref_idx):
        base_ranks = self.base_labels[base_idx]
        ref_ranks = self.ref_labels[ref_idx]

        if base_ranks > ref_ranks:
            order_labels = 0
        elif base_ranks < ref_ranks:
            order_labels = 1
        else:
            order_labels = 2
        return order_labels, base_ranks, ref_ranks

    def sample_refs_partial_nodes(self, leave_ratio, n_rank=8):
        n_sample_total = len(self.ref_labels) * leave_ratio
        n_per_rank = int(n_sample_total / n_rank)
        uniq_ranks = np.unique(self.ref_labels)
        idxs = []
        print(f'sampling node with {leave_ratio} for each rank. (#Per_rank :{n_per_rank})')

        for r in uniq_ranks:
            r_idxs = np.argwhere(self.ref_labels == r).flatten()
            if len(r_idxs) > n_per_rank:
                selected = np.random.choice(r_idxs, n_per_rank, replace=False)
            else:
                selected = np.random.choice(r_idxs, n_per_rank, replace=True)
            idxs.append(selected)
        idxs = np.concatenate(idxs)
        self.ref_imgs = self.ref_imgs[idxs]
        self.ref_labels = self.ref_labels[idxs]
        self.ref_ages = self.ref_ages[idxs]


if __name__=="__main__":
    train_file = 'Uniform_N3_th[30, 42]_R0_G0_train_5000.pickle'
    test_file = 'Uniform_N3_th[30, 42]_R0_G0_test_1000.pickle'
    ds = MorphReference(train_file, test_file, 4)
    from torch.utils.data import DataLoader

    test_loader = DataLoader(ds, batch_size=8, shuffle=False, drop_last=False, num_workers=0)
    for base_img, ref_img, order_labels, [base_age, ref_age], [base_ranks, ref_ranks], item in test_loader:
        break
    print('data pipeline test')
