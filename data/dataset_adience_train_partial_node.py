import torch
import pandas as pd
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import numpy as np
from PIL import Image
import os
import pickle
from copy import deepcopy


class AdienceBaseTrain(Dataset):
    def __init__(self, base_file, sampling_ratio=0.05, use_age=False, transform=None):
        super(Dataset, self).__init__()

        self.mean = [0.485, 0.456, 0.406]
        self.std = [0.229, 0.224, 0.225]
        self.normalize = transforms.Normalize(mean=self.mean, std=self.std)

        if transform is None:
            self.transform = transforms.Compose([
                lambda x: Image.fromarray(x),
                transforms.RandomCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                self.normalize
            ])
        else:
            self.transform = transform

        with open(base_file, 'rb') as f:
            data = pickle.load(f)
            self.imgs = data['data']
            self.ages = data['age']
            self.labels = data['labels']

        self.n_base_imgs = len(self.imgs)

        if use_age:
            self.labels = self.ages - min(self.ages)

        self.sample_partial_nodes(sampling_ratio)
        self.min_age = self.labels.min()
        self.max_age = self.labels.max()




    def __getitem__(self, item):
        base_img = np.asarray(self.imgs[item]).astype('uint8')
        base_img = self.transform(base_img)

        order_label, ref_idx = find_reference(self.labels[item], self.labels, min_rank=self.min_age, max_rank=self.max_age)
        ref_img = np.asarray(self.imgs[ref_idx]).astype('uint8')
        ref_img = self.transform(ref_img)

        # order label generation
        one_hot_order_label = self.get_one_hot_order_label_from_order(order_label)

        # gt ages
        base_age = self.labels[item]
        ref_age = self.labels[ref_idx]
        return base_img, ref_img, one_hot_order_label, order_label, [base_age, ref_age], item

    def __len__(self):
        return len(self.imgs)
    #
    # def get_order_labels(self, base_idx, ref_idx):
    #     base_ranks = self.base_labels[base_idx]
    #     ref_ranks = self.ref_labels[ref_idx]
    #
    #     if base_ranks >= ref_ranks:
    #         order_labels = 0
    #     else:
    #         order_labels = 1
    #     return order_labels, base_ranks, ref_ranks

    def get_one_hot_order_label_from_order(self, order):
        if order == 0:
            one_hot_order_label = torch.tensor([1, 0], dtype=torch.float64)
        elif order == 1:
            one_hot_order_label = torch.tensor([0, 1], dtype=torch.float64)
        else:
            one_hot_order_label = torch.tensor([0.5, 0.5], dtype=torch.float64)

        return one_hot_order_label

    def sample_partial_nodes(self, leave_ratio, n_rank=8):
        n_sample_total = len(self.labels) * leave_ratio
        n_per_rank = int(n_sample_total / n_rank)
        uniq_ranks = np.unique(self.labels)
        idxs = []
        print(f'sampling node with {leave_ratio} for each rank. (#Per_rank :{n_per_rank})')

        for r in uniq_ranks:
            r_idxs = np.argwhere(self.labels == r).flatten()
            if len(r_idxs) > n_per_rank:
                selected = np.random.choice(r_idxs, n_per_rank, replace=False)
            else:
                selected = np.random.choice(r_idxs, n_per_rank, replace=True)
            idxs.append(selected)
        idxs = np.concatenate(idxs)

        self.imgs = self.imgs[idxs]
        self.labels = self.labels[idxs]
        self.ages = self.ages[idxs]






def get_indices_in_range(search_range, ages):
    """find indices of values within range[0] <= x <= range[1]"""
    return np.argwhere(np.logical_and(search_range[0] <= ages, ages <= search_range[1]))


def find_reference(base_rank, ref_ranks, min_rank=0, max_rank=32, epsilon=1e-4):
    order = np.random.choice([0, 1, 2], 1, p=[1/3,1/3,1/3])[0]
    ref_idx = -1
    debug_flag = 0
    while ref_idx == -1:
        if debug_flag == 3:
            raise ValueError(f'Failed to find reference... base_score: {base_rank}')
        if order == 0:    # base_rank > ref_rank
            ref_range_min = min_rank
            ref_range_max = base_rank - epsilon
            candidates = get_indices_in_range([ref_range_min, ref_range_max], ref_ranks)
            if len(candidates) > 0:
                ref_idx = candidates[np.random.choice(len(candidates), 1)[0]][0]
            else:
                order = (order+1)%3
                debug_flag += 1
                continue

        elif order == 1:
            ref_range_min = base_rank + epsilon
            ref_range_max = max_rank
            # base_rank < ref_rank
            candidates = get_indices_in_range([ref_range_min, ref_range_max], ref_ranks)
            if len(candidates) > 0:
                ref_idx = candidates[np.random.choice(len(candidates), 1)[0]][0]
            else:
                order = (order+1)%3
                debug_flag += 1
                continue

        else:             # base_rank < ref_rank
            ref_range_min = base_rank - epsilon
            ref_range_max = base_rank + epsilon
            candidates = get_indices_in_range([ref_range_min, ref_range_max], ref_ranks)
            if len(candidates) > 0:
                ref_idx = candidates[np.random.choice(len(candidates), 1)[0]][0]
            else:
                order = (order+1)%3
                debug_flag += 1
                continue

    return order, ref_idx


if __name__=="__main__":
    train_file = 'Uniform_N3_th[30, 42]_R0_G0_train_5000.pickle'
    test_file = 'Uniform_N3_th[30, 42]_R0_G0_test_1000.pickle'
    ds = AdienceBaseTrain(train_file, test_file, 4)
    from torch.utils.data import DataLoader

    test_loader = DataLoader(ds, batch_size=8, shuffle=False, drop_last=False, num_workers=0)
    for base_img, ref_img, order_labels, [base_age, ref_age], [base_ranks, ref_ranks], item in test_loader:
        break
    print('data pipeline test')
