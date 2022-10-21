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
        tr_ages = data['labels']

    with open(opt.test_file, 'rb') as f:
        data = pickle.load(f)
        te_imgs = data['data']
        te_ages = data['labels']

    # adience
    if opt.dataset == 'adience':
        mapping = dict()
        mapping[0] = 0
        mapping[1] = 0
        mapping[2] = 0
        mapping[3] = 0
        mapping[4] = 1
        mapping[5] = 1
        mapping[6] = 1
        mapping[7] = 1


    # morph
    elif opt.dataset == 'morph':
        mapping = dict()
        tr_ages = tr_ages - tr_ages.min()

        threshold = 19
        for i in range(tr_ages.max() + 1):
            if i < threshold:
                mapping[i] = 0
            else:
                mapping[i] = 1

    tr_bp_lb = []
    for i in range(len(tr_ages)):
        tr_bp_lb.append(mapping[tr_ages[i]])
    tr_bp_lb = np.array(tr_bp_lb)

    loader_dict['train_lb'] = DataLoader(PartialEdgeLabeled(tr_imgs, tr_bp_lb),
                                         batch_size=opt.batch_size, shuffle=True, drop_last=True, num_workers=opt.num_workers)

    loader_dict['test'] = DataLoader(Reference([te_imgs, te_ages], [tr_imgs, tr_ages]),
                                     batch_size=opt.batch_size, shuffle=False, drop_last=False, num_workers=opt.num_workers)


    loader_dict['vanilla_ulb'] = DataLoader(Vanilla(tr_imgs, tr_ages),
                                            batch_size=opt.batch_size, shuffle=False, drop_last=False, num_workers=opt.num_workers)
    return loader_dict



class Vanilla(Dataset):
    def __init__(self, imgs, labels, transform=None):
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

        self.labels = labels
        self.labels = self.labels - self.labels.min()
        self.n_imgs = len(self.imgs)

    def __getitem__(self, item):
        img = np.asarray(self.imgs[item]).astype('uint8')
        img = self.transform(img)

        return img, item

    def __len__(self):
        return len(self.imgs)


class PartialEdgeLabeled(Dataset):
    def __init__(self, imgs, labels, norm_age=False, transform=None):
        super(Dataset, self).__init__()

        self.mean = [0.485, 0.456, 0.406]
        self.std = [0.229, 0.224, 0.225]
        self.normalize = transforms.Normalize(mean=self.mean, std=self.std)


        self.epoch = 0
        self.tau = 0

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

        self.imgs = imgs
        self.labels = labels

        self.n_imgs = len(self.imgs)
        if norm_age:
            self.labels = self.labels - min(self.labels)

        self.max_age = self.labels.max()
        self.min_age = self.labels.min()

    def __getitem__(self, item):
        base_img = np.asarray(self.imgs[item]).astype('uint8')
        base_img = self.transform(base_img)

        order_label, ref_idx = self.find_reference(self.labels[item], self.labels, min_rank=self.min_age,
                                              max_rank=self.max_age)
        ref_img = np.asarray(self.imgs[ref_idx]).astype('uint8')
        ref_img = self.transform(ref_img)

        one_hot_vector = self.generate_onehot_from_order(order_label)

        # order label generation
        # one_hot_order_label = self.get_one_hot_order_label_from_order(order_label)

        # gt ages
        base_age = self.labels[item]
        ref_age = self.labels[ref_idx]
        return base_img, ref_img, one_hot_vector, order_label, [base_age, ref_age], item


    def find_reference(self, base_rank, ref_ranks, min_rank=0, max_rank=32, epsilon=1e-4):
        order = np.random.randint(0, 3)
        ref_idx = -1
        debug_flag = 0
        while ref_idx == -1:
            if debug_flag == 3:
                raise ValueError(f'Failed to find reference... base_score: {base_rank}')
            if order == 0:  # base_rank > ref_rank
                ref_range_min = min_rank
                ref_range_max = base_rank - self.tau - epsilon
                candidates = get_indices_in_range([ref_range_min, ref_range_max], ref_ranks)
                if len(candidates) > 0:
                    ref_idx = candidates[np.random.choice(len(candidates), 1)[0]][0]
                else:
                    order = (order + 1) % 3
                    debug_flag += 1
                    continue
            elif order == 1:  # base_rank < ref_rank
                ref_range_min = base_rank + self.tau + epsilon
                ref_range_max = max_rank
                candidates = get_indices_in_range([ref_range_min, ref_range_max], ref_ranks)
                if len(candidates) > 0:
                    ref_idx = candidates[np.random.choice(len(candidates), 1)[0]][0]
                else:
                    order = (order + 1) % 3
                    debug_flag += 1
                    continue

            else:  # base_rank = ref_rank
                ref_range_min = base_rank - self.tau - epsilon
                ref_range_max = base_rank + self.tau + epsilon
                candidates = get_indices_in_range([ref_range_min, ref_range_max], ref_ranks)
                if len(candidates) > 0:
                    ref_idx = candidates[np.random.choice(len(candidates), 1)[0]][0]
                else:
                    order = (order + 1) % 3
                    debug_flag += 1
        return order, ref_idx

    def __len__(self):
        return len(self.labels)

    def generate_onehot_from_order(self, order):
        if order == 0:
            one_hot_vector = torch.tensor([1, 0], dtype=torch.float32)
        elif order == 1:
            one_hot_vector = torch.tensor([0, 1], dtype=torch.float32)
        elif order == 2:
            one_hot_vector = torch.tensor([0.5, 0.5], dtype=torch.float32)
        else:
            raise ValueError(f'order value {order} is out of expected range.')
        return one_hot_vector

def get_indices_in_range(search_range, ages):
    """find indices of values within range[0] <= x <= range[1]"""
    return np.argwhere(np.logical_and(search_range[0] <= ages, ages <= search_range[1]))



class Reference(Dataset):
    def __init__(self, base_data, ref_data, use_age=False, transform=None):
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

        self.ref_imgs, self.ref_labels = ref_data
        self.base_imgs, self.base_labels = base_data

        self.n_base_imgs = len(self.base_imgs)

        if use_age:
            self.base_labels = self.base_labels - min(self.base_labels)
            self.ref_labels = self.ref_labels - min(self.ref_labels)

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
        base_age = self.base_labels[item]
        ref_age = self.ref_labels[ref_idx]
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


def uniform_sample_nodes(labels, keep_ratio, n_rank=8):
    n_sample_total = len(labels)*keep_ratio
    n_per_rank = int(n_sample_total/n_rank)
    uniq_ranks = np.unique(labels)
    idxs = []
    print(f'sampling node with {keep_ratio} for each rank. (#Per_rank :{n_per_rank})')

    for r in uniq_ranks:
        r_idxs = np.argwhere(labels == r).flatten()
        if len(r_idxs) > n_per_rank:
            selected = np.random.choice(r_idxs, n_per_rank, replace=False)
        else:
            selected = np.random.choice(r_idxs, n_per_rank, replace=True)
        idxs.append(selected)
    idxs = np.concatenate(idxs)
    return idxs


def group_shuffle(pair_list, order_list):
    mapIndexPosition = list(zip(pair_list, order_list))
    random.shuffle(mapIndexPosition)
    pair_list, order_list = zip(*mapIndexPosition)
    return pair_list, order_list