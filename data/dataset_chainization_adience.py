import random
from copy import deepcopy
from PIL import Image

import pickle
import numpy as np

import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms


class ChainizedAdience(Dataset):
    def __init__(self, epoch, imgs, ages, labels, transform=None):
        super(Dataset, self).__init__()

        self.mean = [0.485, 0.456, 0.406]
        self.std = [0.229, 0.224, 0.225]
        self.normalize = transforms.Normalize(mean=self.mean, std=self.std)

        self.epoch = epoch
        self.epoch_gen = 0

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
        self.ages = ages
        self.labels = labels
        self.n_imgs = len(self.imgs)

        self.gt_ranks = deepcopy(self.labels)
        self.batch_sampling_prob = [1, 0]

    def __getitem__(self, item):
        case = np.random.choice([0, 1], 1, p=self.batch_sampling_prob)[0]

        if item + (self.epoch * self.n_imgs) >= self.n_pairs:
            print(f'pair list has been shuffled at epoch {self.epoch}')
            self.epoch = 0
            # shuffle_idx = np.random.permutation(np.arange(self.n_pairs))

            self.pair_list, self.order_list = self.group_shuffle(self.pair_list, self.order_list)
            # self.pseudo_pair_list, self.pseudo_order_list = self.group_shuffle(self.pseudo_pair_list, self.pseudo_order_list)

        if case == 0:  # gt pair
            # item = item + (self.epoch * self.n_imgs)
            item = np.random.randint(len(self.pair_list))
            base_idx, ref_idx = self.pair_list[item]
            order_label = self.order_list[item]

        elif case == 1:  # pseudo pair
            item = np.random.randint(self.n_pseudo_pairs)
            base_idx, ref_idx = self.pseudo_pair_list[item]
            order_label = self.pseudo_order_list[item]

        else:
            raise ValueError(f'** Case is out of range: {case}.')

        # reverse the order
        if np.random.rand(1) > 0.5:
            pass
        else:
            ref_idx, base_idx = base_idx, ref_idx
            order_label = self.reverse_order(order_label)

        base_img = np.asarray(self.imgs[base_idx]).astype('uint8')
        base_img = self.transform(base_img)

        ref_img = np.asarray(self.imgs[ref_idx]).astype('uint8')
        ref_img = self.transform(ref_img)

        # order label generation
        one_hot_order_label = self.get_one_hot_order_label_from_order(order_label)

        # gt ages
        base_age = self.ages[base_idx]
        ref_age = self.ages[ref_idx]
        return base_img, ref_img, one_hot_order_label, order_label, [base_age, ref_age], base_idx.astype(np.int32)

    def get_one_hot_order_label_from_order(self, order):
        if order == 0:
            one_hot_order_label = torch.tensor([1, 0], dtype=torch.float32)
        elif order == 1:
            one_hot_order_label = torch.tensor([0, 1], dtype=torch.float32)
        else:
            one_hot_order_label = torch.tensor([0.5, 0.5], dtype=torch.float32)

        return one_hot_order_label

    def reverse_order(self, order):
        if order == 0:
            order = 1
        elif order == 1:
            order = 0
        return order

    def __len__(self):
        return len(self.imgs)

    def get_one_hot_order_label(self, base_idx, ref_idx):
        base_rank = self.labels[base_idx]
        ref_rank = self.labels[ref_idx]

        if base_rank > ref_rank:
            one_hot_order_label = torch.tensor([1, 0], dtype=torch.float32)
            order_label = 0
        elif base_rank < ref_rank:
            one_hot_order_label = torch.tensor([0, 1], dtype=torch.float32)
            order_label = 1
        else:
            one_hot_order_label = torch.tensor([0.5, 0.5], dtype=torch.float32)
            order_label = 2 ################## TODO: CHECK !!!!!!!!!!!!!!!!!!!

        return one_hot_order_label, order_label

    def pseudo_to_one_hot_label(self, pseudo_label):
        if pseudo_label == 0:
            one_hot_order_label = torch.tensor([1, 0], dtype=torch.float32)
            order_label = 0

        elif pseudo_label == 1:
            one_hot_order_label = torch.tensor([0, 1], dtype=torch.float32)
            order_label = 1
        else:
            one_hot_order_label = torch.tensor([0.5, 0.5], dtype=torch.float32)
            order_label = 2

        return one_hot_order_label, order_label

    def update_pairs(self, annotated_pair_list, order_labels, generated_pair_list, pseudo_labels):
        self.pair_list = annotated_pair_list
        self.n_pairs = len(self.pair_list)
        self.order_list = order_labels

        self.pseudo_pair_list, self.pseudo_order_list = self.group_shuffle(generated_pair_list, pseudo_labels)
        self.n_pseudo_pairs = len(self.pseudo_pair_list)
        self.batch_sampling_prob = [0.2, 0.8]
        print(f'annotated pairs: {self.n_pairs}, generated pairs:{self.n_pseudo_pairs}, total: {self.n_pairs+self.n_pseudo_pairs}')

    def group_shuffle(self, pair_list, order_list):
        mapIndexPosition = list(zip(pair_list, order_list))
        random.shuffle(mapIndexPosition)
        pair_list, order_list = zip(*mapIndexPosition)
        return pair_list, order_list
