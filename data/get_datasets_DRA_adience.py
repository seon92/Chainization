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

    loader_dict['train_lb'] = DataLoader(AdiencePartialEdgeLabeled(tr_imgs, tr_ages, opt.info_ratio),
                                         batch_size=opt.batch_size, shuffle=True, drop_last=True, num_workers=opt.num_workers)

    loader_dict['test'] = DataLoader(AdienceReference([te_imgs, te_ages], [tr_imgs, tr_ages]),
                                     batch_size=opt.batch_size, shuffle=False, drop_last=False, num_workers=opt.num_workers)


    loader_dict['vanilla_ulb'] = DataLoader(AdienceVanilla(tr_imgs, tr_ages),
                                            batch_size=opt.batch_size, shuffle=False, drop_last=False, num_workers=opt.num_workers)
    return loader_dict



class AdienceVanilla(Dataset):
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
        self.n_imgs = len(self.imgs)

    def __getitem__(self, item):
        img = np.asarray(self.imgs[item]).astype('uint8')
        img = self.transform(img)

        return img, item

    def __len__(self):
        return len(self.imgs)


class AdiencePartialEdgeLabeled(Dataset):
    def __init__(self, imgs, labels, info_ratio, norm_age=False, transform=None):
        super(Dataset, self).__init__()

        self.mean = [0.485, 0.456, 0.406]
        self.std = [0.229, 0.224, 0.225]
        self.normalize = transforms.Normalize(mean=self.mean, std=self.std)

        self.info_ratio = info_ratio
        self.epoch = 0

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
        self.tau = 0

        self.n_pairs = int((self.info_ratio/100) * (self.n_imgs * (self.n_imgs - 1) / 2))
        print(f'{self.n_pairs} pairs will be generated')
        self.generate_training_pairs()

        self.generate_order_list()
        # self.build_digraph()
        # self.data_in_epoch = self.n_imgs
        # if self.n_pairs < self.n_imgs:
        #     self.data_in_epoch = self.n_pairs

    def __getitem__(self, item):
        base_idx , ref_idx = self.pair_list[item]
        order_label = self.order_list[item]
        rng = np.random.default_rng()

        if rng.random(1) > 0.5:
            base_idx, ref_idx = ref_idx, base_idx
            order_label = self.reverse_order(order_label)
        base_img = np.asarray(self.imgs[base_idx]).astype('uint8')
        base_img = self.transform(base_img)

        ref_img = np.asarray(self.imgs[ref_idx]).astype('uint8')
        ref_img = self.transform(ref_img)

        one_hot_vector = self.generate_onehot_from_order(order_label)

        # order label generation
        # one_hot_order_label = self.get_one_hot_order_label_from_order(order_label)

        # gt ages
        base_age = self.labels[base_idx]
        ref_age = self.labels[ref_idx]
        return base_img, ref_img, one_hot_vector, order_label, [base_age, ref_age], item

    def __len__(self):
        return self.n_pairs

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

    def generate_training_pairs(self):
        if self.n_pairs > self.n_imgs:
            base_indices = np.random.choice(self.n_imgs, self.n_pairs-self.n_imgs, replace=True)
            base_indices = np.concatenate([base_indices, np.arange(self.n_imgs)])
        else:
            base_indices = np.random.choice(self.n_imgs, self.n_pairs, replace=True)
        ref_indices = np.random.choice(self.n_imgs, self.n_pairs, replace=True)

        pair_list = np.concatenate([base_indices.reshape(-1, 1), ref_indices.reshape(-1, 1)], axis=-1)
        pair_list = np.sort(pair_list, axis=-1)

        pair_set = set([(x, y) for x, y in pair_list])
        cnt = 0
        while len(pair_set) < self.n_pairs:
            cnt += 1
            x, y = np.random.choice(self.n_imgs, 2)
            if x < y:
                pair_set.add((x, y))
            elif x > y:
                pair_set.add((y, x))
            else: #not using self pair
                continue
        print(f'while loop is iterated for {cnt} times')
        self.pair_list = list(pair_set)
        # random.shuffle(self.pair_list)
        self.n_pairs = len(self.pair_list)
        print(f'{self.n_pairs} pairs have been generated!')


    def build_digraph(self):
        G = nx.DiGraph()
        self.mapping = np.arange(self.n_imgs)  #map img_idx -> node_idx
        # add node and its properties to graph
        for i in range(self.n_imgs):
            G.add_node(i, ages=[self.labels[i]], img_inds=[i])

        for idx, (x, y) in enumerate(self.pair_list):
            order = self.order_list[idx]
            if order == 0:
                G.add_edge(self.mapping[y], self.mapping[x])
            elif order == 1:
                G.add_edge(self.mapping[x], self.mapping[y])
            else:
                # check x,y are self pair
                if self.mapping[x] == self.mapping[y]:
                    continue
                else:
                    G = self.merge_nodes(G, self.mapping[x], self.mapping[y])
                    # img_inds_in_deleted_node = np.argwhere(self.mapping==self.mapping[x]).flatten()
                    try:
                        img_inds = G.nodes[self.mapping[y]]['img_inds']
                    except:
                        print('??')
                    self.mapping[img_inds] = self.mapping[y]
        self.G = G

    def merge_nodes(self, G, nodeA, nodeB):
        """
        G의 node A를 node B에 합쳐준다.
        """
        #node update
        G.nodes[nodeB]['ages'] += G.nodes[nodeA]['ages']
        G.nodes[nodeB]['img_inds'] += G.nodes[nodeA]['img_inds']

        #edge update
        for A_nbr in G[nodeA]:
            if G.has_edge(A_nbr, nodeB):
                # 이미 A_nbr이 nodeB와 연결되어 있다면, edge의 weight만 update하면 됨.
                # G[nodeB][A_nbr]['weight'] += G[nodeA][A_nbr]['weight']
                continue

            else:
                # 연결되어 있지 않으므로 새로운 edge를 만들어주어야 함.
                if A_nbr is not nodeB:  # self-loop가 아닌지 확인
                    G.add_edge(nodeB, A_nbr)
                else:  # self-loop
                    continue
        G.remove_node(nodeA)
        return G


    def generate_order_list(self, ):
        self.order_list = []
        for base_idx, ref_idx in self.pair_list:
            base_rank = self.labels[base_idx]
            ref_rank = self.labels[ref_idx]

            if base_rank > ref_rank:
                self.order_list.append(0)
            elif base_rank < ref_rank:
                self.order_list.append(1)
            else:
                self.order_list.append(2)


    def reverse_order(self, order):
        if order == 0:
            rev_order = 1
        elif order == 1:
            rev_order = 0
        else:
            rev_order = 2
        return rev_order

def get_indices_in_range(search_range, ages):
    """find indices of values within range[0] <= x <= range[1]"""
    return np.argwhere(np.logical_and(search_range[0] <= ages, ages <= search_range[1]))


class AdienceReference(Dataset):
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