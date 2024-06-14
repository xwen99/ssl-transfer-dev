#!/usr/bin/env python
# coding: utf-8

import os
import argparse
from pprint import pprint

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision import datasets

import numpy as np
from tqdm import tqdm

from datasets.few_shot_dataset import SetDataManager

from datasets.dtd import DTD
from datasets.pets import Pets
from datasets.cars import Cars
from datasets.food import Food
from datasets.sun397 import SUN397
from datasets.voc2007 import VOC2007
from datasets.flowers import Flowers
from datasets.aircraft import Aircraft
from datasets.birdsnap import Birdsnap
from datasets.caltech101 import Caltech101

import models
from config import *

class FewShotTester():
    def __init__(self, backbone, dataloader, n_way, n_support, n_query, iter_num, device):
        self.backbone = backbone
        self.protonet = ProtoNet(self.backbone)
        self.dataloader = dataloader

        self.n_way = n_way
        self.n_support = n_support
        self.n_query = n_query
        self.iter_num = iter_num
        self.device = device

    def test(self):
        loss, acc, std = self.evaluate(self.protonet, self.dataloader, self.n_support, self.n_query, self.iter_num)
        std = 1.96 * std / np.sqrt(self.iter_num)
        print('Test Acc = %4.2f%% +- %4.2f%%' %(acc, std))
        return acc, std

    def extract_episode(self, data, n_support, n_query):
        # data: N x C x H x W
        n_examples = data.size(1)

        if n_query == -1:
            n_query = n_examples - n_support

        example_inds = torch.randperm(n_examples)[:(n_support+n_query)]
        support_inds = example_inds[:n_support]
        query_inds = example_inds[n_support:]

        xs = data[:, support_inds]
        xq = data[:, query_inds]

        return {
            'xs': xs.to(self.device),
            'xq': xq.to(self.device)
        }

    def evaluate(self, model, data_loader, n_support, n_query, iter_num, desc=None):
        model.eval()

        loss_all = []
        acc_all = []

        if desc is not None:
            data_loader = tqdm(data_loader, desc=desc)

        with torch.no_grad():
            for data, targets in tqdm(data_loader, desc=f'Few-shot test episodes'):
                sample = self.extract_episode(data, n_support, n_query)
                loss_val, acc_val = model.loss(sample)
                loss_all.append(loss_val.item())
                acc_all.append(acc_val.item() * 100.)

        loss = np.mean(loss_all)
        acc = np.mean(acc_all)
        std = np.std(acc_all)

        return loss, acc, std


# Model classes and functions

def euclidean_dist(x, y):
    # x: N x D
    # y: M x D
    n = x.size(0)
    m = y.size(0)
    d = x.size(1)
    assert d == y.size(1)

    x = x.unsqueeze(1).expand(n, m, d)
    y = y.unsqueeze(0).expand(n, m, d)

    return torch.pow(x - y, 2).sum(2)


class Flatten(nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, x):
        return x.view(x.size(0), -1)


class ProtoNet(nn.Module):
    def __init__(self, encoder):
        super(ProtoNet, self).__init__()
        
        self.encoder = encoder

    def loss(self, sample):
        with torch.no_grad():
            xs = Variable(sample['xs']) # support
            xq = Variable(sample['xq']) # query

            n_class = xs.size(0)
            assert xq.size(0) == n_class
            n_support = xs.size(1)
            n_query = xq.size(1)

            target_inds = torch.arange(0, n_class).view(n_class, 1, 1).expand(n_class, n_query, 1).long()
            target_inds = Variable(target_inds, requires_grad=False)

            if xq.is_cuda:
                target_inds = target_inds.cuda()

            x = torch.cat([xs.view(n_class * n_support, *xs.size()[2:]),
                           xq.view(n_class * n_query, *xq.size()[2:])], 0)

            z = self.encoder.forward(x)
            z_dim = z.size(-1)

            z_proto = z[:n_class*n_support].view(n_class, n_support, z_dim).mean(1)
            zq = z[n_class*n_support:]

            dists = euclidean_dist(zq, z_proto)

            log_p_y = F.log_softmax(-dists, dim=1).view(n_class, n_query, -1)

            loss_val = -log_p_y.gather(2, target_inds).squeeze().view(-1).mean()

            _, y_hat = log_p_y.max(2)
            acc_val = torch.eq(y_hat, target_inds.squeeze()).float().mean()

        return loss_val, acc_val


class ResNetBackbone(nn.Module):
    def __init__(self, model_path):
        super().__init__()
        self.model = models.rn50()
        state_dict = torch.load(model_path)['state_dict']
        state_dict = {k.replace('module.', '').replace('visual.', ''): v for k, v in state_dict.items()}
        msg = self.model.load_state_dict(state_dict, strict=False)
        print(msg)
        self.model.eval()

    def forward(self, x):
        return self.model(x)


# name: {class, root, num_classes (not necessary here), metric}
FEW_SHOT_DATASETS = {
    'aircraft': [Aircraft, AIRCRAFT_ROOT, 100, 'mean per-class accuracy'],
    'caltech101': [Caltech101, CALTECH101_ROOT, 102, 'mean per-class accuracy'],
    'cars': [Cars, CARS_ROOT, 196, 'accuracy'],
    'cifar10': [datasets.CIFAR10, CIFAR10_ROOT, 10, 'accuracy'],
    'cifar100': [datasets.CIFAR100, CIFAR100_ROOT, 100, 'accuracy'],
    'dtd': [DTD, DTD_ROOT, 47, 'accuracy'],
    'flowers': [Flowers, FLOWERS_ROOT, 102, 'mean per-class accuracy'],
    'food': [Food, FOOD_ROOT, 101, 'accuracy'],
    'pets': [Pets, PETS_ROOT, 37, 'mean per-class accuracy'],
    'sun397': [SUN397, SUN397_ROOT, 397, 'accuracy'],
}


# Main code
def main(args, dataset=None):
    pprint(args)

    if args.dataset != 'all':
        dataset = args.dataset
    elif dataset is None:
        raise ValueError('dataset must be specified if args.dataset is "all"')

    # load dataset
    dset, data_dir, num_classes, metric = FEW_SHOT_DATASETS[dataset]
    datamgr = SetDataManager(dset, data_dir, num_classes, args.image_size, normalisation=args.norm,
                             n_episode=args.iter_num, n_way=args.n_way, n_support=args.n_support, n_query=args.n_query)
    dataloader = datamgr.get_data_loader(aug=False, normalise=True)

    # load pretrained model
    model = ResNetBackbone(args.model)
    model = model.to(args.device)

    # evaluate model on dataset by protonet few-shot-learning evaluation
    tester = FewShotTester(model, dataloader, args.n_way, args.n_support, args.n_query, args.iter_num, args.device)
    test_acc, test_std = tester.test()

    import csv
    with open(os.path.join(os.path.dirname(args.model), 'fewshot.csv'), 'a') as f:
        writer = csv.writer(f, delimiter='\t')
        if os.path.getsize(os.path.join(os.path.dirname(args.model), 'fewshot.csv')) == 0:
            writer.writerow(['dataset', 'n_way', 'n_support', 'n_query', 'iter_num', 'test_acc', 'test_std'])
        writer.writerow([dataset, args.n_way, args.n_support, args.n_query, args.iter_num, test_acc, test_std])

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Evaluate pretrained self-supervised model on few-shot recognition.')
    parser.add_argument('-m', '--model', type=str, default='deepcluster-v2',
                        help='name of the pretrained model to load and evaluate (deepcluster-v2 | supervised)')
    parser.add_argument('-d', '--dataset', type=str, default='eurosat', help='name of the dataset to evaluate on')
    parser.add_argument('-i', '--image-size', type=int, default=224, help='the size of the input images')
    parser.add_argument('--n-way', type=int, default=5, help='the number of classes per episode (n-way) in few-shot evaluation')
    parser.add_argument('--n-support', type=int, default=5, help='the number of images per class for fitting (n-support) in few-shot evaluation')
    parser.add_argument('--n-query', type=int, default=15, help='the number of images per class for testing (n-query) in few-shot evaluation')
    parser.add_argument('--iter-num', type=int, default=600, help='the number of testing episodes in few-shot evaluation')
    parser.add_argument('-n', '--norm', type=str, default='imagenet', help='normalization methods')
    parser.add_argument('--device', type=str, default='cuda', help='CUDA or CPU training (cuda | cpu)')
    args = parser.parse_args()

    if args.dataset == 'all':
        for dataset in FEW_SHOT_DATASETS.keys():
            main(args, dataset)
    else:
        main(args)