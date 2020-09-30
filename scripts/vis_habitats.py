import sys, os

path = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.insert(0, path)

from haven import haven_chk as hc
from haven import haven_results as hr
from haven import haven_utils as hu
import torch
import torchvision
import tqdm
import pandas as pd
import pprint
import itertools
import os
import pylab as plt
import time
import numpy as np

from src import models
from src import datasets
from src import utils as ut

import argparse

from torch.utils.data import sampler
from torch.utils.data.sampler import RandomSampler
from torch.backends import cudnn
from torch.nn import functional as F
from torch.utils.data import DataLoader

cudnn.benchmark = True

if __name__ == "__main__":
    savedir_base = '/mnt/public/results/toolkit/weak_supervision'
    hash_list = ['a55d2c5dda331b1a0e191b104406dd1c']
    # LCFCN
    # hash_id = 'bcba046296675e9e3af5cd9f353d217b'
    for hash_id in  hash_list:
        exp_dict = hu.load_json(os.path.join(savedir_base, hash_id, 'exp_dict.json'))
        datadir = '/mnt/public/datasets/DeepFish/'
        split = 'train'
        train_set = datasets.get_dataset(dataset_dict=exp_dict["dataset"],
                                        split=split,
                                        datadir=datadir,
                                        exp_dict=exp_dict,
                                        dataset_size=exp_dict['dataset_size'])
        train_loader = DataLoader(train_set,
                                # sampler=val_sampler,
                                batch_size=1,
                                collate_fn=ut.collate_fn,
                                num_workers=0)
        for i, batch in enumerate(train_loader):
            points = (batch['points'].squeeze() == 1).numpy()
            if points.sum() == 0:
                continue
            savedir_image = os.path.join('.tmp/habitats/%s/%d.png' % (batch['meta'][0]['habitat'], i))
            img = hu.denormalize(batch['images'], mode='rgb')
            # img_pred = model.predict_on_batch(batch)
            hu.save_image(savedir_image, img, points=points, radius=1)