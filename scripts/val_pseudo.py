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
    # hash_list = []
    # # lcfcn loss with affinity+shared+pseudo mask on seg dataset
    # hash_list += ['37bc7b4aa55e77592f10d60d2a9ebdc3']

    # # lcfcn loss with affinity+shared on seg dataset
    # hash_list += ['66ffec29b63f1ade0e7c79b23997d0b3']

    hash_list = ['a55d2c5dda331b1a0e191b104406dd1c',
                 '13b0f4e395b6dc5368f7965c20e75612',
                 'fcc1acac9ff5c2fa776d65ac76c3892b']

    # # lcfcn loss wiith shared on localization dataset
    # hash_list += ['a55d2c5dda331b1a0e191b104406dd1c']

    # # use this one a55d2c5dda331b1a0e191b104406dd1c for all the lcfcn results
    # hash_list += ['a55d2c5dda331b1a0e191b104406dd1c']

    # # lcfcn loss wiith affinity+shared on segmentation dataset
    # hash_list += ['9c7533a7c61f72919b9afd749dbb88e1']



    # lcfcn loss with_affinity=True
    # hash_id = '84ced18cf5c1fb3ad5820cc1b55a38fa'
    
    # point level
    # hash_id = 'd7040c9534b08e765f48c6cb034b26b2'

    # LCFCN
    # hash_id = 'bcba046296675e9e3af5cd9f353d217b'
    for hash_id in  hash_list:
        exp_dict = hu.load_json(os.path.join(savedir_base, hash_id, 'exp_dict.json'))
        fname = '.tmp/train_dict_%s.pkl' % hash_id
        datadir = '/mnt/public/datasets/DeepFish/'
        if os.path.exists(fname) and 0: 
            train_dict = hu.load_pkl(fname)
        else:
            split = 'train'
            exp_dict['model']['count_mode'] = 0
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

            # Model
            # ==================
            model = models.get_model(model_dict=exp_dict['model'],
                                    exp_dict=exp_dict,
                                    train_set=train_set).cuda()

            model_path = os.path.join(savedir_base, hash_id, 'model_best.pth')

            # load best model
            model.load_state_dict(hu.torch_load(model_path))
            train_dict = model.val_on_loader(train_loader)
            
            hu.save_pkl(fname, train_dict)
        print('results for hash: %s' % hash_id)
        pprint.pprint(train_dict)
        # loop over the val_loader and saves image
        # for i, batch in enumerate(train_loader):
        #     image_name = batch['meta'][0]['name']
        #     savedir_image = os.path.join('/mnt/public/predictions', "pseudo_masks", hash_id, split, "%s.png" % (image_name))
        #     img_pred = model.predict_on_batch(batch)
        #     hu.save_image(savedir_image, img_pred)
            # print('saved: %d/%d' %(i, len(train_loader)))

