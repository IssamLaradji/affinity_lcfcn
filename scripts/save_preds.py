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
    exp_dict = {
        "batch_size": 1,
        "dataset": {
            "n_classes": 2,
            "name": "JcuFish"
        },
        "dataset_size": {
            "train": "all",
            "val": "all"
        },
        "lr": 1e-05,
        "max_epoch": 1000,
        "model": {
            "base": "fcn8_vgg16",
            "loss": "lcfcn_loss",
            "n_channels": 3,
            "n_classes": 2,
            "name": "semseg",
            # "with_affinity": 1
        },
        "num_channels": 1,
        "optimizer": "adam"
    }

    # lcfcn loss with_affinity=True
    # hash_dir = '84ced18cf5c1fb3ad5820cc1b55a38fa'

    # point level
    # hash_dir = 'd7040c9534b08e765f48c6cb034b26b2'

    # LCFCN
    hash_dir = 'bcba046296675e9e3af5cd9f353d217b'

    savedir = '/mnt/public/predictions'
    datadir = '/mnt/public/datasets/DeepFish/'

    split = 'test'
    test_set = datasets.get_dataset(dataset_dict=exp_dict["dataset"],
                                    split=split,
                                    datadir=datadir,
                                    exp_dict=exp_dict,
                                    dataset_size=exp_dict['dataset_size'])
    test_loader = DataLoader(test_set,
                             # sampler=val_sampler,
                             batch_size=1,
                             collate_fn=ut.collate_fn,
                             num_workers=0)

    # Model
    # ==================
    model = models.get_model(model_dict=exp_dict['model'],
                             exp_dict=exp_dict,
                             train_set=test_set).cuda()

    model_path = '/mnt/public/results/toolkit/weak_supervision/%s/model_best.pth' % hash_dir

    # load best model
    model.load_state_dict(hu.torch_load(model_path))

    # loop over the val_loader and saves image
    for i, batch in enumerate(test_loader):
        savedir_image = os.path.join("%s" % savedir, "save_preds", "%s" % hash_dir, "%s" % split, "%d.png" % i)

        image = batch['images']
        original = hu.denormalize(image, mode='rgb')[0]
        gt = np.asarray(batch['masks'])

        image = F.interpolate(image, size=gt.shape[-2:], mode='bilinear', align_corners=False)
        img_pred = hu.save_image(savedir_image,
                                 original,
                                 mask=model.predict_on_batch(batch), return_image=True)

        img_gt = hu.save_image(savedir_image,
                               original,
                               mask=gt, return_image=True)
        # add text_on_image here
        img_gt = models.text_on_image('', np.array(img_gt), color=(0, 0, 0))
        img_pred = models.text_on_image('', np.array(img_pred), color=(0, 0, 0))

        if 'points' in batch:
            pts = (batch['points'][0].numpy().copy()).astype('uint8')

            pts[pts != 255] = 1
            pts[pts == 255] = 0
            img_gt = np.array(hu.save_image(savedir_image, img_gt / 255.,
                                            points=pts.squeeze(),
                                            radius=2, return_image=True))
        img_list = [np.array(img_gt), np.array(img_pred)]
        hu.save_image(savedir_image, np.hstack(img_list))
