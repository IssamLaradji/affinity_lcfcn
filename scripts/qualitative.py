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
    datadir = '/mnt/public/datasets/DeepFish/'
    # on localiization
    hash_list = [#Point loss
                 '63f29eec3dbe1e03364f198ed7d4b414',
                 # LCFCN
                 'a55d2c5dda331b1a0e191b104406dd1c',
                 #A-LCFCN
                 '13b0f4e395b6dc5368f7965c20e75612',
                 # A-LCFCN+PM
                 'fcc1acac9ff5c2fa776d65ac76c3892b']

    main_hash = 'fcc1acac9ff5c2fa776d65ac76c3892b'
    exp_dict = hu.load_json(os.path.join(savedir_base, main_hash, 'exp_dict.json'))
    exp_dict['count_mode'] = 0
    test_set = datasets.get_dataset(dataset_dict=exp_dict["dataset"],
                                            split='test',
                                            datadir=datadir,
                                            exp_dict=exp_dict,
                                            dataset_size=exp_dict['dataset_size'])
    test_loader = DataLoader(test_set,
                                    # sampler=val_sampler,
                                    batch_size=1,
                                    collate_fn=ut.collate_fn,
                                    num_workers=0)

    for i, batch in enumerate(test_loader):
        points = (batch['points'].squeeze() == 1).numpy()
        if points.sum() == 0:
            continue
        savedir_image = os.path.join('.tmp/qualitative/%d.png' % (i))
        img = hu.denormalize(batch['images'], mode='rgb')
        img_org = np.array(hu.save_image(savedir_image, img, mask=batch['masks'].numpy(),  return_image=True))

        img_list = [img_org]
        with torch.no_grad():
            for hash_id in hash_list:
                score_path = os.path.join(savedir_base, hash_id, 'score_list_best.pkl')
                score_list = hu.load_pkl(score_path)
                
                exp_dict = hu.load_json(os.path.join(savedir_base, hash_id, 'exp_dict.json'))
                print(i, exp_dict['model']['loss'], exp_dict['model'].get('with_affinity'), 'score:', score_list[-1]['test_class1'])
                
                model = models.get_model(model_dict=exp_dict['model'],
                                            exp_dict=exp_dict,
                                            train_set=test_set).cuda()

                model_path = os.path.join(savedir_base, hash_id, 'model_best.pth')
                model.load_state_dict(hu.torch_load(model_path), with_opt=False)
                mask_pred = model.predict_on_batch(batch)
                img_pred = np.array(hu.save_image(savedir_image, img, mask=mask_pred,  return_image=True))
                img_list += [img_pred]

        img_cat = np.concatenate(img_list, axis=1)
        hu.save_image(savedir_image, img_cat)

          