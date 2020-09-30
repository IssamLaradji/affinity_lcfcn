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
import pandas as pd
cudnn.benchmark = True

if __name__ == "__main__":
    savedir_base = '/mnt/public/results/toolkit/weak_supervision'
   
    hash_list = ['b04090f27c7c52bcec65f6ba455ed2d8',
                '6d4af38d64b23586e71a198de2608333',
                '84ced18cf5c1fb3ad5820cc1b55a38fa',
                '63f29eec3dbe1e03364f198ed7d4b414',
                '017e7441c2f581b6fee9e3ac6f574edc']
    datadir = '/mnt/public/datasets/DeepFish/'
    
    score_list = []
    for hash_id in hash_list:
        fname = os.path.join('/mnt/public/predictions/game/%s.pkl' % hash_id)
        exp_dict = hu.load_json(os.path.join(savedir_base, hash_id, 'exp_dict.json'))
        if os.path.exists(fname):
            print('FOUND:', fname)
            val_dict = hu.load_pkl(fname)
        else:
            
            train_set = datasets.get_dataset(dataset_dict=exp_dict["dataset"],
                                            split='train',
                                            datadir=datadir,
                                            exp_dict=exp_dict,
                                            dataset_size=exp_dict['dataset_size'])

            test_set = datasets.get_dataset(dataset_dict=exp_dict["dataset"],
                                            split='test',
                                            datadir=datadir,
                                            exp_dict=exp_dict,
                                            dataset_size=exp_dict['dataset_size'])

            test_loader = DataLoader(test_set,
                                        batch_size=1,
                                        collate_fn=ut.collate_fn,
                                        num_workers=0)
            pprint.pprint(exp_dict)
            # Model
            # ==================
            model = models.get_model(model_dict=exp_dict['model'],
                                        exp_dict=exp_dict,
                                        train_set=train_set).cuda()

            model_path = os.path.join(savedir_base, hash_id, 'model_best.pth')

            # load best model
            model.load_state_dict(hu.torch_load(model_path))
            val_dict = model.val_on_loader(test_loader)

            val_dict['hash_id'] = hash_id
            pprint.pprint(val_dict)

            hu.save_pkl(fname, val_dict)

        val_dict['model'] = exp_dict['model']
        score_list += [val_dict]

    print(pd.DataFrame(score_list))


