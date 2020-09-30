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
from src.models import metrics

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
    hash_dct = {'b04090f27c7c52bcec65f6ba455ed2d8': 'Fully_Supervised',
                '6d4af38d64b23586e71a198de2608333': 'LCFCN',
                '84ced18cf5c1fb3ad5820cc1b55a38fa': 'LCFCN+Affinity_(ours)',
                '63f29eec3dbe1e03364f198ed7d4b414': 'Point-level_Loss ',
                '017e7441c2f581b6fee9e3ac6f574edc': 'Cross_entropy_Loss+pseudo-mask'}
    datadir = '/mnt/public/datasets/DeepFish/'

    score_list = []
    for hash_id in hash_list:
        fname = os.path.join('/mnt/public/predictions/habitat/%s.pkl' % hash_id)
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
            # loop over the val_loader and saves image
            # get counts
            habitats = []
            for i, batch in enumerate(test_loader):
                habitat = batch['meta'][0]['habitat']
                habitats += [habitat]
            habitats = np.array(habitats)

            val_dict = {}
            val_dict_lst = []
            for h in np.unique(habitats):
                val_meter = metrics.SegMeter(split=test_loader.dataset.split)

                for i, batch in enumerate(tqdm.tqdm(test_loader)):
                    habitat = batch['meta'][0]['habitat']
                    if habitat != h:
                        continue

                    val_meter.val_on_batch(model, batch)
                    score_dict = val_meter.get_avg_score()
                    pprint.pprint(score_dict)

                val_dict[h] = val_meter.get_avg_score()
                val_dict_dfc = pd.DataFrame([val_meter.get_avg_score()])
                val_dict_dfc.insert(0, "Habitat", h, True)
                val_dict_dfc.rename(
                    columns={'test_score': 'mIoU', 'test_class0': 'IoU class 0', 'test_class1': 'IoU class 1',
                             'test_mae': 'MAE', 'test_game': 'GAME'}, inplace=True)
                val_dict_lst.append(val_dict_dfc)
                val_dict_df = pd.concat(val_dict_lst, axis=0)
                val_dict_df.to_csv(os.path.join('/mnt/public/predictions/habitat/', "%s_habitat_score_df.csv" % hash_id),
                                   index=False)
                val_dict_df.to_latex(os.path.join('/mnt/public/predictions/habitat/', "%s_habitat_score_df.tex" % hash_id),
                                     index=False, caption=hash_dct[hash_id], label=hash_dct[hash_id])

            hu.save_pkl(fname, val_dict)

        val_dict['model'] = exp_dict['model']
        score_list += [val_dict]

    print(pd.DataFrame(score_list))
    # score_df = pd.DataFrame(score_list)
    # score_df.to_csv(os.path.join('/mnt/public/predictions/habitat/', "habitat_score_df.csv"))
    # score_df.to_latex(os.path.join('/mnt/public/predictions/habitat/', "habitat_score_df.tex"))

