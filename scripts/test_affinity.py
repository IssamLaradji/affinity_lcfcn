import sys, os
path = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.insert(0, path)

import torch
import torchvision
import tqdm
import pandas as pd
import pprint
import itertools
import os
import pylab as plt
import exp_configs
import time
import numpy as np

from src import models
from src import datasets
from src import utils as ut

from src import misc
import argparse

from torch.utils.data import sampler
from torch.utils.data.sampler import RandomSampler
from torch.backends import cudnn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from src import datasets
# from src import optimizers 
import torchvision
from src import models
cudnn.benchmark = True

from haven import haven_utils as hu
from haven import haven_img as hi
from haven import haven_results as hr
from haven import haven_chk as hc
# from src import looc_utils as lu
from PIL import Image
import scipy.io
from src import models
from src import utils as ut
import exp_configs

import argparse
import numpy as np
import time
import cv2, pprint
from PIL import Image
import torch
from SEAM.infer_SEAM import infer_SEAM
from SEAM.infer_aff import infer_aff
from SEAM.network import resnet38_SEAM, resnet38_aff


def get_indices_in_radius(height, width, radius=5):
    search_dist = []
    for x in range(1, radius):
        search_dist.append((0, x))

    for y in range(1, radius):
        for x in range(-radius+1, radius):
            if x*x + y*y < radius*radius:
                search_dist.append((y, x))

    full_indices = np.reshape(np.arange(0, height * width, dtype=np.int64),
                              (height, width))
    radius_floor = radius-1
    cropped_height = height - radius_floor
    cropped_width = width - 2 * radius_floor

    indices_from = np.reshape(full_indices[:-radius_floor, radius_floor:-radius_floor], [-1])

    indices_from_to_list = []

    for dy, dx in search_dist:

        indices_to = full_indices[dy:dy + cropped_height, radius_floor + dx:radius_floor + dx + cropped_width]
        indices_to = np.reshape(indices_to, [-1])

        indices_from_to = np.stack((indices_from, indices_to), axis=1)

        indices_from_to_list.append(indices_from_to)

    concat_indices_from_to = np.concatenate(indices_from_to_list, axis=0)

    return concat_indices_from_to

def get_affinity_labels(segm_map, indices_from, indices_to, n_classes=2):
    # _, n_classes, _, _ = segm_map.shape
    segm_map_flat = np.reshape(segm_map, -1)

    segm_label_from = np.expand_dims(segm_map_flat[indices_from], axis=0)
    segm_label_to = segm_map_flat[indices_to]

    valid_label = np.logical_and(np.less(segm_label_from, n_classes), np.less(segm_label_to, n_classes))

    equal_label = np.equal(segm_label_from, segm_label_to)

    pos_affinity_label = np.logical_and(equal_label, valid_label)

    bg_pos_affinity_label = np.logical_and(pos_affinity_label, np.equal(segm_label_from, 0)).astype(np.float32)
    fg_pos_affinity_label = np.logical_and(pos_affinity_label, np.greater(segm_label_from, 0)).astype(np.float32)

    neg_affinity_label = np.logical_and(np.logical_not(equal_label), valid_label).astype(np.float32)

    return torch.from_numpy(bg_pos_affinity_label), torch.from_numpy(fg_pos_affinity_label), \
            torch.from_numpy(neg_affinity_label)
            



if __name__ == "__main__":
    exp_dict = {'batch_size': 1,
                'dataset': {'n_classes': 2, 'name': 'JcuFish'},
                'dataset_size': {'train': 'all', 'val': 'all'},
                'lr': 1e-06,
                'max_epoch': 100,
                'model': {'base': 'fcn8_vgg16',
                        'loss': 'point_level',
                        'n_channels': 3,
                        'n_classes': 2,
                        'name': 'semseg'},
                'num_channels': 1,
                'optimizer': 'adam'}
    pprint.pprint(exp_dict)
    train_set = datasets.get_dataset(dataset_dict=exp_dict["dataset"],
                                     split="train",
                                     datadir='/mnt/public/datasets/DeepFish',
                                     exp_dict=exp_dict,
                                     dataset_size=exp_dict['dataset_size'])

    model_seam = resnet38_SEAM.Net().cuda()
    model_seam.load_state_dict(torch.load(os.path.join('/mnt/public/weights', 'resnet38_SEAM.pth')))

    model_aff = resnet38_aff.Net().cuda()
    model_aff.load_state_dict(torch.load(os.path.join('/mnt/public/weights', 'resnet38_aff_SEAM.pth')), strict=False)

    # ut.generate_seam_segmentation(train_set,
    #                               path_base='/mnt/datasets/public/issam/seam',
    #                             #   path_base='D:/Issam/SEAM_model/'
    #                               )
    # stop
    model = models.get_model(model_dict=exp_dict['model'], exp_dict=exp_dict, train_set=train_set).cuda()
    exp_id = hu.hash_dict(exp_dict)
    fname = os.path.join('/mnt/public/results/toolkit/weak_supervision', exp_id, 'model.pth')
    model.model_base.load_state_dict(torch.load(fname)['model'], strict=False)
    
    for k in range(5):
        batch_id = np.where(train_set.labels)[0][k]
        batch = ut.collate_fn([train_set[batch_id]])
        logits = F.softmax(model.model_base.forward(batch['images'].cuda()), dim=1)
   
        img = batch['images'].cuda()
        logits_new = model_aff.apply_affinity( batch['images'], logits, crf=0)
        
        i1 = hu.save_image('old.png', 
                    img=hu.denormalize(img, mode='rgb'), 
                    mask=logits.argmax(dim=1).cpu().numpy(), return_image=True)

        i2 = hu.save_image('new.png', 
                    img=hu.denormalize(img, mode='rgb'), 
                    mask=logits_new.argmax(dim=1).cpu().numpy(), return_image=True)
        hu.save_image('tmp/tmp%d.png' % k, np.concatenate([np.array(i1), np.array(i2)], axis=1))      
        print('saved %d' %k)
        