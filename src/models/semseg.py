# -*- coding: utf-8 -*-

import os, pprint, tqdm
import numpy as np
from haven import haven_utils as hu 
import torch
import torch.nn as nn
import torch.nn.functional as F
from src import utils as ut
from src import models
from . import losses
from src.modules.lcfcn import lcfcn_loss
from skimage import segmentation
from skimage.segmentation import expand_labels
from skimage.color import label2rgb

from . import optimizers, metrics, networks

DEVICE = 'cpu'
if torch.cuda.is_available():
    DEVICE = 'cuda'

def distance(x1, x2):
    diff = torch.abs(x1 - x2)
    return torch.pow(diff, 2).sum(dim=1)
class SemSeg(torch.nn.Module):
    def __init__(self, exp_dict, train_set):
        super().__init__()
        self.exp_dict = exp_dict
        self.train_hashes = set()
        self.n_classes = self.exp_dict['model'].get('n_classes', 1)

        
        self.first_time = True
        self.epoch = 0
        self.train_set = train_set
        self.init_model()
        

    def init_model(self):
        self.model_base = networks.get_network(self.exp_dict['model']['base'],
                                              n_classes=self.n_classes,
                                              exp_dict=self.exp_dict)
        
    

        self.cuda()
        self.opt = optimizers.get_optimizer(self.exp_dict['optimizer'], self.model_base, self.exp_dict)


    def get_state_dict(self):
        state_dict = {"model": self.model_base.state_dict(),
                      "opt": self.opt.state_dict(),
                      'epoch':self.epoch}

        return state_dict

    def load_state_dict(self, state_dict, with_opt=True):
        self.model_base.load_state_dict(state_dict["model"], strict=False)
        if with_opt:
            self.opt.load_state_dict(state_dict["opt"])
        self.epoch = state_dict['epoch']

    def train_on_loader(self, train_loader):
        
        self.train()
        self.epoch += 1
        n_batches = len(train_loader)

        pbar = tqdm.tqdm(desc="Training", total=n_batches, leave=False)
        train_monitor = TrainMonitor()
    
        for batch in train_loader:
            # for i in range(50): 
            #     score_dict = self.train_on_batch(batch)
            #     print(i, 'loss:', print(score_dict['train_loss']))
            score_dict = self.train_on_batch(batch)
            train_monitor.add(score_dict)
            msg = ' '.join(["%s: %.3f" % (k, v) for k,v in train_monitor.get_avg_score().items()])
            pbar.set_description('Training - %s' % msg)
            pbar.update(1)
            
        pbar.close()

        return train_monitor.get_avg_score()

    def train_on_batch(self, batch):
        self.train()
        # add to seen images
        for m in batch['meta']:
            self.train_hashes.add(m['hash'])

        self.opt.zero_grad()

        images = batch["images"].cuda()
        
        # compute loss
        loss_name = self.exp_dict['model']['loss']
        loss = 0

        if batch['meta'][0]['index'] in self.train_set.full_images:
            assert batch["masks"].sum() > 0
            logits = self.model_base(images)
            # full supervision
            loss = losses.compute_cross_entropy(images, logits, masks=batch["masks"].cuda())


        elif loss_name in 'cross_entropy':
            logits = self.model_base(images)
            # full supervision
            loss = losses.compute_cross_entropy(images, logits, masks=batch["masks"].cuda())

        elif loss_name in 'pseudo_mask':
            from src.datasets import jcu_fish
            logits = self.model_base(images, return_cam=0)
            # logits, logits_aff = self.model_base(images, return_cam=True)
            # full supervision
            if self.exp_dict['model'].get('count_mode'):
                hash_id = '13b0f4e395b6dc5368f7965c20e75612'
            else:
                hash_id = '9c7533a7c61f72919b9afd749dbb88e1'
            path = '/mnt/public/predictions/pseudo_masks/%s/train' % hash_id
            masks = jcu_fish.binary_loader(os.path.join(path, batch['meta'][0]['name']+'.png'))
            masks = np.array(self.train_set.gt_transform(masks))
            masks[masks==255] = 1
            masks = torch.as_tensor(masks)
            # print(masks.unique())
            # hu.save_image('tmp.png', hu.denormalize(images, mode='rgb'), mask=masks.numpy())
            loss = losses.compute_cross_entropy(images, logits, masks=masks[None].cuda())


        elif loss_name in 'point_level':
            logits = self.model_base(images)
            # point supervision
            loss = losses.compute_point_level(images, logits, point_list=batch['point_list'])
            
        elif loss_name in 'cons_point_loss':
            logits = self.model_base(images)
            # implementation needed
            loss = losses.compute_const_point_loss(self.model_base, images, logits, point_list=batch['point_list'])

        elif loss_name in 'rot_point_loss':
            logits = self.model_base(images)
            # implementation needed
            loss = losses.compute_rot_point_loss(self.model_base, images, logits, point_list=batch['point_list'])


        elif loss_name == 'joint_cross_entropy':
            logits = self.model_base(images)
            loss = ut.joint_loss(logits, batch["masks"].float().cuda())

        elif loss_name == 'lcfcn_crossentropy_crf_loss':
            cam, logits_aff = self.model_base(images, return_cam=True, crf=True)
            loss = lcfcn_loss.compute_loss(points=(batch['points']==1).long().cuda(), 
                                           probs=F.softmax(cam, dim=1)[:,1])
            loss += 0.5*losses.compute_cross_entropy(images, cam, masks=logits_aff.argmax(dim=1).cuda())

        elif loss_name == 'lcfcn_crossentropy_loss':
            cam, logits_aff = self.model_base(images, return_cam=True)
            loss = lcfcn_loss.compute_loss(points=(batch['points']==1).long().cuda(), 
                                           probs=F.softmax(cam, dim=1)[:,1])
            loss += 0.5*losses.compute_cross_entropy(images, logits_aff, masks=cam.argmax(dim=1).detach().cuda())

        elif loss_name == 'lcfcn_loss':
            logits = self.model_base(images)
            loss = lcfcn_loss.compute_loss(points=(batch['points']==1).long().cuda(), 
                                           probs=F.softmax(logits, dim=1)[:,1])

        elif loss_name == 'lcfcn_const_mean_loss':
            logits = self.model_base(images)
            loss = lcfcn_loss.compute_loss(points=(batch['points']==1).long().cuda(), 
                                           probs=F.softmax(logits, dim=1)[:,1])
            loss = loss + losses.compute_const_point_mean_loss(self.model_base, images, logits, point_list=None)

        elif loss_name == 'lcfcn_const_loss':
            logits = self.model_base(images)
            loss = lcfcn_loss.compute_loss(points=(batch['points']==1).long().cuda(), 
                                           probs=F.softmax(logits, dim=1)[:,1])
            loss = loss + losses.compute_const_point_loss(self.model_base, images, logits, point_list=None)

        else:
            raise ValueError('nope')
        if loss != 0:
            loss.backward()
            if self.exp_dict['model'].get('clip_grad'):
                ut.clip_gradient(self.opt, 0.5)
         
            self.opt.step()
            
        return {'train_loss': float(loss)}

    @torch.no_grad()
    def predict_on_batch(self, batch):
        self.eval()
        image = batch['images'].cuda()
    
        if self.n_classes == 1:
            res = self.model_base.forward(image)
            if 'shape' in batch['meta'][0]:
                res = F.upsample(res, size=batch['meta'][0]['shape'],              
                            mode='bilinear', align_corners=False)
            res = (res.sigmoid().data.cpu().numpy() > 0.5).astype('float')
        else:
            self.eval()
            logits = self.model_base.forward(image)
            res = logits.argmax(dim=1).data.cpu().numpy()

        return res 

    @torch.no_grad()
    def vis_on_batch(self, batch, savedir_image):
        self.eval()
        images = batch["images"].to(DEVICE)
        points = batch["points"].long().to(DEVICE)
        logits = self.model_base.forward(images)
        probs = logits.sigmoid().cpu().numpy()

        blobs = lcfcn_loss.get_blobs(probs=probs[0,1])

        pred_counts = (np.unique(blobs)!=0).sum()
        pred_blobs = blobs
        pred_probs = probs.squeeze()

        # loc 
        pred_count = pred_counts.ravel()[0]
        pred_blobs = pred_blobs.squeeze()
        pred_points = lcfcn_loss.blobs2points(pred_blobs).squeeze()
        img_org = hu.get_image(batch["images"],denorm="rgb")

        i1 = convert(np.array(img_org), batch['points'][0], enlarge=20)
        i2 = convert(np.array(img_org), pred_blobs, enlarge=0)
        i3 = convert(np.array(img_org), pred_points, enlarge=20)
        
        
        hu.save_image(savedir_image, np.hstack([i1, i2, i3]))

    def val_on_loader(self, loader, savedir_images=None, n_images=0):
        self.eval()
        val_meter = metrics.SegMeter(split=loader.dataset.split)
        
        i_count = 0
        for i, batch in enumerate(tqdm.tqdm(loader)):
            # make sure it wasn't trained on
            for m in batch['meta']:
                assert(m['hash'] not in self.train_hashes)

            val_meter.val_on_batch(self, batch)
            if i_count < n_images and len(batch['point_list'][0]):
                self.vis_on_batch(batch, savedir_image=os.path.join(savedir_images, 
                    '%d.png' % batch['meta'][0]['index']))
                i_count += 1

        
        return val_meter.get_avg_score()
        
    @torch.no_grad()
    def compute_uncertainty(self, images, replicate=False, scale_factor=None, n_mcmc=20, method='entropy'):
        self.eval()
        set_dropout_train(self)

        # put images to cuda
        images = images.cuda()
        _, _, H, W= images.shape

        if scale_factor is not None:
            images = F.interpolate(images, scale_factor=scale_factor)
        # variables
        input_shape = images.size()
        batch_size = input_shape[0]

        if replicate and False:
            # forward on n_mcmc batch      
            images_stacked = torch.stack([images] * n_mcmc)
            images_stacked = images_stacked.view(batch_size * n_mcmc, *input_shape[1:])
            logits = self.model_base(images_stacked)
            

        else:
            # for loop over n_mcmc
            logits = torch.stack([self.model_base(images) for _ in range(n_mcmc)])
            
            logits = logits.view(batch_size * n_mcmc, *logits.size()[2:])

        logits = logits.view([n_mcmc, batch_size, *logits.size()[1:]])
        _, _, n_classes, _, _ = logits.shape
        # binary do sigmoid 
        if n_classes == 1:
            probs = logits.sigmoid()
        else:
            probs = F.softmax(logits, dim=2)

        if scale_factor is not None:
            probs = F.interpolate(probs, size=(probs.shape[2], H, W))

        self.eval()

        if method == 'entropy':
            score_map = - xlogy(probs).mean(dim=0).sum(dim=1)

        if method == 'bald':
            left = - xlogy(probs.mean(dim=0)).sum(dim=1)
            right = - xlogy(probs).sum(dim=2).mean(0)
            bald = left - right
            score_map = bald


        return score_map 


def convert(img, mask, enlarge=0):
    if enlarge != 0:
        mask = expand_labels(mask, enlarge).astype('uint8')
    m = label2rgb(mask, bg_label=0)
    m = segmentation.mark_boundaries(m, mask.astype('uint8'))
    i = 0.5 * np.array(img) / 255.
    ind = m != 0
    i[ind] = m[ind] 

    return i

class TrainMonitor:
    def __init__(self):
        self.score_dict_sum = {}
        self.n = 0

    def add(self, score_dict):
        for k,v in score_dict.items():
            if k not in self.score_dict_sum:
                self.score_dict_sum[k] = score_dict[k]
            else:
                self.n += 1
                self.score_dict_sum[k] += score_dict[k]

    def get_avg_score(self):
        return {k:v/(self.n + 1) for k,v in self.score_dict_sum.items()}

def set_dropout_train(model):
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Dropout) or isinstance(module, torch.nn.Dropout2d):
            module.train()

def xlogy(x, y=None):
    z = torch.zeros(())
    if y is None:
        y = x
    assert y.min() >= 0
    return x * torch.where(x == 0., z.cuda(), torch.log(y))

