import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
from haven import haven_img as hi
import sys, os
from haven import haven_utils as hu
# path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
# sys.path.insert(0, path)
# from datasets import get_dataset
# from net import get_model
from src.misc import torchutils
from src.misc import imutils
import tqdm

def tensorize(x):
    return torch.stack([torch.from_numpy(b) 
        for b in x])

class CAM(torch.nn.Module):
    def __init__(self, base_model, max_step, lr=0.1, decay=1e-4, exp_dict=None):
        super().__init__()
        self.exp_dict = exp_dict
        self.model = base_model
        param_groups = self.model.trainable_parameters()
        # self.optimizer = torchutils.PolyOptimizer([
        #     {'params': param_groups[0], 'lr':lr, 'weight_decay': decay},
        #     {'params': param_groups[1], 'lr': 10*lr, 'weight_decay': decay},
        # ], lr=lr, weight_decay=decay, max_step=max_step)
        self.optimizer = torch.optim.Adam(self.parameters(), lr=1e-3, weight_decay=0.0005)
    def train_on_batch(self, batch):
        self.train()
        img = torch.stack(batch['img']).cuda(non_blocking=True)
        label = torch.stack(batch['label']).cuda(non_blocking=True)
        x = self.model(img)
        loss = F.multilabel_soft_margin_loss(x, label)
        return loss

    def train_on_loader(self, loader):
        self.model.train()
        loss_history = []
        n_batches = len(loader)
        pbar = tqdm.tqdm(total=n_batches)
        for step, pack in enumerate(loader):
            self.model.zero_grad()
            loss = self.train_on_batch(pack)
            loss_history.append(float(loss))
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            pbar.set_description("Train loss: %.4f" % np.mean(loss_history))
            pbar.update(1)

        pbar.close()

        return {"train_loss": np.mean(loss_history)}

    def val_on_loader(self, loader):
        self.model.eval()

        loss_history = []
        n_batches = len(loader)
        pbar = tqdm.tqdm(total=n_batches)
        with torch.no_grad():
            for pack in loader:
                img = torch.stack(pack['img']).cuda(non_blocking=True)
                label = torch.stack(pack['label']).cuda(non_blocking=True)

                x = self.model(img)
                loss1 = F.multilabel_soft_margin_loss(x, label)
                loss_history.append(float(loss1))
                pbar.set_description("Val loss: %.4f" % np.mean(loss_history))
                pbar.update(1)

        pbar.close()

        val_loss = np.mean(loss_history)
        return {"val_loss": val_loss, 'val_score': -val_loss} 
    
    @torch.no_grad()
    def vis_on_loader(self, loader, savedir_images, epoch=None):
        self.model.eval()
        count = 0 
        for batch in tqdm.tqdm(loader):
            if batch['label'][0][1:].sum() == 0:
                continue
            cam_dict = self.make_multiscale_cam(batch)
            keys = cam_dict['keys']
            high_res = cam_dict['high_res']
            im_rgb = batch['original'][0]
            pred_cam = hu.f2l(hi.gray2cmap(cam_dict['cam_crf']))
            hu.save_image(os.path.join(savedir_images, 
                                batch['meta'][0]['name']+'_cam_epoch:%d.jpg' % (epoch)), 
                                0.5*(np.array(im_rgb/255.)) + 0.5*pred_cam)

            for i, k in enumerate(keys):
                pred = hu.f2l(hi.gray2cmap(high_res[i]))            
                # im_rgb = hu.denormalize(im, 'rgb')
                # im_rgb = hu.f2l(im_rgb)
                hu.save_image(os.path.join(savedir_images, 
                                    batch['meta'][0]['name']+'_class:%d_epoch:%d.jpg' % (k, epoch)), 
                                    0.5*(np.array(im_rgb/255.)) + 0.5*pred)
                
            if count > 3:
                break
            count += 1
           

    def trainval(self, train_loader, val_loader):
        for ep in range(self.exp_dict["cam_num_epoches"]):
            metrics = {}
            metrics.update(self.train_on_loader(train_loader))
            metrics.update(self.val_on_loader(val_loader))
        torch.save(self.model.module.state_dict(), exp_dict["cam_weights_name"] + '.pth')
        torch.cuda.empty_cache()
    
    def get_cam(self, x):
        x = self.model.stage1(x)
        x = self.model.stage2(x)
        x = self.model.stage3(x)
        x = self.model.stage4(x)

        x = F.conv2d(x, self.model.classifier.weight)
        x = F.relu(x)

        x = x[0] + x[1].flip(-1)

        return x

    @torch.no_grad()
    def make_multiscale_cam(self, batch):
        self.eval()
        self.model.eval()
        label = batch['label'][0]
        size = batch['size'][0]

        strided_size = imutils.get_strided_size(size, 4)
        strided_up_size = imutils.get_strided_up_size(size, 16)

        outputs = [self.get_cam(img.cuda(non_blocking=True))
                for img in batch['img_msf']]

        strided_cam = torch.sum(torch.stack(
            [F.interpolate(torch.unsqueeze(o, 0), strided_size, mode='bilinear', align_corners=False)[0] for o
            in outputs]), 0)

        highres_cam = [F.interpolate(torch.unsqueeze(o, 1), strided_up_size,
                                    mode='bilinear', align_corners=False) for o in outputs]
        highres_cam = torch.sum(torch.stack(highres_cam, 0), 0)[:, 0, :size[0], :size[1]]

        valid_cat = torch.nonzero(label)[:, 0]

        strided_cam = strided_cam[valid_cat]
        strided_cam /= F.adaptive_max_pool2d(strided_cam, (1, 1)) + 1e-5

        highres_cam = highres_cam[valid_cat]
        highres_cam /= F.adaptive_max_pool2d(highres_cam, (1, 1)) + 1e-5

        # save cams
        cam_dict = {"keys": valid_cat, "cam": strided_cam.cpu(), "high_res": highres_cam.cpu().numpy()} 
        cam_dict['cam_crf'] = compute_crf(batch, cam_dict)
        return cam_dict

        
    def get_state_dict(self):
        ret = {}
        ret["optimizer"] = self.optimizer.state_dict()
        ret["model"] = self.model.state_dict()
        return ret

    def set_state_dict(self, state_dict):
        self.optimizer.load_state_dict(state_dict['optimizer'])
        self.model.load_state_dict(state_dict['model'])


def compute_crf(batch, cam_dict):
    img = batch['original'][0]

    cams = cam_dict['high_res']
    keys = np.pad(cam_dict['keys'] + 1, (1, 0), mode='constant')

    # 1. find confident fg & bg
    fg_conf_cam = np.pad(cams, ((1, 0), (0, 0), (0, 0)), mode='constant', 
                        constant_values=0.30)
    fg_conf_cam = np.argmax(fg_conf_cam, axis=0)
    pred = imutils.crf_inference_label(img, fg_conf_cam, n_labels=keys.shape[0])
    fg_conf = keys[pred]

    bg_conf_cam = np.pad(cams, ((1, 0), (0, 0), (0, 0)), mode='constant', 
                        constant_values=0.05)
    bg_conf_cam = np.argmax(bg_conf_cam, axis=0)
    pred = imutils.crf_inference_label(img, bg_conf_cam, n_labels=keys.shape[0])
    bg_conf = keys[pred]

    # 2. combine confident fg & bg
    conf = fg_conf.copy()
    conf[fg_conf == 0] = 255
    conf[bg_conf + fg_conf == 0] = 0

    return conf