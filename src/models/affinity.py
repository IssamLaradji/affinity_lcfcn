import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
# import sys, os
# from ..misc import torchutils
from src.misc import pyutils, torchutils, indexing, imutils
import src.datasets.voc12
import src.datasets.voc12.dataloader
import skimage
import torch
# from src.misc import indexing
from src.datasets import BatchAff
import tqdm

class AFF(torch.nn.Module):
    def __init__(self, base_model, cam_model, max_step, lr=0.1, decay=1e-4, exp_dict=None):
        super().__init__()
        irn_crop_size = 512
        self.path_index = indexing.PathIndex(radius=10, 
                            default_size=(exp_dict["model"]["irn_crop_size"] // 4, 
                                          exp_dict["model"]["irn_crop_size"] // 4))
        self.extract_aff_lab_func = src.datasets.voc12.dataloader.GetAffinityLabelFromIndices(self.path_index.src_indices, 
                                                                self.path_index.dst_indices)
        self.batchaff = BatchAff()
        self.exp_dict = exp_dict

        self.model = base_model.cuda()
        self.cam_model = cam_model
        param_groups = self.model.trainable_parameters()

        self.optimizer = torchutils.PolyOptimizer([
            {'params': param_groups[0], 'lr': 1*lr, 
                            'weight_decay': decay},
            {'params': param_groups[1], 'lr': 10*lr, 
                            'weight_decay': decay}
        ], lr=lr, weight_decay=decay, 
                    max_step=max_step)

    def get_instance_segmentation(self, batch):
        img_name = batch['name'][0]
        size = torch.from_numpy(np.asarray(batch['size']))[0]

        edge, dp = self.model.get_edge_displacement(batch['img_msf'][0].cuda(non_blocking=True), 
                                                        self.exp_dict["model"]["irn_crop_size"],
                                                        stride=4)

        dp = dp.cpu().numpy()

        #cam_dict = np.load(args.cam_out_dir + '/' + img_name + '.npy', allow_pickle=True).item()
        cam_dict = batch["cam_dict"]
        cams = cam_dict['cam'].cuda()
        keys = cam_dict['keys']

        if self.exp_dict.get('centroid_mode') == 'points':
            centroids = get_points()
        else:
            centroids = find_centroids_with_refinement(dp)

        instance_map = cluster_centroids(centroids, dp)
        instance_cam = separte_score_by_mask(cams, instance_map)

        rw = indexing.propagate_to_edge(instance_cam, edge, beta=self.exp_dict["model"]["beta"], exp_times=self.exp_dict["model"]["exp_times"], radius=5)

        rw_up = F.interpolate(rw, scale_factor=4, mode='bilinear', align_corners=False)[:, 0, :size[0], :size[1]]
        rw_up = rw_up / torch.max(rw_up)

        rw_up_bg = F.pad(rw_up, (0, 0, 0, 0, 1, 0), value=self.exp_dict["model"]["ins_seg_bg_thres"])

        num_classes = len(keys)
        num_instances = instance_map.shape[0]

        instance_shape = torch.argmax(rw_up_bg, 0).cpu().numpy()
        instance_shape = pyutils.to_one_hot(instance_shape, maximum_val=num_instances*num_classes+1)[1:]
        instance_class_id = np.repeat(keys, num_instances)

        detected = detect_instance(rw_up.cpu().numpy(), instance_shape, instance_class_id,
                                    max_fragment_size=size[0] * size[1] * 0.01)

        return detected

    def train_on_batch(self, batch):
        pack = batch
        with torch.no_grad():
            bg_pos_label = [] 
            fg_pos_label = []
            neg_label = []
            rescaled_images = []

            for i in range(len(pack['original'])):
                batch = {k: [v[i]] for k, v in pack.items()} 
                cam_dict = self.cam_model.make_multiscale_cam(batch)
                batch["cam_dict"] = cam_dict
                detected = self.get_instance_segmentation(batch)
                rescaled_dict = self.batchaff.transform_img_aff(batch['original'][0], cam_dict["cam_crf"])
                rescaled_images.append(torch.from_numpy(rescaled_dict["img"]))
                _aff_bg_pos_label, _aff_fg_pos_label, _aff_neg_label = self.extract_aff_lab_func(rescaled_dict["reduced_mask"])
                
                bg_pos_label.append(_aff_bg_pos_label.cuda(non_blocking=True))
                fg_pos_label.append(_aff_fg_pos_label.cuda(non_blocking=True))
                neg_label.append(_aff_neg_label.cuda(non_blocking=True))

            bg_pos_label = torch.stack(bg_pos_label, 0)
            fg_pos_label = torch.stack(fg_pos_label, 0)
            neg_label = torch.stack(neg_label, 0)

        images = torch.stack(rescaled_images, 0).cuda(non_blocking=True)
        pos_aff_loss, neg_aff_loss, dp_fg_loss, dp_bg_loss = self.model(images, True)

        bg_pos_aff_loss = torch.sum(bg_pos_label * pos_aff_loss) / (torch.sum(bg_pos_label) + 1e-5)
        fg_pos_aff_loss = torch.sum(fg_pos_label * pos_aff_loss) / (torch.sum(fg_pos_label) + 1e-5)
        pos_aff_loss = bg_pos_aff_loss / 2 + fg_pos_aff_loss / 2
        neg_aff_loss = torch.sum(neg_label * neg_aff_loss) / (torch.sum(neg_label) + 1e-5)

        dp_fg_loss = torch.sum(dp_fg_loss * torch.unsqueeze(fg_pos_label, 1)) / (2 * torch.sum(fg_pos_label) + 1e-5)
        dp_bg_loss = torch.sum(dp_bg_loss * torch.unsqueeze(bg_pos_label, 1)) / (2 * torch.sum(bg_pos_label) + 1e-5)
        total_loss = (pos_aff_loss + neg_aff_loss) / 2 + (dp_fg_loss + dp_bg_loss) / 2

        return total_loss
        
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
            pbar.set_description("Training loss: %.4f" % np.mean(loss_history))
            pbar.update(1)

        pbar.close()
        return {"train_loss": np.mean(loss_history)}
        
    @torch.no_grad()
    def val_on_loader(self, loader):
        self.model.eval()

        loss_history = []
        n_batches = len(loader)
        pbar = tqdm.tqdm(total=n_batches)
        with torch.no_grad():

            for pack in loader:
                img = torch.stack(pack['img'])

                label = torch.stack(pack['label']).cuda(non_blocking=True)

                x = self.model(img)
                loss1 = F.multilabel_soft_margin_loss(x, label)

                loss_history.append(float(loss1))
                pbar.set_description("Val loss: %.4f" % np.mean(loss_history))
                pbar.update(1)

        pbar.close()
        return {"%s_loss": np.mean(loss_history)} 

    def trainval(self):
        for ep in range(self.exp_dict["cam_num_epoches"]):
            metrics = {}
            metrics.update(self.train_on_loader())
            metrics.update(self.val_on_loader())
        torch.save(self.model.module.state_dict(), exp_dict["cam_weights_name"] + '.pth')
        torch.cuda.empty_cache()

def find_centroids_with_refinement(displacement, iterations=300):
    # iteration: the number of refinement steps (u), set to any integer >= 100.

    height, width = displacement.shape[1:3]

    # 1. initialize centroids as their coordinates
    centroid_y = np.repeat(np.expand_dims(np.arange(height), 1), width, axis=1).astype(np.float32)
    centroid_x = np.repeat(np.expand_dims(np.arange(width), 0), height, axis=0).astype(np.float32)

    for i in range(iterations):

        # 2. find numbers after the decimals
        uy = np.ceil(centroid_y).astype(np.int32)
        dy = np.floor(centroid_y).astype(np.int32)
        y_c = centroid_y - dy

        ux = np.ceil(centroid_x).astype(np.int32)
        dx = np.floor(centroid_x).astype(np.int32)
        x_c = centroid_x - dx

        # 3. move centroids
        centroid_y += displacement[0][uy, ux] * y_c * x_c + \
                      displacement[0][dy, ux] *(1 - y_c) * x_c + \
                      displacement[0][uy, dx] * y_c * (1 - x_c) + \
                      displacement[0][dy, dx] * (1 - y_c) * (1 - x_c)

        centroid_x += displacement[1][uy, ux] * y_c * x_c + \
                      displacement[1][dy, ux] *(1 - y_c) * x_c + \
                      displacement[1][uy, dx] * y_c * (1 - x_c) + \
                      displacement[1][dy, dx] * (1 - y_c) * (1 - x_c)

        # 4. bound centroids
        centroid_y = np.clip(centroid_y, 0, height-1)
        centroid_x = np.clip(centroid_x, 0, width-1)

    centroid_y = np.round(centroid_y).astype(np.int32)
    centroid_x = np.round(centroid_x).astype(np.int32)

    return np.stack([centroid_y, centroid_x], axis=0)

def cluster_centroids(centroids, displacement, thres=2.5):
    # thres: threshold for grouping centroid (see supp)

    dp_strength = np.sqrt(displacement[1] ** 2 + displacement[0] ** 2)
    height, width = dp_strength.shape

    weak_dp_region = dp_strength < thres

    dp_label = skimage.measure.label(weak_dp_region, connectivity=1, background=0)
    dp_label_1d = dp_label.reshape(-1)

    centroids_1d = centroids[0]*width + centroids[1]

    clusters_1d = dp_label_1d[centroids_1d]

    cluster_map = imutils.compress_range(clusters_1d.reshape(height, width) + 1)

    return pyutils.to_one_hot(cluster_map)

def separte_score_by_mask(scores, masks):
    instacne_map_expanded = torch.from_numpy(np.expand_dims(masks, 0).astype(np.float32))
    instance_score = torch.unsqueeze(scores, 1) * instacne_map_expanded.cuda()
    return instance_score

def detect_instance(score_map, mask, class_id, max_fragment_size=0):
    # converting pixel-wise instance ids into detection form

    pred_score = []
    pred_label = []
    pred_mask = []

    for ag_score, ag_mask, ag_class in zip(score_map, mask, class_id):
        if np.sum(ag_mask) < 1:
            continue
        segments = pyutils.to_one_hot(skimage.measure.label(ag_mask, connectivity=1, background=0))[1:]
        # connected components analysis

        for seg_mask in segments:
            if np.sum(seg_mask) < max_fragment_size:
                pred_score.append(0)
            else:
                pred_score.append(np.max(ag_score * seg_mask))
            pred_label.append(ag_class)
            pred_mask.append(seg_mask)

    return {'score': np.stack(pred_score, 0),
           'mask': np.stack(pred_mask, 0),
           'class': np.stack(pred_label, 0)}



if __name__ == '__main__':
    datadir_base = ''
    exp_dict = {'model':'game'}
    aff = AFF(datadir_base, exp_dict, max_step=1)
    batch = torch.utils.data.dataloader.default_collate(
                        [aff.train_dataset[i] for i in range(8)])
    loss = aff.train_on_batch(batch)
    print('test aff')
