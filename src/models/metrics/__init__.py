from collections import defaultdict

from scipy import spatial
import numpy as np
import torch

from collections import defaultdict
from src.modules.lcfcn import lcfcn_loss
from scipy import spatial
import numpy as np
import torch

from collections import defaultdict

from scipy import spatial
import numpy as np
import torch


class SegMeter:
    def __init__(self, split):
        self.cf = None
        self.n_samples = 0
        self.split = split
        self.ae = 0
        self.game = 0

    def val_on_batch(self, model, batch):
        masks = batch["masks"].squeeze()
        self.n_samples += batch['images'].shape[0]
        pred_mask = model.predict_on_batch(batch).squeeze()

        # counts
        blobs = lcfcn_loss.get_blobs(pred_mask)
        points = lcfcn_loss.blobs2points(blobs)
        pred_counts = float(points.sum())
        self.ae += np.abs(float((batch['points']==1).sum()) - pred_counts)

        gt_points = batch['points'].squeeze().clone()
        gt_points[gt_points!=1] = 0
        self.game += lcfcn_loss.compute_game(pred_points=points.squeeze(), 
                                             gt_points=gt_points, L=3)
        # 
        # print(masks.sum())
        ind = masks != 255
        masks = masks[ind]
        pred_mask = pred_mask[ind]

        

        labels = np.arange(model.n_classes)
        cf = confusion_multi_class(torch.as_tensor(pred_mask).float().cuda(), masks.cuda().float(),
                                    labels=labels)
                                   
                                
        if self.cf is None:
            self.cf = cf 
        else:
            self.cf += cf

    def get_avg_score(self):
        # return -1 
        Inter = np.diag(self.cf)
        G = self.cf.sum(axis=1)
        P = self.cf.sum(axis=0)
        union = G + P - Inter

        nz = union != 0
        iou = Inter / np.maximum(union, 1)
        mIoU = np.mean(iou[nz])
        iou[~nz] = np.nan
        val_dict = {'%s_score' % self.split: mIoU}
        for c in range(self.cf.shape[1]):
            val_dict['%s_class%d' % (self.split, c)] = iou[c]
        val_dict['%s_mae' % (self.split)] = self.ae / self.n_samples
        val_dict['%s_game' % (self.split)] = self.game / self.n_samples
        return val_dict



def confusion_multi_class(prediction, truth, labels):
    """
    cf = confusion_matrix(y_true=prediction.cpu().numpy().ravel(),
            y_pred=truth.cpu().numpy().ravel(),
                    labels=labels)
    """
    nclasses = labels.max() + 1
    cf2 = torch.zeros(nclasses, nclasses, dtype=torch.float, device=prediction.device)
    prediction = prediction.view(-1).long()
    truth = truth.view(-1)
    to_one_hot = torch.eye(int(nclasses), dtype=cf2.dtype, device=prediction.device)
    for c in range(nclasses):
        true_mask = (truth == c)
        pred_one_hot = to_one_hot[prediction[true_mask]].sum(0)
        cf2[:, c] = pred_one_hot

    return cf2.cpu().numpy()


def confusion_binary_class(prediction, truth):
    confusion_vector = prediction / truth

    tp = torch.sum(confusion_vector == 1).item()
    fp = torch.sum(confusion_vector == float('inf')).item()
    tn = torch.sum(torch.isnan(confusion_vector)).item()
    fn = torch.sum(confusion_vector == 0).item()
    cm = np.array([[tn,fp],
                   [fn,tp]])
    return cm



class SegMeterBinary:
    def __init__(self, split):
        self.cf = None
        self.struct_list = []
        self.split = split

    def val_on_batch(self, model, batch):
        masks_org = batch["masks"]

        pred_mask_org = model.predict_on_batch(batch)
        ind = masks_org != 255
        masks = masks_org[ind]
        pred_mask = pred_mask_org[ind]
        self.n_classes = model.n_classes
        if model.n_classes == 1:
            cf = confusion_binary_class(torch.as_tensor(pred_mask).float().cuda(), masks.cuda().float())
        else:
            labels = np.arange(model.n_classes)
            cf = confusion_multi_class(torch.as_tensor(pred_mask).float().cuda(), masks.cuda().float(),
                                    labels=labels)

        if self.cf is None:
            self.cf = cf
        else:
            self.cf += cf

        # structure
        struct_score = float(struct_metric.compute_struct_metric(pred_mask_org, masks_org))
        self.struct_list += [struct_score]

    def get_avg_score(self):
        TP = np.diag(self.cf)
        TP_FP = self.cf.sum(axis=1)
        TP_FN = self.cf.sum(axis=0)
        TN = TP[::-1]
        

        FP = TP_FP - TP
        FN = TP_FN - TP

        iou = TP / (TP + FP + FN)
        dice = 2*TP / (FP + FN + 2*TP)

        iou[np.isnan(iou)] = -1
        dice[np.isnan(dice)] = -1

        mDice = np.mean(dice)
        mIoU = np.mean(iou)

        prec = TP / (TP + FP)
        recall = TP / (TP + FN)
        spec = TN/(TN+FP)
        fscore = (( 2.0 * prec * recall ) / (prec + recall))

        val_dict = {}
        if self.n_classes == 1:
            val_dict['%s_dice' % self.split] = dice[0]
            val_dict['%s_iou' % self.split] = iou[0]

            val_dict['%s_prec' % self.split] = prec[0]
            val_dict['%s_recall' % self.split] = recall[0]
            val_dict['%s_spec' % self.split] = spec[0]
            val_dict['%s_fscore' % self.split] = fscore[0]

            val_dict['%s_score' % self.split] = dice[0]
            val_dict['%s_struct' % self.split] = np.mean(self.struct_list)
        return val_dict

# def confusion_multi_class(prediction, truth, labels):
#     """
#     cf = confusion_matrix(y_true=prediction.cpu().numpy().ravel(),
#             y_pred=truth.cpu().numpy().ravel(),
#                     labels=labels)
#     """
#     nclasses = labels.max() + 1
#     cf2 = torch.zeros(nclasses, nclasses, dtype=torch.float,
#                       device=prediction.device)
#     prediction = prediction.view(-1).long()
#     truth = truth.view(-1)
#     to_one_hot = torch.eye(int(nclasses), dtype=cf2.dtype,
#                            device=prediction.device)
#     for c in range(nclasses):
#         true_mask = (truth == c)
#         pred_one_hot = to_one_hot[prediction[true_mask]].sum(0)
#         cf2[:, c] = pred_one_hot

#     return cf2.cpu().numpy()



def confusion_binary_class(pred_mask, gt_mask):
    intersect = pred_mask.bool() & gt_mask.bool()

    fp_tp = (pred_mask ==1).sum().item()
    fn_tp = gt_mask.sum().item()
    tn_fn = (pred_mask ==0).sum().item()

    tp = (intersect == 1).sum().item()
    fp = fp_tp - tp
    fn = fn_tp - tp
    tn = tn_fn - fn 

    cm = np.array([[tp, fp],
                   [fn, tn]])
    return cm