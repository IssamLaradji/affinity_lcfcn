import torch
import torch.nn as nn
import torch.sparse as sparse
import torch.nn.functional as F
import numpy as np
from . import resnet38d
from ..tool import pyutils


def crf_inference(img, probs, t=10, scale_factor=1, labels=2):
    import pydensecrf.densecrf as dcrf
    from pydensecrf.utils import unary_from_softmax

    h, w = img.shape[:2]
    n_labels = labels

    d = dcrf.DenseCRF2D(w, h, n_labels)

    unary = unary_from_softmax(probs)
    unary = np.ascontiguousarray(unary)

    d.setUnaryEnergy(unary)
    d.addPairwiseGaussian(sxy=3/scale_factor, compat=3)
    d.addPairwiseBilateral(sxy=80/scale_factor, srgb=13, rgbim=np.copy(img), compat=10)
    Q = d.inference(t)

    return np.array(Q).reshape((n_labels, h, w))

class Net(resnet38d.Net):
    def __init__(self, n_classes, exp_dict):
        super(Net, self).__init__()

        self.f8_3 = torch.nn.Conv2d(512, 64, 1, bias=False)
        self.f8_4 = torch.nn.Conv2d(1024, 128, 1, bias=False)
        self.f8_5 = torch.nn.Conv2d(4096, 256, 1, bias=False)

        self.f9 = torch.nn.Conv2d(448, 448, 1, bias=False)
        
        torch.nn.init.kaiming_normal_(self.f8_3.weight)
        torch.nn.init.kaiming_normal_(self.f8_4.weight)
        torch.nn.init.kaiming_normal_(self.f8_5.weight)
        torch.nn.init.xavier_uniform_(self.f9.weight, gain=4)

        self.not_training = [self.conv1a, self.b2, self.b2_1, self.b2_2]

        self.from_scratch_layers = [self.f8_3, self.f8_4, self.f8_5, self.f9]

        self.predefined_featuresize = int(448//8)
        self.radius = 5
        self.ind_from, self.ind_to = pyutils.get_indices_of_pairs(radius=self.radius, size=(self.predefined_featuresize, self.predefined_featuresize))
        self.ind_from = torch.from_numpy(self.ind_from); self.ind_to = torch.from_numpy(self.ind_to)

        self.dropout7 = torch.nn.Dropout2d(0.5)

        self.fc8 = nn.Conv2d(4096, n_classes, 1, bias=False)
        self.beta = exp_dict['model'].get('beta', 8)
        self.logt = exp_dict['model'].get('logt', 4)
        return

    def apply_affinity(self, img, logits, crf=False):
        h_org, w_org = img.shape[2:]
        padded_size = (int(np.ceil(img.shape[2]/8)*8), int(np.ceil(img.shape[3]/8)*8))
        p2d = (0, padded_size[1] - img.shape[3], 0, padded_size[0] - img.shape[2])
        img = F.pad(img, p2d)
        beta = self.beta
        logt = self.logt
        aff_mat = torch.pow(self.forward(img.cuda(), True), beta)
        trans_mat = aff_mat / torch.sum(aff_mat, dim=0, keepdim=True)

        for _ in range(logt):
            trans_mat = torch.matmul(trans_mat, trans_mat)

        n_classes = logits.shape[1]
        logits = F.pad(logits, p2d)
        _,_, h, w = logits.shape
        # indices_from_to = get_indices_in_radius(h, w, radius=5)
        # labels = get_affinity_labels(logits.argmax(dim=1).cpu().numpy(), indices_from_to[:,0], indices_from_to[:,1])
        cam = F.avg_pool2d(logits, 8, 8)
        cam_vec = cam.view(n_classes, -1)
        cam_vec = torch.matmul(cam_vec.cuda(), trans_mat)
        logits_new = cam_vec.view(1, n_classes, h//8, w//8)
        
        logits_new = torch.nn.functional.interpolate(logits_new, (h, w), mode='bilinear') 
        if crf: 
            img_8 = img[0].cpu().numpy().transpose((1,2,0))
            img_8 = np.ascontiguousarray(img_8)
            mean = (0.485, 0.456, 0.406)
            std = (0.229, 0.224, 0.225)
            img_8[:,:,0] = (img_8[:,:,0]*std[0] + mean[0])*255
            img_8[:,:,1] = (img_8[:,:,1]*std[1] + mean[1])*255
            img_8[:,:,2] = (img_8[:,:,2]*std[2] + mean[2])*255
            img_8[img_8 > 255] = 255
            img_8[img_8 < 0] = 0
            img_8 = img_8.astype(np.uint8)
            cam_rw = logits_new[0].detach().cpu().numpy()
            cam_rw = crf_inference(img_8, cam_rw, t=1)
            cam_rw = torch.from_numpy(cam_rw).view(1, 2, img.shape[2], img.shape[3]).cuda()
            logits_new = cam_rw

        logits_new = torch.nn.functional.interpolate(logits_new, (h_org, w_org), mode='bilinear') 
        return logits_new
    
    def output_logits(self, x, to_dense=False):

        d = super().forward_as_dict(x)

        cam = self.fc8(self.dropout7(d['conv6']))
        h, w = x.shape[-2:]
        return torch.nn.functional.interpolate(cam, (h, w), mode='bilinear', align_corners=True) 
        

    def forward(self, x, to_dense=False):

        d = super().forward_as_dict(x)

        f8_3 = F.elu(self.f8_3(d['conv4']))
        f8_4 = F.elu(self.f8_4(d['conv5']))
        f8_5 = F.elu(self.f8_5(d['conv6']))
        x = F.elu(self.f9(torch.cat([f8_3, f8_4, f8_5], dim=1)))

        if x.size(2) == self.predefined_featuresize and x.size(3) == self.predefined_featuresize:
            ind_from = self.ind_from
            ind_to = self.ind_to
        else:
            min_edge = min(x.size(2), x.size(3))
            radius = (min_edge-1)//2 if min_edge < self.radius*2+1 else self.radius
            ind_from, ind_to = pyutils.get_indices_of_pairs(radius, (x.size(2), x.size(3)))
            ind_from = torch.from_numpy(ind_from); ind_to = torch.from_numpy(ind_to)

        x = x.view(x.size(0), x.size(1), -1).contiguous()
        ind_from = ind_from.contiguous()
        ind_to = ind_to.contiguous()

        ff = torch.index_select(x, dim=2, index=ind_from.cuda(non_blocking=True))
        ft = torch.index_select(x, dim=2, index=ind_to.cuda(non_blocking=True))

        ff = torch.unsqueeze(ff, dim=2)
        ft = ft.view(ft.size(0), ft.size(1), -1, ff.size(3))

        aff = torch.exp(-torch.mean(torch.abs(ft-ff), dim=1))

        if to_dense:
            aff = aff.view(-1).cpu()

            ind_from_exp = torch.unsqueeze(ind_from, dim=0).expand(ft.size(2), -1).contiguous().view(-1)
            indices = torch.stack([ind_from_exp, ind_to])
            indices_tp = torch.stack([ind_to, ind_from_exp])

            area = x.size(2)
            indices_id = torch.stack([torch.arange(0, area).long(), torch.arange(0, area).long()])

            aff_mat = sparse.FloatTensor(torch.cat([indices, indices_id, indices_tp], dim=1),
                                      torch.cat([aff, torch.ones([area]), aff])).to_dense().cuda()

            return aff_mat

        else:
            return aff


    def get_parameter_groups(self):
        groups = ([], [], [], [])

        for m in self.modules():

            if (isinstance(m, nn.Conv2d) or isinstance(m, nn.modules.normalization.GroupNorm)):

                if m.weight.requires_grad:
                    if m in self.from_scratch_layers:
                        groups[2].append(m.weight)
                    else:
                        groups[0].append(m.weight)

                if m.bias is not None and m.bias.requires_grad:

                    if m in self.from_scratch_layers:
                        groups[3].append(m.bias)
                    else:
                        groups[1].append(m.bias)

        return groups



