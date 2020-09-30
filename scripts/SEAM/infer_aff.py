import torch
import torchvision
from .tool import imutils

import argparse
import importlib
import numpy as np

from .voc12 import data
from torch.utils.data import DataLoader
import scipy.misc
import torch.nn.functional as F
import os.path
from .network import resnet38_aff
from torchvision.transforms import ToPILImage, ToTensor


def get_indices_in_radius(height, width, radius):

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

def HCW_to_CHW(tensor, sal=False):
    if sal:
        tensor = np.expand_dims(tensor, axis=0)
    else:
        tensor = np.transpose(tensor, (1, 2, 0))
    return tensor


def name_img(name, img, SEAM_model):
    name = name


    # label = ToTensor()(label)
    # img = ToPILImage()(img)
    model = SEAM_model
    unit = 1
    scales = [0.5, 1.0, 1.5, 2.0]
    inter_transform = torchvision.transforms.Compose(
        [np.asarray,
         model.normalize,
         # ToTensor(),
         imutils.HWC_to_CHW
         ])
    intera_transform = torchvision.transforms.Compose(
        [ToTensor(),
         HCW_to_CHW
         ])

    img = inter_transform(img)
    img = intera_transform(img)
    img = img[None]

    return name, img

def infer_aff(name, img, cam_dict, weights_dir = "", model=None):

    weights =weights_dir
    # network ="network.resnet38_aff"
    alpha = 6
    beta = 8
    logt = 6
    crf = False

    if model is None:
        model = resnet38_aff.Net()
        model.load_state_dict(torch.load(weights), strict=False)

    model.eval()
    model.cuda()

    # infer_dataset = voc12.data.VOC12ImageDataset(infer_list, voc12_root=voc12_root,
    #                                            transform=torchvision.transforms.Compose(
    #     [np.asarray,
    #      model.normalize,
    #      imutils.HWC_to_CHW]))
    # infer_data_loader = DataLoader(infer_dataset, shuffle=False, num_workers=num_workers, pin_memory=True)
    name, img = name_img(name, img, model)

    # for iter, (name, img) in enumerate(infer_data_loader):

    # name = name[0]
    # print(iter)

    orig_shape = img.shape
    padded_size = (int(np.ceil(img.shape[2]/8)*8), int(np.ceil(img.shape[3]/8)*8))

    p2d = (0, padded_size[1] - img.shape[3], 0, padded_size[0] - img.shape[2])
    img = F.pad(img, p2d)

    dheight = int(np.ceil(img.shape[2]/8))
    dwidth = int(np.ceil(img.shape[3]/8))

    # cam = np.load(os.path.join(cam_dir, name + '.npy'), allow_pickle=True).item()
    cam = cam_dict

    cam_full_arr = np.zeros((21, orig_shape[2], orig_shape[3]), np.float32)
    for k, v in cam.items():
        cam_full_arr[k+1] = v
    cam_full_arr[0] = (1 - np.max(cam_full_arr[1:], (0), keepdims=False))**alpha
    #cam_full_arr[0] = 0.2
    cam_full_arr = np.pad(cam_full_arr, ((0, 0), (0, p2d[3]), (0, p2d[1])), mode='constant')

    with torch.no_grad():
        aff_mat = torch.pow(model.forward(img.cuda(), True), beta)

        trans_mat = aff_mat / torch.sum(aff_mat, dim=0, keepdim=True)
        for _ in range(logt):
            trans_mat = torch.matmul(trans_mat, trans_mat)

        cam_full_arr = torch.from_numpy(cam_full_arr)
        cam_full_arr = F.avg_pool2d(cam_full_arr, 8, 8)

        cam_vec = cam_full_arr.view(21, -1)

        cam_rw = torch.matmul(cam_vec.cuda(), trans_mat)
        cam_rw = cam_rw.view(1, 21, dheight, dwidth)

        cam_rw = torch.nn.Upsample((img.shape[2], img.shape[3]), mode='bilinear')(cam_rw)

        if crf:
            img_8 = img[0].numpy().transpose((1,2,0))#F.interpolate(img, (dheight,dwidth), mode='bilinear')[0].numpy().transpose((1,2,0))
            img_8 = np.ascontiguousarray(img_8)
            mean = (0.485, 0.456, 0.406)
            std = (0.229, 0.224, 0.225)
            img_8[:,:,0] = (img_8[:,:,0]*std[0] + mean[0])*255
            img_8[:,:,1] = (img_8[:,:,1]*std[1] + mean[1])*255
            img_8[:,:,2] = (img_8[:,:,2]*std[2] + mean[2])*255
            img_8[img_8 > 255] = 255
            img_8[img_8 < 0] = 0
            img_8 = img_8.astype(np.uint8)
            cam_rw = cam_rw[0].cpu().numpy()
            cam_rw = imutils.crf_inference(img_8, cam_rw, t=1)
            cam_rw = torch.from_numpy(cam_rw).view(1, 21, img.shape[2], img.shape[3]).cuda()


        _, cam_rw_pred = torch.max(cam_rw, 1)

        preds = np.uint8(cam_rw_pred.cpu().data[0])[:orig_shape[2], :orig_shape[3]]
        probs = cam_rw.cpu().data[0][:, :orig_shape[2], :orig_shape[3]]
        # scipy.misc.imsave(os.path.join(out_rw, name + '.png'), res)
        # print("saved : %s" %os.path.join(out_rw, name + '.png'))
        assert probs.shape[1] == preds.shape[0]
        assert probs.shape[2] == preds.shape[1]
        print("Done infer_aff")
        return preds, probs
