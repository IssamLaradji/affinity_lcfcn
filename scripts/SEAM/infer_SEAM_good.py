
import numpy as np
import torch
import cv2
import os
import voc12.data
import scipy.misc
import importlib
from torch.utils.data import DataLoader
import torchvision
from tool import imutils, pyutils#, visualization
import argparse
from PIL import Image
import torch.nn.functional as F
import pandas as pd
import PIL.Image


def msf_img_lists(name, img, label, SEAM_model):
    name, img, label = name, img, label
    model = SEAM_model
    unit = 1
    scales = [0.5, 1.0, 1.5, 2.0]
    inter_transform = torchvision.transforms.Compose(
        [np.asarray,
         model.normalize,
         imutils.HWC_to_CHW])

    rounded_size = (int(round(img.size[0] / unit) * unit), int(round(img.size[1] / unit) * unit))

    ms_img_list = []
    for s in scales:
        target_size = (round(rounded_size[0] * s),
                       round(rounded_size[1] * s))
        s_img = img.resize(target_size, resample=PIL.Image.CUBIC)
        ms_img_list.append(s_img)

    if inter_transform:
        for i in range(len(ms_img_list)):
            ms_img_list[i] = inter_transform(ms_img_list[i])

    msf_img_list = []
    for i in range(len(ms_img_list)):
        msf_img_list.append(ms_img_list[i])
        msf_img_list.append(np.flip(ms_img_list[i], -1).copy())

    return name, msf_img_list, label

def infer_SEAM():
    # parser = argparse.ArgumentParser()
    # parser.add_argument("--weights", required=True, type=str)
    # parser.add_argument("--network", default="network.resnet38_SEAM", type=str)
    # parser.add_argument("--infer_list", default="voc12/train.txt", type=str)
    # parser.add_argument("--num_workers", default=8, type=int)
    # parser.add_argument("--voc12_root", default='VOC2012', type=str)
    # parser.add_argument("--out_cam", default=None, type=str)
    # parser.add_argument("--out_crf", default=None, type=str)
    # parser.add_argument("--out_cam_pred", default=None, type=str)
    # parser.add_argument("--out_cam_pred_alpha", default=0.26, type=float)

    """
    --weights "C:/Users/Alzay/NewPC_OneDrive/OneDrive - James Cook University/dev/ISPS/Alz_temp/SEAM/SEAM_model/resnet38_SEAM.pth" 
    --infer_list "D:/prototypes/_cam_dir/val.txt" 
    --out_cam "D:/prototypes/_cam_dir" 
    --voc12_root "D:/Datasets/Pascal_Voc_Dataset/pascal_2012/VOCdevkit/VOC2012/" 
    --out_crf "D:\prototypes\_crf_dir"
    """
    weights ="C:/Users/Alzay/NewPC_OneDrive/OneDrive - James Cook University/dev/ISPS/Alz_temp/SEAM/SEAM_model/resnet38_SEAM.pth"
    network ="network.resnet38_SEAM"
    num_workers =1
    voc12_root ="D:/Datasets/Pascal_Voc_Dataset/pascal_2012/VOCdevkit/VOC2012/"
    infer_list =os.path.join(voc12_root,"ImageSets", "Segmentation", "trainval.txt")
    # infer_list =[os.path.splitext(filename)[0] for filename in os.listdir(os.path.join(voc12_root,"JPEGImages"))]
    out_cam ="D:/prototypes/_cam_dir"
    out_crf ="D:\prototypes\_crf_dir"
    out_cam_pred ="D:/prototypes/out_cam_pred/"
    out_cam_pred_alpha =0.26

    # args = parser.parse_args()
    crf_alpha = [4,24]
    model = getattr(importlib.import_module(network), 'Net')()
    model.load_state_dict(torch.load(weights))

    model.eval()
    model.cuda()
        
    infer_dataset = voc12.data.VOC12ClsDatasetMSF(infer_list, voc12_root=voc12_root,
                                                  scales=[0.5, 1.0, 1.5, 2.0],
                                                  inter_transform=torchvision.transforms.Compose(
                                                       [np.asarray,
                                                        model.normalize,
                                                        imutils.HWC_to_CHW]))

    infer_data_loader = DataLoader(infer_dataset, shuffle=False, num_workers=num_workers, pin_memory=True)

    n_gpus = torch.cuda.device_count()
    model_replicas = torch.nn.parallel.replicate(model, list(range(n_gpus)))

    for iter, (img_name, img_list, label) in enumerate(infer_data_loader):
        img_name = img_name[0]; label = label[0]

        img_path = voc12.data.get_img_path(img_name, voc12_root)
        orig_img = np.asarray(Image.open(img_path))
        orig_img_size = orig_img.shape[:2]

        def _work(i, img):
            with torch.no_grad():
                with torch.cuda.device(i%n_gpus):
                    _, cam = model_replicas[i%n_gpus](img.cuda())
                    cam = F.upsample(cam[:,1:,:,:], orig_img_size, mode='bilinear', align_corners=False)[0]
                    cam = cam.cpu().numpy() * label.clone().view(20, 1, 1).numpy()
                    if i % 2 == 1:
                        cam = np.flip(cam, axis=-1)
                    return cam

        thread_pool = pyutils.BatchThreader(_work, list(enumerate(img_list)),
                                            batch_size=12, prefetch_size=0, processes=num_workers)

        cam_list = thread_pool.pop_results()

        sum_cam = np.sum(cam_list, axis=0)
        sum_cam[sum_cam < 0] = 0
        cam_max = np.max(sum_cam, (1,2), keepdims=True)
        cam_min = np.min(sum_cam, (1,2), keepdims=True)
        sum_cam[sum_cam < cam_min+1e-5] = 0
        norm_cam = (sum_cam-cam_min-1e-5) / (cam_max - cam_min + 1e-5)

        cam_dict = {}
        for i in range(20):
            if label[i] > 1e-5:
                cam_dict[i] = norm_cam[i]

        if out_cam is not None:
            np.save(os.path.join(out_cam, img_name + '.npy'), cam_dict)

        if out_cam_pred is not None:
            bg_score = [np.ones_like(norm_cam[0])*out_cam_pred_alpha]
            pred = np.argmax(np.concatenate((bg_score, norm_cam)), 0)
            scipy.misc.imsave(os.path.join(out_cam_pred, img_name + '.png'), pred.astype(np.uint8))

        def _crf_with_alpha(cam_dict, alpha):
            v = np.array(list(cam_dict.values()))
            bg_score = np.power(1 - np.max(v, axis=0, keepdims=True), alpha)
            bgcam_score = np.concatenate((bg_score, v), axis=0)
            crf_score = imutils.crf_inference(orig_img, bgcam_score, labels=bgcam_score.shape[0])

            n_crf_al = dict()

            n_crf_al[0] = crf_score[0]
            for i, key in enumerate(cam_dict.keys()):
                n_crf_al[key+1] = crf_score[i+1]

            return n_crf_al

        if out_crf is not None:
            for t in crf_alpha:
                crf = _crf_with_alpha(cam_dict, t)
                folder = out_crf + ('_%.1f'%t)
                if not os.path.exists(folder):
                    os.makedirs(folder)
                np.save(os.path.join(folder, img_name + '.npy'), crf)

        print(iter)


if __name__ == '__main__':

    infer_SEAM()