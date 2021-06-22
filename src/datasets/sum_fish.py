
import glob, torch
from haven import haven_utils as hu
import numpy as np
import os
from PIL import Image
import torch.utils.data as data
import torchvision.transforms as transforms
import pandas as pd
from src.modules.lcfcn import lcfcn_loss
from src import datasets
import copy
import fnmatch
import itertools as it

def filter_names(img_names, labels, counts, points_names, img_names_test):
    i = []
    l = []
    c = []
    p = []
    for k, im in enumerate(img_names):
        if im in img_names_test:
            continue
        i += [img_names[k]]
        l += [labels[k]]
        c += [counts[k]]
        p += [points_names[k]]

    return i, np.array(l), np.array(c), p 


class SumFish(data.Dataset):
    def __init__(self, split, datadir, exp_dict, habitat=None):

        self.count_mode = False
        if split == 'train':
            self.count_mode = exp_dict['model'].get('count_mode')
            # self.img_names, self.labels, self.mask_names = get_seg_data(datadir, split, habitat=habitat)
            self.img_names, self.labels, self.mask_names = get_sum_data(datadir, "train_val")
            if self.count_mode:
                self.path = os.path.join(datadir, 'Localization')
                self.img_names, self.labels, self.counts, self.points_names = get_loc_data(os.path.join(datadir, 'Localization'), split, habitat=habitat)

                img_names_test, _, _ = get_seg_data(datadir, split='val', habitat=habitat)
                self.img_names, self.labels, self.counts, self.points_names  = filter_names(self.img_names, self.labels, self.counts, self.points_names, img_names_test)
                
                img_names_test, _, _ = get_seg_data(datadir, split='test', habitat=habitat)
                self.img_names, self.labels, self.counts, self.points_names  = filter_names(self.img_names, self.labels, self.counts, self.points_names, img_names_test)

            # self.img_names_other, self.labels_other= get_clf_data(datadir, split, habitat=habitat)
            
            n_full_images = exp_dict['dataset'].get('n_full_images') 
            if n_full_images is not None :
                self.full_images = np.where(self.labels == 1)[0][: n_full_images]
            else:
                self.full_images  = []

            n_fish_images = exp_dict['dataset'].get('n_fish_images') 
            if n_fish_images is not None :
                ind_bg = np.where(self.labels == 0)[0]
                ind_fg = np.where(self.labels == 1)[0][: n_fish_images]
                ind_all = np.hstack([ind_bg, ind_fg])
                self.img_names = self.img_names[ind_all]
                self.labels = self.labels[ind_all] 
                self.mask_names = self.mask_names[ind_all]

        elif split == 'val':
            # self.img_names, self.labels, self.mask_names = get_seg_data(datadir, split, habitat=habitat)
            self.img_names, self.labels, self.mask_names = get_sum_data(datadir, "TEST")
            # self.img_names_other, self.labels_other= get_clf_data(datadir, split, habitat=habitat)

        elif split == 'test':
            self.img_names, self.labels, self.mask_names = get_sum_data(datadir, "TEST")
            # self.img_names_other, self.labels_other= get_clf_data(datadir, split, habitat=habitat)

        # self.size = 256
        self.size = (256,256)
        # self.images = [x for x in self.img_names if x not in bug_images_list]
        self.images =  self.img_names
        self.gts = self.mask_names
        # self.images_other = self.img_names_other
        # self.gts_other = self.labels_other
        self.split = split
        self.datadir = datadir
        self.dataset_size = len(self.images)

        self.habitats = ["habitats"]
        # for img_name in self.images:
        #     habitat = img_name.split('_')[0].split('/')[1]
        #     self.habitats += [habitat]

        self.habitats = np.array(self.habitats)
        self.unique_habitats = np.unique(self.habitats)
        self.unique_habitats.sort()
        self.unique_habitats = {h:i for i, h in enumerate(self.unique_habitats)}
        self.img_transform = transforms.Compose([
            transforms.Resize(self.size, interpolation=Image.BILINEAR),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

       
        self.gt_transform = transforms.Compose([
            transforms.Resize(self.size, interpolation=Image.NEAREST),
            # transforms.rotate(-90),
            # transforms.ToTensor()
            ])

        with_fish = self.labels.sum()
        print(split, ':', len(self), ', with_fish:', with_fish, ', without_fish:', len(self.labels) - with_fish)
        
        
    def __getitem__loc(self, index):
        name = self.img_names[index]
        image_pil = Image.open(self.path + "/images/"+ name + ".jpg")
        W, H = image_pil.size
        original = copy.deepcopy(image_pil)
        image = self.img_transform(image_pil)
        h, w = image.shape[-2:]
        # get points
        points = Image.open(self.path + "/masks/"+ name + ".png")#[..., np.newaxis]
        # points = self.gt_transform(points)
        points = np.array(points).clip(0,1)
        points = points.squeeze()

        points_new = np.zeros((h,w))
        y_list, x_list = np.where(points)
        point_list = []
        for y,x in zip(y_list, x_list):
            y_new, x_new = int(y * (h/H)), int(x * (w/W))
            points_new[y_new, x_new] = 1
            point_list += [{'y': y, 'x':x, 'cls': 1}]

        points = points_new

        counts = torch.LongTensor(np.array([int(points.sum())]))
        if (points == -1).all():
            pass
        else:
            assert int(np.count_nonzero(points)) == counts[0]
        assert counts.item() == self.counts[index]


        batch = {"images": image,
                 'original': original,
                 'masks': torch.as_tensor(points).long(),
                 "labels": float(self.labels[index] > 0),
                 "counts": float(self.counts[index]),
                 'size': image_pil.size,
                 'point_list':point_list,

                "points": torch.FloatTensor(np.array(points)),
                 "meta": {"index": index,
                          'name': self.images[index],
                   'hash': hu.hash_dict({'id': self.images[index]}),
                          "image_id": index,
                          'habitat': self.habitats[index],
                          'size': self.size,
                          "split": self.split}}

        return batch

    def __getitem__(self, index):
        if self.count_mode:
            return self.__getitem__loc(index)
        # Segmentation
        if self.split == 'train':
            image = rgb_loader(os.path.join(self.datadir, "train_val/images", self.images[index]))
            gt = rgb_loader(os.path.join(self.datadir, "train_val/masks", self.gts[index]))
        elif self.split == 'val' or 'test':
            image = rgb_loader(os.path.join(self.datadir, "TEST/images", self.images[index]))
            gt = rgb_loader(os.path.join(self.datadir, "TEST/masks", self.gts[index]))

        original = copy.deepcopy(image)

        image = self.img_transform(image)
        gt = self.gt_transform(gt)
        # gt = self.img_transform(gt)
        gt_name = self.gts[index]
        gt = np.array(gt)
        # gt = gt[:,:,0]
        # gt[gt==255] = 1

        # image,gt =  processSUIMDataRFHW(image, gt, sal=False)
        image,gt =  processSUIMDfish(image, gt)

        if np.sum(gt):
            labels = 1
        else:
            labels = 0

        img_size = (gt.shape[0], gt.shape[1])

        # Classification
        # image_other = rgb_loader(os.path.join(self.datadir, "Classification", self.images_other[index] + '.jpg'))
        # image_other = self.img_transform(image_other)
        
        points = lcfcn_loss.get_points_from_mask(gt, bg_points=-1)
        # hu.save_image('tmp.png', hu.denormalize(image, 'rgb'), points=points, radius=2)
        # hu.save_image('tmp.png', hu.denormalize(image, 'rgb'), mask=gt.numpy(), radius=2)
        uniques = np.unique(points)
        point_list = []
        for u in uniques:
            if u == 255:
                continue
            y_list, x_list = np.where(points == u)
            for y, x in zip(y_list, x_list):
                point_list += [{'y': y, 'x':x, 'cls': int(u)}]

        batch = {'images': image,
                #  'image_other': image_other,
                 'original':original,
                 'masks': torch.as_tensor(gt).long(),
                 'points': torch.LongTensor(points),
                 'label' : torch.from_numpy(np.ndarray([labels])),
                #  "labels_other": float(self.labels_other[index] > 0),
                 'size': img_size,

                 'point_list':point_list,

                 'meta': {'name': self.images[index],
                          'hash': hu.hash_dict({'id': self.images[index]}),
                          # 'hash':self.images[index],
                          'habitat':"habitats",
                          # 'habitat':self.habitats[index],
                          'shape': gt.squeeze().shape,
                          'index': index,
                          'split': self.split,
                          'size': self.size}}

        return batch

    def __len__(self):
        return self.dataset_size


def rgb_loader(path):
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')


def binary_loader(path):
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('L')

def sum_loader(path):
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img


# for seg,
def get_seg_data(path_base, split,  habitat=None ):
    df = pd.read_csv(os.path.join(path_base,  'Segmentation/%s.csv' % split))
    df = slice_df_reg(df, habitat)
    img_names = np.array(df['ID'])
    mask_names = np.array(df['ID'])
    labels = np.array(df['labels'])
    return img_names, labels, mask_names

# for sum,
def get_sum_data(path_base, splits):
    img = os.listdir(os.path.join(path_base,  '%s/images/' % splits ))
    msk = os.listdir(os.path.join(path_base,  '%s/masks/' % splits ))
    img = [x for x in img if x.endswith(".jpg")]
    msk = [x for x in msk if x.endswith(".bmp")]
    img_names = np.array(img)
    mask_names = np.array(msk)
    labels = np.array([1 for x in img])
    return img_names, labels, mask_names



def slice_df_reg(df, habitat):
    if habitat is None:
        return df
    return df[df['ID'].apply(lambda x: True if x.split("/")[1].split("_")[0]
                        == habitat else False)]

def get_loc_data(datadir, split,  habitat=None ):
    df = pd.read_csv(os.path.join(datadir,  '%s.csv' % split))
    df = slice_df_reg(df, habitat)
    img_names = np.array(df['ID'])
    points_names = np.array(df['ID'])
    counts = np.array(df['counts'])
    labels = np.array(df['labels'])
    return img_names,labels, counts, points_names
    
# for clf,
# def get_clf_data(datadir, split,  habitat=None ):
#     df = pd.read_csv(os.path.join(datadir,'Classification/%s.csv' % split))
#     df = slice_df(df, habitat)
#     img_names = np.array(df['ID'])
#     labels =  np.array(df['labels'])
#     return img_names, labels

# helpers
def slice_df(df, habitat):
    if habitat is None:
        return df
    return df[df['ID'].apply(lambda x: True if x.split("/")[0] == habitat else False)]

# bug_images_list = [
# "7482_F2_f000520",
# "7117_Lutjanus_argentimaculatus_adult_2_f000030",
# "7117_Lutjanus_a_testclip_far_away_short_f000180",
# "7398_F6_f000070",
# "7482_F2_f000290",
# "7585_F3_f000020",
# "7585_F4_f000000",
# "7623_F2_f000200",
# "7623_F2_f000240",
# "7623_F2_f000320",
# "7623_F2_f000460",
# "9866_acanthopagrus_palmaris_f000060",
# "9894_gerres_2_f000000",
# "9894_gerres_2_f000010"
# ]

#
#
# def getPaths(data_dir):
#     # read image files from directory
#     exts = ['*.png','*.PNG','*.jpg','*.JPG', '*.JPEG', '*.bmp']
#     image_paths = []
#     for pattern in exts:
#         for d, s, fList in os.walk(data_dir):
#             for filename in fList:
#                 if (fnmatch.fnmatch(filename, pattern)):
#                     fname_ = os.path.join(d,filename)
#                     image_paths.append(fname_)
#     return image_paths
#
# def getSaliency(mask):
#     # one combined category: HD/RO/FV/WR
#     imw, imh = mask.shape[0], mask.shape[1]
#     sal = np.zeros((imw, imh))
#     for i in range(imw):
#         for j in range(imh):
#             if (mask[i,j,0]==0 and mask[i,j,1]==0 and mask[i,j,2]==1):
#                 sal[i, j] = 1
#             elif (mask[i,j,0]==1 and mask[i,j,1]==0 and mask[i,j,2]==0):
#                 sal[i, j] = 1
#             elif (mask[i,j,0]==1 and mask[i,j,1]==1 and mask[i,j,2]==0):
#                 sal[i, j] = 1
#             elif (mask[i,j,0]==0 and mask[i,j,1]==1 and mask[i,j,2]==1):
#                 sal[i, j] = 0.8
#             else: pass
#     return np.expand_dims(sal, axis=-1)

# """
# RGB color code and object categories:
# ------------------------------------
# 000 BW: Background waterbody
# 001 HD: Human divers
# 010 PF: Plants/sea-grass
# 011 WR: Wrecks/ruins
# 100 RO: Robots/instruments
# 101 RI: Reefs and invertebrates
# 110 FV: Fish and vertebrates
# 111 SR: Sand/sea-floor (& rocks)
# """
# def getRobotFishHumanReefWrecks(mask):
#     # for categories: HD, RO, FV, WR, RI
#     imw, imh = mask.shape[0], mask.shape[1]
#     Human = np.zeros((imw, imh))
#     Robot = np.zeros((imw, imh))
#     Fish = np.zeros((imw, imh))
#     Reef = np.zeros((imw, imh))
#     Wreck = np.zeros((imw, imh))
#     for i in range(imw):
#         for j in range(imh):
#             if (mask[i,j,0]==0 and mask[i,j,1]==0 and mask[i,j,2]==1):
#                 Human[i, j] = 1
#             elif (mask[i,j,0]==1 and mask[i,j,1]==0 and mask[i,j,2]==0):
#                 Robot[i, j] = 1
#             elif (mask[i,j,0]==1 and mask[i,j,1]==1 and mask[i,j,2]==0):
#                 Fish[i, j] = 1
#             elif (mask[i,j,0]==1 and mask[i,j,1]==0 and mask[i,j,2]==1):
#                 Reef[i, j] = 1
#             elif (mask[i,j,0]==0 and mask[i,j,1]==1 and mask[i,j,2]==1):
#                 Wreck[i, j] = 1
#             else: pass
#     return np.stack((Robot, Fish, Human, Reef, Wreck), -1)
#
# def getRobotFishHumanWrecks(mask):
#     # for categories: HD, RO, FV, WR
#     imw, imh = mask.shape[0], mask.shape[1]
#     Human = np.zeros((imw, imh))
#     Robot = np.zeros((imw, imh))
#     Fish = np.zeros((imw, imh))
#     Wreck = np.zeros((imw, imh))
#     for i in range(imw):
#         for j in range(imh):
#             if (mask[i,j,0]==0 and mask[i,j,1]==0 and mask[i,j,2]==1):
#                 Human[i, j] = 1
#             elif (mask[i,j,0]==1 and mask[i,j,1]==0 and mask[i,j,2]==0):
#                 Robot[i, j] = 1
#             elif (mask[i,j,0]==1 and mask[i,j,1]==1 and mask[i,j,2]==0):
#                 Fish[i, j] = 1
#             elif (mask[i,j,0]==0 and mask[i,j,1]==1 and mask[i,j,2]==1):
#                 Wreck[i, j] = 1
#             else: pass
#     return np.stack((Robot, Fish, Human, Wreck), -1)
#
# def processSUIMDataRFHW(img, mask, sal=False):
#     # scaling image data and masks
#     img = img / 255
#     mask = mask /255
#     mask[mask > 0.5] = 1
#     mask[mask <= 0.5] = 0
#     m = []
#     for i in range(mask.shape[0]):
#         if sal:
#             m.append(getSaliency(mask[i]))
#         else:
#             m.append(getRobotFishHumanReefWrecks(mask[i]))
#             #m.append(getRobotFishHumanWrecks(mask[i]))
#     m = np.array(m)
#     return (img, m)



def getFish(mask):
    imw, imh = mask.shape[0], mask.shape[1]
    Fish = np.zeros((imw, imh))
    for i in range(imw):
        for j in range(imh):
            if (mask[i,j,0]==1 and mask[i,j,1]==1 and mask[i,j,2]==0):
                Fish[i, j] = 1
            else: pass
    return  Fish


def processSUIMDfish(img, mask):
    # scaling image data and masks
    img = img / 255
    mask = mask /255
    mask[mask > 0.5] = 1
    mask[mask <= 0.5] = 0
    m = getFish(mask)
    m = np.array(m)
    return (img, m)


""" python trainval.py -e weakly_SUMfish -sb <savedir_base> -d <datadir> -r 1"""