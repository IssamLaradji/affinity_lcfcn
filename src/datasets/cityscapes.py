#%%
# from haven import utils as mlkit_ut

from PIL import Image
import numpy as np
import torch
from scipy.ndimage.morphology import distance_transform_edt
import os
from haven import haven_utils as hu
from haven import haven_img as hi
from skimage.io import imread
from scipy.io import loadmat
import torchvision.transforms.functional as FT
import numpy as np
import torch
from skimage.io import imread
import torchvision.transforms.functional as FT
from skimage.transform import rescale
import torchvision
from torchvision import datasets
from torchvision.transforms import transforms
import pylab as plt
from skimage.color import label2rgb
# from repos.selectivesearch.selectivesearch import selective_search


class CityScapes:
    def __init__(self, split, exp_dict):

        path = '/mnt/public/datasets/cityscapes'
        self.split = split
        self.exp_dict = exp_dict
        self.n_classes = 19

        if split == "train":
            resize = True
            flip = False
            
            
            self.transforms = lambda image, targets:joint_transform(image, 
                                                targets, resize=resize, flip=flip)
        else:
            resize = True
            split = 'val'
            self.transforms = lambda image, targets:joint_transform_val(image, 
                                            targets, resize=resize)
            
        # self.effort_per_image = 90*60
        self.dataset = CityScapesTrainIds(path, 
                    split=split, 
                    mode='fine',
                    # transforms=transforms,
                    target_type=['semantic','instance'])

        self.transform = joint_transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        images, targets_instance = self.dataset[index]
        targets, mask_inst = targets_instance
        images, targets, flipped = self.transforms(images, targets)

        mask_inst = transforms.Resize(size=512, interpolation=Image.NEAREST)(mask_inst)
        mask_inst = torch.from_numpy(np.array(mask_inst))
        mask_inst = mask_inst.long()
        inst = torch.zeros(targets.shape).long()
        classes = np.unique(targets)

        category_id2label_id = {28:1, 26:2, 27:3, 24:4, 25:5,
                                32:6, 31:7, 33:8}

        uniques = np.unique(mask_inst)
        point_list = []
        selected = set()
        inst_id = 0
        for category_id in category_id2label_id.keys():
            instances = uniques[(uniques>=category_id*1000) & (uniques<(category_id+1)*1000)]
            if len(instances) == 0:
                continue

            # ind = ((mask_inst>=category_id*1000) & 
            #          (mask_inst<(category_id+1)*1000))
            # class_id = category_id2label_id[category_id]
            for i, u in enumerate(instances):
                seg_ind = mask_inst==u
                inst[seg_ind] = inst_id
                inst_id += 1
                dist = distance_transform_edt(seg_ind)
                yx = np.unravel_index(dist.argmax(), dist.shape)
                class_id = int(targets[yx])
                if class_id == 255:
                    continue

                if class_id >= self.n_classes:
                    raise ValueError('not found')
                
                selected.add(class_id)
                point_list += [{'y':yx[0], 'x':yx[1], 'cls':class_id}]
                
        for l in np.setdiff1d(classes, list(selected) + [255]):
            y_list, x_list = np.where(targets == l)
            yc, xc = get_median(y_list, x_list)
            class_id = int(targets[yc, xc])
            if class_id == 255:
                    continue
                
            if class_id >= self.n_classes:
                    raise ValueError('not found')

            point_list += [{'y':yc, 'x':xc, 'cls':class_id}]

        # if 1:
        #     y_list = [p['y'] for p in point_list]
        #     x_list = [p['x'] for p in point_list]
        #     img_prop_lbl = hi.points_on_image(y_list, x_list, inv_transform(images), radius=10)
        #     hu.save_image('tmp.png', img_prop_lbl)

        assert flipped == False
        # cost_mask = CsObject().get_clicks_from_polygons(images.size(1), images.size(2), polygons)
        # cost = torch.from_numpy(cost)

        # masks, cost = targets
        # regions = self.region[index]
        
        # batch_dict = {'images':img, 
        #               'masks':mask,
        #               'meta':{'index':index, 'split':self.split, 'size':(H,W), 'name':name}}
        # point_list = pascal.get
        H,W = images.shape[-2:]
        points = torch.ones((H,W)) * 255
        for p in point_list:
            points[p['y'], p['x']] = p['cls']
        # hu.save_image(fname='tmp.png', img=hu.get_image(images, denorm=True), 
        #             points=(points!=255), )
        batch = {"images": images,
                #  "cost_mask": cost_mask,
                #  "region_list": region_list,
                # 'inst':inst,
                'point_list':point_list,
                'points':points,
                 'flipped':flipped,
                 "masks": targets,
                 "original":inv_transform(images),
                 "meta": {"index": index,
                        'hash':hu.hash_dict({'id':index, 'split':self.split}),
                          "name": self.dataset.images[index],
                          "size": images.shape[-2:],
                          "image_id": index,
                          "split": self.split}}
        return batch

# =====================================
# helpers
def get_random(y_list, x_list):
    with hu.random_seed(1):
        yi = np.random.choice(y_list)
        x_tmp = x_list[y_list == yi]
        xi = np.random.choice(x_tmp)

    return yi, xi

def get_median(y_list, x_list):
    tmp = y_list
    mid = max(0, len(tmp)//2 - 1)
    yi = tmp[mid]
    tmp = x_list[y_list == yi]
    mid = max(0, len(tmp)//2 - 1)
    xi = tmp[mid]

    return yi, xi

def joint_transform(image, targets, resize=False, flip=False):
    mask = targets

    # Resize
    if resize:
        image = transforms.Resize(size=512, interpolation=Image.BILINEAR)(image)
        mask = transforms.Resize(size=512, interpolation=Image.NEAREST)(mask)

    # Random crop
    # i, j, h, w = transforms.RandomCrop.get_params(
    #     image, output_size=(256, 512))
    # image = FT.crop(image, i, j, h, w)
    # mask = FT.crop(mask, i, j, h, w)

    # Random horizontal flipping
    flipped = False
    if np.random.randint(2) == 0 and flip:
        image = FT.hflip(image)
        mask = FT.hflip(mask)
        flipped = True

    # # Random vertical flipping
    # if random.random() > 0.5:
    #     image = FT.vflip(image)
    #     mask = FT.vflip(mask)

    # Transform to tensor
    image = FT.to_tensor(image)
    mask = torch.from_numpy(np.array(mask))
    mask = mask.long()

    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    image_normalized = transforms.Normalize(mean=mean, std=std)(image)

    # cost = CsObject().get_clicks_from_polygons(image.size(1), image.size(2), polygons)
    # cost = torch.from_numpy(cost)
    # return image_normalized, (mask, cost)
    return image_normalized, mask, flipped

# =====================================
# helpers
def joint_transform_val(image, targets, resize=False):
    mask = targets
    if resize:
        # Resize
        image = transforms.Resize(size=512, interpolation=Image.BILINEAR)(image)
        mask = transforms.Resize(size=512, interpolation=Image.NEAREST)(mask)

    # # Random crop
    # i, j, h, w = transforms.RandomCrop.get_params(
    #     image, output_size=(256, 512))
    # image = FT.crop(image, i, j, h, w)
    # mask = FT.crop(mask, i, j, h, w)

    # # Random horizontal flipping
    # if random.random() > 0.5:
    #     image = FT.hflip(image)
    #     mask = FT.hflip(mask)

    # # Random vertical flipping
    # if random.random() > 0.5:
    #     image = FT.vflip(image)
    #     mask = FT.vflip(mask)

    # Transform to tensor
    image = FT.to_tensor(image)
    mask = torch.from_numpy(np.array(mask))
    mask = mask.long()

    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    image_normalized = transforms.Normalize(mean=mean, std=std)(image)

    # cost = CsObject().get_clicks_from_polygons(image.size(1), image.size(2), polygons)
    # cost = torch.from_numpy(cost)
    # return image_normalized, (mask, cost)
    flipped = False
    return image_normalized, mask, flipped

def inv_transform(images):
    inv_normalize = transforms.Normalize(
    mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225],
    std=[1/0.229, 1/0.224, 1/0.255])

    inv_image = inv_normalize(images.float())
    images_arr = np.array(FT.to_pil_image(inv_image.float()))

    return images_arr


class CityScapesTrainIds(datasets.Cityscapes):
    def __init__(self, root, split='train', mode='fine', target_type='instance',
                 transform=None, target_transform=None, transforms=None):
        super().__init__(root, split, mode, target_type,
                 transform, target_transform, transforms)

    def _get_target_suffix(self, mode, target_type):
        if target_type == 'instance':
            return '{}_instanceIds.png'.format(mode)
        elif target_type == 'semantic':
            # return '{}_labelIds.png'.format(mode)
            return '{}_labelTrainIds.png'.format(mode)
        elif target_type == 'color':
            return '{}_color.png'.format(mode)
        else:
            return '{}_polygons.json'.format(mode)


if __name__ == "__main__":
    dataset = CityScapes(split="train", exp_dict={})
    batch = dataset[0]
    images = batch["images"]
    masks = batch["masks"]

    inv_normalize = transforms.Normalize(
    mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225],
    std=[1/0.229, 1/0.224, 1/0.255])

    inv_image = inv_normalize(images.float())
    
    # combined = 0.5*inv_image + 0.5*masks
    masks_arr = np.array(FT.to_pil_image(masks.float()))
    images_arr = np.array(FT.to_pil_image(inv_image.float()))

    image_label_overlay = label2rgb(masks_arr, image=images_arr)

    inv_image = FT.to_pil_image(inv_image)

    plt.imshow(images_arr)
    plt.savefig("img.png")
    plt.imshow(masks_arr)
    plt.savefig("mask.png")
    plt.imshow(image_label_overlay)
    plt.savefig("overlay.png")