# from . import semseg_cost
import torch
import os
import tqdm 
from . import semseg, affinity, cam
import torch
from src.models import networks
from src.misc import indexing

def get_model(model_dict, exp_dict=None, train_set=None):
    if model_dict['name'] in ["wisenet"]:
        model =  wisenet.WiseNet(exp_dict, train_set)

    if model_dict['name'] in ["semseg_active"]:
        model =  semseg_active.get_semsegactive(semseg.SemSeg)(exp_dict, train_set)
    if model_dict['name'] in ["semseg_active_counting"]:
        model =  semseg_active.get_semsegactive(semseg_counting.SemSegCounting)(exp_dict, train_set)

    if model_dict['name'] in ["semseg_counting"]:
        model =  semseg_counting.SemSegCounting(exp_dict)

    if model_dict['name'] in ["semseg"]:
        model =  semseg.SemSeg(exp_dict, train_set=train_set)

        # load pretrained
        if 'pretrained' in model_dict:
            model.load_state_dict(torch.load(model_dict['pretrained']))
 
    if model_dict['name'] in ["affinity"]:
        n_pascal_images = 1464
        batch_size = exp_dict["batch_size"]
        max_epoch = exp_dict["max_epoch"]
        base_model = networks.get_network('resnet50_cam', 2, exp_dict).cuda()
        cam_model = cam.CAM(base_model=base_model, max_step=1464)
        # TODO: load state dict
        max_step = (n_pascal_images // exp_dict["batch_size"]) * exp_dict["max_epoch"]
        path_index = indexing.PathIndex(radius=10, default_size=(exp_dict["model"]["irn_crop_size"] // 4,
                                                                 exp_dict["model"]["irn_crop_size"] // 4))
        backbone = networks.resnet50_irn.AffinityDisplacementLoss(path_index)
        model = affinity.AFF(backbone, cam_model, max_step, exp_dict=exp_dict)

    return model




def max_norm(p, version='torch', e=1e-5):
	if version is 'torch':
		if p.dim() == 3:
			C, H, W = p.size()
			p = F.relu(p)
			max_v = torch.max(p.view(C,-1),dim=-1)[0].view(C,1,1)
			min_v = torch.min(p.view(C,-1),dim=-1)[0].view(C,1,1)
			p = F.relu(p-min_v-e)/(max_v-min_v+e)
		elif p.dim() == 4:
			N, C, H, W = p.size()
			p = F.relu(p)
			max_v = torch.max(p.view(N,C,-1),dim=-1)[0].view(N,C,1,1)
			min_v = torch.min(p.view(N,C,-1),dim=-1)[0].view(N,C,1,1)
			p = F.relu(p-min_v-e)/(max_v-min_v+e)
	elif version is 'numpy' or version is 'np':
		if p.ndim == 3:
			C, H, W = p.shape
			p[p<0] = 0
			max_v = np.max(p,(1,2),keepdims=True)
			min_v = np.min(p,(1,2),keepdims=True)
			p[p<min_v+e] = 0
			p = (p-min_v-e)/(max_v+e)
		elif p.ndim == 4:
			N, C, H, W = p.shape
			p[p<0] = 0
			max_v = np.max(p,(2,3),keepdims=True)
			min_v = np.min(p,(2,3),keepdims=True)
			p[p<min_v+e] = 0
			p = (p-min_v-e)/(max_v+e)
	return p

def adaptive_min_pooling_loss(x):
    # This loss does not affect the highest performance, but change the optimial background score (alpha)
    n,c,h,w = x.size()
    k = h*w//4
    x = torch.max(x, dim=1)[0]
    y = torch.topk(x.view(n,-1), k=k, dim=-1, largest=False)[0]
    y = F.relu(y, inplace=False)
    loss = torch.sum(y)/(k*n)
    return loss

def max_onehot(x):
    n,c,h,w = x.size()
    x_max = torch.max(x[:,1:,:,:], dim=1, keepdim=True)[0]
    x[:,1:,:,:][x[:,1:,:,:] != x_max] = 0
    return x

def text_on_image(text, image, color=None):
    """Adds test on the image
    
    Parameters
    ----------
    text : [type]
        [description]
    image : [type]
        [description]
    
    Returns
    -------
    [type]
        [description]
    """
    import cv2
    font = cv2.FONT_HERSHEY_SIMPLEX
    bottomLeftCornerOfText = (10,40)
    fontScale              = 0.8
    if color is None:
        fontColor              = (1,1,1)
    else:
        fontColor              = color
    lineType               = 1
    # img_mask = skimage.transform.rescale(np.array(img_mask), 1.0)
    # img_np = skimage.transform.rescale(np.array(img_points), 1.0)
    img_np = cv2.putText(image, text, 
        bottomLeftCornerOfText, 
        font, 
        fontScale,
        fontColor,
        thickness=2
        # lineType
        )
    return img_np