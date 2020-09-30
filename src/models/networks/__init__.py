from . import fcn8_vgg16, fcn8_vgg16_multiscale, unet2d, unet_resnet, attu_net, fcn8_resnet, deeplab
from . import resnet_seam, infnet
from . import resnet50_cam, resnet50_irn, resnet50, fcn8_resnet
from torchvision import models
import torch, os
import torch.nn as nn


def get_network(network_name, n_classes, exp_dict):
    if network_name == 'infnet':
        model_base = infnet.InfNet(n_classes=1, loss=exp_dict['model']['loss'])

    if network_name == 'fcn8_vgg16_att':
        model_base = fcn8_vgg16.FCN8VGG16(n_classes=n_classes, with_attention=True)

    if network_name == 'fcn8_vgg16':
        model_base = fcn8_vgg16.FCN8VGG16(n_classes=n_classes, 
                    with_attention=exp_dict['model'].get('with_attention'),
                    with_affinity=exp_dict['model'].get('with_affinity'),
                    with_affinity_average=exp_dict['model'].get('with_affinity_average'),
                    shared=exp_dict['model'].get('shared'),
                    exp_dict=exp_dict
                    )

    if network_name == "fcn8_vgg16_multiscale":
        model_base = fcn8_vgg16_multiscale.FCN8VGG16(n_classes=n_classes)

    if network_name == "unet_resnet":
        model_base = unet_resnet.ResNetUNet(n_class=n_classes)
    
    if network_name == "resnet_seam":
        model_base = resnet_seam.ResNetSeam()
        # path_base = '/mnt/datasets/public/issam/seam'
        # model_base.load_state_dict(torch.load(os.path.join(path_base, 'resnet38_SEAM.pth')))
        weights_dict = model_base.resnet38d.convert_mxnet_to_torch(args.weights)

        model.load_state_dict(weights_dict, strict=False)

    if network_name == "attu_net":
        model_base = attu_net.AttU_Net()

    if network_name == "resnet50_cam":
        return resnet50_cam.Net(n_classes=n_classes)
    elif network_name == "resnet50_irn":
        return resnet50_irn
    elif network_name == "fcn8_resnet":
        return fcn8_resnet.FCN8(n_classes)

    return model_base

