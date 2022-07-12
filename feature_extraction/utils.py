'''
Author: zhouqy
Date: 2022-06-04 11:40:49
LastEditors: zhouqy
LastEditTime: 2022-07-11 23:43:09
Description: Model loading, image transform, feature extraction of different models
'''
import os
import torch
import urllib
from torchvision import transforms as trn


import timm
from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD,IMAGENET_INCEPTION_MEAN,IMAGENET_INCEPTION_STD
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform

# model zoo
from alexnet import *


def load_alexnet(model_checkpoints):
    """This function initializes an Alexnet and load
    its weights from a pretrained model

    Args:
    model_checkpoints (str): model checkpoints location.

    Returns:
    model
        pytorch model of alexnet
    """
    model = alexnet()
    model_file = model_checkpoints
    checkpoint = torch.load(model_file, map_location=lambda storage, loc: storage)
    model_dict =["conv1.0.weight", "conv1.0.bias", "conv2.0.weight", "conv2.0.bias", "conv3.0.weight", "conv3.0.bias", "conv4.0.weight", "conv4.0.bias", "conv5.0.weight", "conv5.0.bias", "fc6.1.weight", "fc6.1.bias", "fc7.1.weight", "fc7.1.bias", "fc8.1.weight", "fc8.1.bias"]
    state_dict={}
    i=0
    for k,v in checkpoint.items():
        state_dict[model_dict[i]] =  v
        i+=1

    model.load_state_dict(state_dict)
    if torch.cuda.is_available():
        model.cuda()
    model.eval()
    return model


def load_model(model_name):
    """This function specifies the model name and load
    its weights from a pretrained model

    Args:
    model_name (str): model name used to extract feature.

    Returns:
    model
        model with pretrained model weights
    """

    if model_name == 'alexnet':
        # load Alexnet
        # Download pretrained Alexnet from:
        # https://download.pytorch.org/models/alexnet-owt-4df8aa71.pth
        # and save in the current directory
        checkpoint_path = "./alexnet.pth"
        if not os.path.exists(checkpoint_path):
            url = "https://download.pytorch.org/models/alexnet-owt-4df8aa71.pth"
            urllib.request.urlretrieve(url, "./alexnet.pth")
        model = load_alexnet(checkpoint_path)
    else:
        return
    return model


def image_transform(model_name):
    """_summary_

    Args:
        model_name (str): model name used to extract feature.

    Returns:
        transform
    """

    if model_name == 'alexnet':
        img_mean = IMAGENET_DEFAULT_MEAN
        img_std = IMAGENET_DEFAULT_STD
    else:
        return
    transform = trn.Compose([
    trn.ToPILImage(),
    trn.Resize(224),
    trn.ToTensor(),
    trn.Normalize(img_mean, img_std),
    ])
    return transform

def feature_extraction(model_name, model, image):
    """feature extraction. This function can be extended to different models.

    Args:
        model_name (str): specify model name.
        model (model): model
        image (torch.Tensor): image input

    Returns:
        x (list): a list of features of multiple candidate layers. 
    """
    if model_name == 'alexnet':
        x = model.forward(image)
    else:
        return
    return x
