
###
# This file will:
# 1. Generate and save features in a given folder
###
import timm
from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform

import glob
import numpy as np
import urllib
import torch
import cv2
import argparse
import time
import random
from tqdm import tqdm
from torchvision import transforms as trn
import os

from PIL import Image
from sklearn.preprocessing import StandardScaler
from torch.autograd import Variable as V
from sklearn.decomposition import PCA, IncrementalPCA
from sklearn.random_projection import SparseRandomProjection
from utils import *
seed = 42
# Torch RNG
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
# Python RNG
np.random.seed(seed)
random.seed(seed)


def get_video_from_mp4(file, sampling_rate):
    cap = cv2.VideoCapture(file)
    frameCount = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frameWidth = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frameHeight = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    buf = np.empty((int(frameCount / sampling_rate), frameHeight,
                   frameWidth, 3), np.dtype('uint8'))
    fc = 0
    ret = True
    while fc < frameCount and ret:
        fc += 1
        (ret, frame) = cap.read()
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        if fc % sampling_rate == 0:
            buf[int((fc - 1) / sampling_rate)] = frame

    cap.release()
    return np.expand_dims(buf, axis=0),int(frameCount / sampling_rate)


def get_activations_and_save(model_name, video_list, save_dir, dim_rd, 
                    test_idx, pca_ratio, sampling_rate = 4, token='cls'):


    model = load_model(model_name)
    transform = image_transform(model_name, model=model)

    fea_tot = []
    layer_3d = 0
    j = 0
    for video_file in tqdm(video_list):
        vid,num_frames = get_video_from_mp4(video_file, sampling_rate)
        video_file_name = os.path.split(video_file)[-1].split(".")[0]
        activations = []
        for frame in range(num_frames):
            img = vid[0, frame, :, :, :]
            input_img = transform(img).unsqueeze(0)
            if torch.cuda.is_available():
                input_img=input_img.cuda()
            with torch.no_grad():
                x = feature_extraction(model_name, model, input_img)
            for i,feat in enumerate(x):
                if frame==0:
                    # print(feat.size())
                    feat_tmp = feat.data.cpu().numpy()
                    if j == 0:
                        if len(feat_tmp.shape) == 2 and layer_3d == 0:
                            layer_3d = i
                    activations.append(feat_tmp.ravel())
                else:
                    activations[i] =  activations[i] + feat.data.cpu().numpy().ravel()
        activ_aver = [a/float(num_frames) for a in activations]
        fea_tot.append(activ_aver)
        j += 1
    
    print('Features before layer-%d can be considered dimension reduction'%layer_3d)
    for layer in range(len(activ_aver)):
        fea_tot_layer = [fea_tot[i][layer] for i in range(len(fea_tot))]
        fea_tot_layer = np.stack(fea_tot_layer)
        print('not_rd', layer, fea_tot_layer.shape[-1])
        x_train = np.delete(fea_tot_layer[:1000], test_idx, 0)
        x_test = fea_tot_layer[:1000][test_idx]
        if layer < layer_3d:
            if dim_rd == 'pca':
                x_train = StandardScaler().fit_transform(x_train)
                x_test = StandardScaler().fit_transform(x_test)
                ipca = PCA(n_components=pca_ratio, svd_solver='full')
                ipca.fit(x_train)
                x_train = ipca.transform(x_train)
                x_test = ipca.transform(x_test)
                print(dim_rd, layer, ipca.n_components_)
            elif dim_rd == 'srp':
                x_train = StandardScaler().fit_transform(x_train)
                x_test = StandardScaler().fit_transform(x_test)
                isrp = SparseRandomProjection()
                isrp.fit(x_train)
                x_train = isrp.transform(x_train)
                x_test = isrp.transform(x_test)
                print(dim_rd, layer, isrp.n_components_)

        if not os.path.exists(os.path.join(save_dir, dim_rd)):
            os.makedirs(os.path.join(save_dir, dim_rd))
        train_save_path = os.path.join(save_dir, dim_rd, "train_%s"%str(layer+1).zfill(2))
        test_save_path = os.path.join(save_dir, dim_rd, "test_%s"%str(layer+1).zfill(2))
        np.save(train_save_path, x_train)
        np.save(test_save_path, x_test)

def main():
    parser = argparse.ArgumentParser(description='Feature Extraction')
    parser.add_argument('-rp','--root_path', help='root path', type=str)
    parser.add_argument('-vdir','--video_data_dir', help='video data directory',
            default = 'AlgonautsVideos268_All_30fpsmax', type=str)
    parser.add_argument('-fname','--feature_name', help='feature name, for example, alexnet or vit',
            default = 'alexnet', type=str)
    parser.add_argument('-ad','--activation_dir',help='directory containing DNN activations',
                        default = 'feature_zoo', type=str)
    parser.add_argument('-dim_rd','--dim_rd', help='feature dimension reduction', 
            default = 'srp', type=str)
    parser.add_argument('-pca_ratio','--pca_ratio', help='pca component ratio', 
            default = 0.98, type=float)
    parser.add_argument('-srate','--sample_rate', help='sample rate', default = 4, type=int)
    parser.add_argument('-token', '--token', help='token category', default='cls', type=str)
    parser.add_argument('-gpu', '--gpu_id', help='gpu id', default='7', type=str)
    args = vars(parser.parse_args())


    os.environ['CUDA_VISIBLE_DEVICES']=args['gpu_id']
    save_dir = os.path.join(args['root_path'], args['activation_dir'], args['feature_name'])
    if not os.path.exists(save_dir):
            os.makedirs(save_dir)

    video_dir = os.path.join(args['root_path'], args['video_data_dir'])
    video_list = glob.glob(os.path.join(video_dir, '*.mp4'))
    video_list.sort()
    print('Total Number of Videos: ', len(video_list))
    
    # split test set from training set.
    if os.path.exists(os.path.join(args['root_path'], 'test_idx')):
        test_idx = np.load(os.path.join(args['root_path'], 'test_idx.npy'))
    else:
        np.random.seed(400)
        test_idx = np.random.permutation(np.arange(1000))[:100]
        np.save(os.path.join(args['root_path'], 'test_idx'), test_idx)

    print("-------------Saving activations ----------------------------")
    get_activations_and_save(args['feature_name'], video_list, save_dir, args['dim_rd'], 
                            sampling_rate=args['sample_rate'], test_idx=test_idx, 
                            pca_ratio=args['pca_ratio'],
                            token=args['token'])


if __name__ == "__main__":
    main()
