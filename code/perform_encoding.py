'''
Author: zhouqy
Date: 2022-06-04 11:40:49
LastEditors: zhouqy
LastEditTime: 2022-07-11 23:41:14
Description:  Using features from all layers to 
predict neural activities of different ROIs and different subjects.
'''
import numpy as np
import os
import glob
import argparse
import csv
import time
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
import torch
from torch import nn
from torch import optim
from torch.utils.data import Dataset, DataLoader
from tensorboardX import SummaryWriter
import time
import pickle
from earlystopping import EarlyStopping
import pandas as pd



def vectorized_correlation(x,y):
    """Calculate the PCC.

    Args:
        x (np.array of shape (sample_num, voxel_num)): True responses.
        y (np.array of shape (sample_num, voxel_num)): Predicted reponses.

    Returns:
        corr_vec (np.array of shape (voxel_num)): 
            PCC between the true and predicted responses of different voxels.
    """
    dim = 0

    centered_x = x - x.mean(axis=dim, keepdims=True)
    centered_y = y - y.mean(axis=dim, keepdims=True)

    covariance = (centered_x * centered_y).sum(axis=dim, keepdims=True)

    bessel_corrected_covariance = covariance / (x.shape[dim] - 1)

    x_std = x.std(axis=dim, keepdims=True)+1e-8
    y_std = y.std(axis=dim, keepdims=True)+1e-8

    corr = bessel_corrected_covariance / (x_std * y_std)
    corr_vec = corr.ravel()
    return corr_vec


def weights_init(m):
    '''
    Initialize weights of linear layer.
    '''
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.xavier_normal_(m.weight.data)
        nn.init.constant_(m.bias.data, 0.0)


class FC_reg(nn.Module):
    
    def __init__(self, o_dim, i_dim, dim_list):
        """Create FC network with layer weights

        Args:
            o_dim (int): the number of voxels.
            i_dim (int): feature dimension, equal to sum(dim_list)
            dim_list (list): a list with dimensions of features of all layers.
        """
        super(FC_reg, self).__init__()
        self.dim_list = dim_list
        self.model = nn.Linear(i_dim, o_dim)
        self.layer_weight = torch.nn.Parameter(torch.ones(len(dim_list))/len(dim_list),
                                                requires_grad=True)
    def forward(self, x):
        fil = []
        self.layer_weight_pos = torch.abs(self.layer_weight)
        for i in range(len(self.dim_list)):
            w = self.layer_weight_pos[i].repeat(self.dim_list[i])
            fil.append(w)
        fil = torch.cat(fil, 0).unsqueeze(0)
        x_fil = torch.mul(x, fil)

        out = self.model(x_fil)
        return out

class Ds_loader(Dataset):
    def __init__(self, train_activ, train_fmri, test_activ, test_fmri, 
                        split='train'):
        """load dataset (feature, fmri)

        Args:
            train_activ (np.array): image feature array of the training set
            train_fmri (np.array): fmri array of the training set
            test_activ (np.array): image feature array of the test set
            test_fmri (np.array): fmri array of the test set
            split (str, optional): load training set or test set. Defaults to 'train'.
        """
        if split == 'train':
            self.activ = train_activ
            self.fmri = train_fmri
        elif split == 'test':
            self.activ = test_activ
            self.fmri = test_fmri
        print(self.activ.shape, self.fmri.shape)

    def __getitem__(self, idx):
        activ = self.activ[idx]
        fmri = self.fmri[idx]
        return activ, fmri

    def __len__(self):
        return self.fmri.shape[0]


def get_activations(activations_dir):
    """This function loads neural network features/activations 
    into a numpy array.

    Args:
    activations_dir : str
        Path of Neural Network features

    Returns:
    train_activations : np.array
        matrix of dimensions # train_sample_number x # dimension
        containing activations of train videos
    test_activations : np.array
        matrix of dimensions #test_sample_number x # dimension
        containing activations of test videos
    """
    train_files = glob.glob(os.path.join(activations_dir, 'train_*.npy'))
    test_files = glob.glob(os.path.join(activations_dir, 'test_*.npy'))
    train_files.sort()
    test_files.sort()
    train_activations = []
    test_activations = []
    dim_list = []   # record dimension of features from every layer.
    for file in train_files:
        arr = np.load(file).squeeze()
        train_activations.append(arr)
        dim_list.append(arr.shape[-1])
    for file in test_files:
        arr = np.load(file).squeeze()
        test_activations.append(arr)
    train_activations = np.hstack(train_activations)
    test_activations = np.hstack(test_activations)


    scaler = StandardScaler()
    train_activations = scaler.fit_transform(train_activations)
    test_activations = scaler.fit_transform(test_activations)
    print(dim_list)
    return train_activations, test_activations, dim_list


def load_dict(filename_):
    """load .pkl data

    Args:
        filename_ (str): the path of .pkl data file.

    Returns:
        ret_di (dict): data
    """
    with open(filename_, 'rb') as f:
        u = pickle._Unpickler(f)
        u.encoding = 'latin1'
        ret_di = u.load()
    return ret_di

def get_fmri(fmri_dir, ROI, test_idx):
    """This function loads fMRI data 
    into a numpy array for to a given ROI.

    Args:
    fmri_dir : str
        path to fMRI data.
    ROI : str
        name of ROI.

    Returns:
    ROI_data_train: np.array
    matrix of dimensions #train_vids x #voxels
    containing fMRI responses to train videos of a given ROI

    ROI_data_test: np.array
    """
    # Loading ROI data
    ROI_file = os.path.join(fmri_dir, ROI + ".pkl")
    ROI_data = load_dict(ROI_file)

    # averaging ROI data across repetitions
    ROI_data_tot = np.mean(ROI_data["train"], axis = 1)
    ROI_data_train = np.delete(ROI_data_tot, test_idx, 0)
    ROI_data_test = ROI_data_tot[test_idx]
    if ROI == "WB":
        voxel_mask = ROI_data['voxel_mask']
        return ROI_data_train, ROI_data_test, voxel_mask

    return ROI_data_train, ROI_data_test

def predict_fmri_fc2(train_activations, test_activations, 
                        train_fmri, test_fmri,
                        batch_size, epochs, summary,
                        w_coe, dim_list,layer_coe, model_dir, metric, patience, delta):
    """This function fits a layer-weighted linear regressor using train_activations and train_fmri,
    then returns the predicted fmri_pred_test using the fitted weights and
    test_activations.

    Args:
    train_activations : np.array
        matrix of dimensions #train_vids x #pca_components
        containing activations of train videos.
    test_activations : np.array
        matrix of dimensions #test_vids x #pca_components
        containing activations of test videos
    train_fmri : np.array
        matrix of dimensions #train_vids x  #voxels
        containing fMRI responses to train videos
    test_fmri : np.array
        matrix of dimensions #test_vids x  #voxels
        containing fMRI responses to test videos
    batch_size : int, batch size
    epochs : int, maximum iteration number
    summary: tensorboard summary
    w_coe: float, regularization coefficient of weight
    dim_list: list, a list of feature dimensions of all layers
    layer_coe: float, regularization coefficient of layer weight
    model_dir: str, directory to save model
    metric: str, metric for early-stopping
    patience/delta: int/float, early-stopping patience, delta
    Returns:
    model, early-stopping epoch number
    """
    data_loader = {split: DataLoader(
            Ds_loader(train_activations, train_fmri, test_activations, test_fmri, 
            split=split), 
            batch_size=batch_size,
            shuffle=(split=='train'), drop_last=True) for split in ('train', 'test')}

    fea_dim = sum(dim_list)
    print('Dimension of all features=', fea_dim)
    reg = FC_reg(train_fmri.shape[1], fea_dim, dim_list).cuda()
    reg.apply(weights_init)
    optimizer = optim.Adam(reg.parameters(), lr=1e-4)
    mseloss = nn.MSELoss()
    early_stopping = EarlyStopping(patience, verbose=True, delta=delta, 
                                    path=os.path.join(model_dir, 'model.pth'))


    for epoch in range(epochs):

        mses = {'train':0, 'test':0}
        pccs = {'train':0, 'test':0}
        w_regs = {'train':0, 'test':0}
        layer_spar_regs = {'train':0, 'test':0}
        tot_losses = {'train':0, 'test':0}

        for split in ('train', 'test'):
            if split == 'train':
                reg.train()
            else:
                reg.eval()
            for i,(activ, fmri) in enumerate(data_loader[split]):
                activ = activ.float().cuda()
                fmri = fmri.float().cuda()
                pred_fmri = reg(activ)
                loss = mseloss(pred_fmri, fmri)
                w_reg = torch.sqrt(torch.mean(reg.model.weight**2, 1)).mean()
                layer_spar_reg = torch.mean(torch.abs(reg.layer_weight))
                pcc = vectorized_correlation(fmri.detach().cpu().numpy(),
                                              pred_fmri.detach().cpu().numpy()).mean()
                tot_loss = loss + w_coe * w_reg + \
                            layer_coe * layer_spar_reg

                mses[split] += loss
                w_regs[split] += w_reg
                layer_spar_regs[split] += layer_spar_reg
                pccs[split] += pcc
                tot_losses[split] += tot_loss
                if split == 'train':
                    reg.zero_grad()
                    tot_loss.backward()
                    optimizer.step()

            for key in ['mses', 'w_regs', 'layer_spar_regs', 'pccs', 'tot_losses']:
                summary.add_scalar(split+'/'+key, eval(key)[split]/(i+1), epoch+1)

        if metric == 'mse':
            valid_loss = mses['test']/(i+1)
        elif metric == 'pcc':
            valid_loss = -pccs['test']/(i+1)
        early_stopping(valid_loss, reg, epoch)
        if early_stopping.early_stop:
            print("Early stopping")
            break
    reg.load_state_dict(torch.load(os.path.join(model_dir, 'model.pth')))
    summary.close()
    return reg, early_stopping.epoch

def write_csv(data, header, csv_path):
    """Write results into .csv file.

    Args:
        data (dict): Store multiple rows of data in the form of a 
                    dictionary that needs to be written to a csv
        header (list): header
        csv_path (str): Save to this file.
    """

    if not os.path.isfile(csv_path):
        with open(csv_path, "w") as csv_file:
            writer = csv.DictWriter(
                csv_file, fieldnames=header)
            writer.writeheader()
    with open(csv_path, "a") as csv_file:
        writer = csv.DictWriter(
            csv_file, fieldnames=header)
        writer.writerow(data)


def main():

    parser = argparse.ArgumentParser(description='Encoding model analysis for Algonauts 2021')
    parser.add_argument('-rp','--root_path', help='root path', type=str)
    parser.add_argument('-rd','--result_dir', help='saves predicted fMRI activity',
                        default = 'results', type=str)
    parser.add_argument('-ad','--activation_dir',help='directory containing DNN features',
                        default = 'feature_zoo', type=str)
    parser.add_argument('-model','--model',help='model name under which predicted fMRI activity will be saved', 
                        default = 'alexnet', type=str)
    parser.add_argument('-m','--mode',help='hyper_tune, train, test', default = 'train', type=str)
    parser.add_argument('-fd','--fmri_dir',help='directory containing fMRI activity', 
                        default = 'participants_data_v2021', type=str)
    parser.add_argument('-b', '--batch_size',help='batch size of dnn training', default = 16, type=int)
    parser.add_argument('-e', '--epochs',help='epochs of dnn training', default = 1, type=int)
    parser.add_argument('-cp', '--csv_path',help='path of csv file', default = 'summary_', type=str)
    parser.add_argument('-fn', '--fold_num',help='kfold cross validation', default = 10, type=int)
    parser.add_argument('-wc', '--w_coe',help='coe of weight regularization', default = 0, type=float)
    parser.add_argument('-lay', '--layer_coe',help='coe of layer sparseness regularization', default = 0, type=float)
    parser.add_argument('-sub', '--sub_num',help='subject number', default = 1, type=int)
    parser.add_argument('-roi', '--roi',help='roi name', type=str)
    parser.add_argument('-gpu', '--gpu_id', help='gpu id', default='7', type=str)
    parser.add_argument('-dim_rd','--dim_rd', help='feature dimension reduction', default = 'not_rd', type=str)
    parser.add_argument('-ti_st', '--time_stamp', help='only need in mode test', default=' ', type=str)
    # settings for early-stopping
    parser.add_argument('-mtc', '--metric', help='metric used for early-stopping', 
                        default='mse', type=str, choices=['mse', 'pcc'])
    parser.add_argument('-pat', '--patience',help='patience time for early-stopping', 
                        default = 1, type=int)
    parser.add_argument('-delta', '--delta',help='delta value for early-stopping', 
                        default = 1e-2, type=float)

    parser.add_argument('-comp', '--component', default = 0, type=int)
    args = vars(parser.parse_args())
    print(args)
    
    os.environ['CUDA_VISIBLE_DEVICES']=args['gpu_id']
    w_coe = args['w_coe']
    layer_coe = args['layer_coe']
    hyper_suffix = '_'.join([str(args['w_coe']), str(args['layer_coe'])])
    track = 'mini_track'
    ROI = args['roi']
    mode = args['mode']
    batch_size = args['batch_size']
    epochs = args['epochs']
    num_subs = args['sub_num']

    activation_dir = os.path.join(args['root_path'], 
                            args['activation_dir'], args['model'], args['dim_rd'])
    fmri_dir = os.path.join(args['root_path'], args['fmri_dir'], track)
    root_path = os.path.join(args['root_path'], 
            args['result_dir'], args['model'], args['dim_rd'], 'multilayer', track)
    csv_save_path = os.path.join(root_path, args['csv_path']+(ROI)+'.csv')


    time_stamp = time.strftime("%m%d%H%M%S", time.localtime())
    subs=[]
    for s in range(num_subs):
      subs.append('sub'+str(s+1).zfill(2))
    layer_contribution = []
    # load features, same for different subjects.
    train_activations, test_activations, dim_list = get_activations(activation_dir)
    test_idx = np.load(os.path.join(args['root_path'], 'test_idx.npy'))
    # fit model for each subject.
    for sub in subs:
        print(sub)
        score_final = {ROI:0}
        print("ROI is : ", ROI)

        sub_fmri_dir = os.path.join(fmri_dir, sub)
        results_dir = os.path.join(root_path, sub)
        if not os.path.exists(results_dir):
            os.makedirs(results_dir)

        # load the fmri data of this subject.
        train_fmri, test_fmri = get_fmri(sub_fmri_dir, ROI, test_idx)
        num_voxels = test_fmri.shape[-1]
        print("The number of voxels=", num_voxels)
        # different modes.
        if mode == 'hyper_tune':    # This mode is for hyper-parameter tuning.
            fold_num = args['fold_num'] # evaluating by k-fold cross-validation
            kf = KFold(n_splits=fold_num, shuffle=False)
            score_avg = 0.
            for i, (train_idx, val_idx) in enumerate(kf.split(train_activations)):
                log_dir = os.path.join(results_dir, ROI, 'logs', 
                                        '_'.join([mode, hyper_suffix, time_stamp]), str(i+1))
                model_dir = os.path.join(results_dir, ROI, 'models', 
                                        '_'.join([mode, hyper_suffix, time_stamp]), str(i+1))
                if not os.path.exists(model_dir):
                    os.makedirs(model_dir)
                summary = SummaryWriter(logdir=log_dir)
                print('Cross validation, Fold {}'.format(i+1))
                
                train_activ_fold = train_activations[train_idx]
                val_activ_fold = train_activations[val_idx]
                train_fmri_fold = train_fmri[train_idx]
                val_fmri_fold = train_fmri[val_idx]

                reg_model, epoch_saved = predict_fmri_fc2(train_activ_fold, val_activ_fold,
                                            train_fmri_fold, val_fmri_fold,
                                            epochs=epochs, batch_size=batch_size, 
                                            summary=summary, 
                                            w_coe=w_coe, dim_list=dim_list,
                                            layer_coe=layer_coe,
                                            model_dir=model_dir,
                                            metric=args['metric'], patience=args['patience'],
                                            delta=args['delta'])

                pred_fmri = reg_model.forward(
                    torch.from_numpy(val_activ_fold).float().cuda()).detach().cpu().numpy()
                score = vectorized_correlation(val_fmri_fold, pred_fmri)
                print("Mean correlation for ROI %s in %s is: %.3f"%(ROI, sub, score.mean()))
                score_avg += score.mean()
                print("----------------------------------------------------------------------------")
            print('w_coe=%s, layer_coe=%s'%(str(args['w_coe']), str(args['layer_coe'])))
            print("Cross-validation mean correlation for ROI %s in %s is: %.3f"%(
                        ROI, sub, score_avg/fold_num))
            score_final[ROI] = score_avg/fold_num

        elif mode == 'train':
            if w_coe == 0 and layer_coe == 0:
                data = pd.read_csv(csv_save_path)
                # # # # # find the best hyper-params on the first subject # # # # # # # # # # # # 
                target_data = data[(data['mode']=='hyper_tune') & (
                                    data['fold_num'] == 2) & (
                                    data['sub'] == 1)]
                best_score = target_data[ROI].max()
                hyper_suffix = target_data[target_data[ROI]==best_score].iloc[0]['hyper_suffix']
                # # # # # # # # # # # # # # # # # # # # # # # # # # # 
                w_coe = float(hyper_suffix.split('_')[0])
                layer_coe = float(hyper_suffix.split('_')[1])
                print('Using results of hyper-param Grid-Search, w_coe=%.2f, layer_coe=%.2f'%(
                        w_coe, layer_coe))

            log_dir = os.path.join(results_dir, ROI, 'logs', 
                                    '_'.join([mode, hyper_suffix, time_stamp]))
            model_dir = os.path.join(results_dir, ROI, 'models', 
                                    '_'.join([mode, hyper_suffix, time_stamp]))
            if not os.path.exists(model_dir):
                os.makedirs(model_dir)
            summary = SummaryWriter(logdir=log_dir)

            reg_model, epoch_saved = predict_fmri_fc2(train_activations, test_activations,
                                        train_fmri, test_fmri,
                                        epochs=epochs, batch_size=batch_size, 
                                        summary=summary, 
                                        w_coe=w_coe, dim_list=dim_list,
                                        layer_coe=layer_coe,
                                        model_dir=model_dir,
                                        metric=args['metric'], patience=args['patience'],
                                        delta=args['delta'])

            pred_fmri = reg_model.forward(
                torch.from_numpy(test_activations).float().cuda()).detach().cpu().numpy()
            
            # save model for test convienence.
            torch.save(reg_model.state_dict(), os.path.join(model_dir, 'model.pth'))
            score = vectorized_correlation(test_fmri, pred_fmri)
            print("Correlation for ROI : ",ROI, "in ",sub, " is :", round(score.mean(), 3))
            score_final[ROI] = score.mean()

        elif mode == 'test':
            # load model weights and test
            epoch_saved=-1
            model_dir = os.path.join(results_dir, ROI, 'models', 
                                    '_'.join([mode, hyper_suffix, args['time_stamp']]))
            fea_dim = sum(dim_list)
            reg_model = FC_reg(num_voxels, fea_dim, dim_list)
            reg_model.load_state_dict(torch.load(os.path.join(model_dir, 'model.pth')))
            reg_model.cuda()
            reg_model.eval()
            pred_fmri = reg_model.forward(
                torch.from_numpy(test_activations).float().cuda()).detach().cpu().numpy()
            
            score = vectorized_correlation(test_fmri, pred_fmri)
            print("Correlation for ROI : ",ROI, "in ",sub, " is :", round(score.mean(), 3))
            score_final[ROI] = score.mean()

        if mode == 'train' or mode == 'test':
            # calculate the contribution of different layers.
            score_lay_cont_ls = []
            for layer in range(len(dim_list)):
                test_activations_copy = test_activations.copy()
                if layer == 0:
                    test_activations_copy[:, dim_list[layer]:] = 0
                else:
                    test_activations_copy[:, :sum(dim_list[:layer])] = 0
                    test_activations_copy[:, (sum(dim_list[:layer])+dim_list[layer]):] = 0
                
                pred_fmri = reg_model.forward(
                    torch.from_numpy(test_activations_copy).float().cuda()).detach().cpu().numpy()
                score_lay_cont = vectorized_correlation(test_fmri, pred_fmri)
                score_lay_cont_ls.append(score_lay_cont)
            
            tmp_sum = sum([score_lay_cont.mean() for score_lay_cont in score_lay_cont_ls])
            cont_mean_sum = [score_lay_cont.mean()/tmp_sum for score_lay_cont in score_lay_cont_ls]
            layer_contribution.append(cont_mean_sum)

        score_final['mode'] = mode
        score_final['fold_num'] = args['fold_num']
        score_final['sub'] = int(sub[3:])
        score_final['hyper_suffix'] = hyper_suffix
        score_final['epoch_saved'] = epoch_saved
        write_csv(score_final, 
            ['mode', 'fold_num', 'sub', 'hyper_suffix', 'epoch_saved', ROI], 
            csv_save_path)
    if mode == 'train' or mode == 'test':
        # save the layer contribution as np.array of shape 
        # (the number of subjects, the number of candidate layers.)
        np.save(os.path.join(root_path, 'layer_contribution'+'_'+ROI), np.array(layer_contribution))
if __name__ == "__main__":
    main()
