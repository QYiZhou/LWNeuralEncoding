'''
Author: zhouqy
Date: 2022-06-04 11:40:49
LastEditors: zhouqy
LastEditTime: 2022-07-11 23:40:46
Description:  Collate the encoding results and save them in a .csv file.
'''

import os
import pandas as pd
import argparse
import collections
import csv

def write_csv(root_path, csv_prefix, ROIs):
    """write results of different ROIs into one .csv file.

    Args:
        root_path (str): root path of the existing .csv files fo different ROIs.
        csv_prefix (str): prefix of the existing .csv files of different ROIs.
        ROIs (list): a list including different ROIs.
    """
    res_avg = collections.defaultdict()
    res_std = collections.defaultdict()
    for roi in ROIs:
        csv_save_path = os.path.join(root_path, (csv_prefix+roi)+'.csv')
        data = pd.read_csv(csv_save_path)
        target_data = data[(data['mode']=='train') & (data['fold_num']==2)][-10:]
        res_avg[roi] = target_data[roi].mean()
        res_std[roi] = target_data[roi].std()

    csv_save_path = os.path.join(root_path, 'avg_res.csv')
    
    if not os.path.isfile(csv_save_path):
        with open(csv_save_path, "w") as csv_file:
            writer = csv.DictWriter(
                csv_file, fieldnames=ROIs)
            writer.writeheader()
    with open(csv_save_path, "a") as csv_file:
        writer = csv.DictWriter(
            csv_file, fieldnames=ROIs)
        writer.writerow(res_avg)
        writer.writerow(res_std)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='read summary results and calculate averages of all subjects')
    parser.add_argument('-model','--model',help='model name under which predicted fMRI activity will be saved', 
                            default = 'alexnet', type=str)
    parser.add_argument('-cp', '--csv_prefix',help='prefix of existing csv files', 
                            default = 'summary_pcc_', type=str)
    parser.add_argument('-rd', '--result_dir',help='saves predicted fMRI activity', type=str)
    parser.add_argument('-rp', '--root_path',help='root path', type=str)
    parser.add_argument('-dim_rd', '--dim_rd', 
                        help='dimension reduction: not_rd (no reduction), or srp', 
                        default='srp')
    args = vars(parser.parse_args())

    root_path = os.path.join(args['root_path'], args['result_dir'], 
                    args['model'], args['dim_rd'], 'multilayer/mini_track')
    ROIs = ['V1','V2','V3','V4', 'LOC','EBA','FFA','STS','PPA']
    write_csv(root_path, args['csv_prefix'], ROIs)

