from __future__ import print_function

import argparse
import os
import shutil
import time
import random

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data as data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torch.nn.functional as F
import pickle
from utils import *

from tqdm import tqdm
from utils import Bar, Logger, AverageMeter, accuracy, mkdir_p, savefig
from mpl_toolkits.axes_grid1 import ImageGrid

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib as mpl

from scipy.cluster import hierarchy 

def compute_classwise_feature_correlation(feature_path):
    
    # if os.path.isfile(os.path.join(feature_path, 'ft_map_train.pkl')) and os.path.isfile(os.path.join(feature_path, 'ft_map_test.pkl')):
    #     return      
    
    train_feature_path = os.path.join(feature_path, 'train_feature.pkl')
    test_feature_path = os.path.join(feature_path, 'test_feature.pkl')
    
    train_label_path = os.path.join(feature_path, 'train_label.pkl')
    test_label_path = os.path.join(feature_path, 'test_label.pkl')
    
    with open(train_feature_path, 'rb') as f:
        train_feature = pickle.load(f)
    
    with open(test_feature_path, 'rb') as f:
        test_feature = pickle.load(f)
        
    with open(train_label_path, 'rb') as f:
        train_label = pickle.load(f)
    
    with open(test_label_path, 'rb') as f:
        test_label = pickle.load(f)
        
    ft_map_train = {}
    ft_map_test = {}
    num_classes = len(list(set(train_label)))
    
    for ft_map, targets, feature_matrix in [(ft_map_train, train_label, train_feature),(ft_map_test, test_label, test_feature)]:
        feature_mean = torch.mean(feature_matrix, dim=0)
        
        normalized_feature = F.normalize(feature_matrix - feature_mean, dim=1)
        correlation_matrix = normalized_feature @ normalized_feature.T
        targets = torch.Tensor(targets)
        
        feature_map_mean = torch.zeros((num_classes, num_classes))
        feature_map_std = torch.zeros((num_classes, num_classes))
        overall_distribution = torch.zeros((num_classes, num_classes, 20))
    
        for i in range(num_classes):
            for j in range(num_classes):
                print(i,j)
                if i <= j:
                    idx_i = (targets == i).nonzero().reshape(-1)
                    idx_j = (targets == j).nonzero().reshape(-1)
                    corr = correlation_matrix[idx_i][:, idx_j]
                    
                    if i != j:
                        feature_map_mean[i][j] = feature_map_mean[j][i] = torch.mean(corr).item()
                        feature_map_std[i][j] = feature_map_std[j][i] = torch.std(corr).item()
                        overall_distribution[i][j] = overall_distribution[j][i] = torch.histc(corr, bins=20, min=0, max=1).cpu()
                    
                    else:
                        feature_map_mean[i][j] = torch.mean(corr[~torch.eye(corr.shape[0], dtype=bool)]).item()
                        feature_map_std[i][j] = torch.std(corr[~torch.eye(corr.shape[0], dtype=bool)]).item()
                        overall_distribution[i][j] = torch.histc(corr[~torch.eye(corr.shape[0], dtype=bool)], bins=20, min=0, max=1).cpu()
                
        ft_map['mean'] = feature_map_mean
        ft_map['std'] = feature_map_std
        ft_map['hist'] = overall_distribution
            
                
    with open(os.path.join(feature_path, 'ft_map_train.pkl'), 'wb') as f:
        pickle.dump(ft_map_train, f)
        
    with open(os.path.join(feature_path, 'ft_map_test.pkl'), 'wb') as f:
        pickle.dump(ft_map_test, f)


def draw_feature_correlation_heatmap(datasets):
    
    fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(16, 10), layout='compressed')
    
    flat_ax = axes.flat
    for idx, ds in enumerate(datasets):
        feature_path = os.path.join('./feature', ds.lower().replace('-',''))
        with open(os.path.join(feature_path, 'ft_map_train.pkl'), 'rb') as f:
            ft_map = pickle.load(f)
        
        distance_matrix = 1 - ft_map['mean']
        
        Z = hierarchy.linkage(distance_matrix, method='complete')
        
        threshold = np.max(Z[:, 2]) * 0.8
        
        labels = hierarchy.fcluster(Z, threshold, criterion='distance')
        clusters = {}
        
        for i, label in enumerate(labels):
            if label not in clusters:
                clusters[label] = []
            clusters[label].append(i) 
            
        order = []
        for key in clusters:
            order.extend(clusters[key])
        
        ordered_ft_map = ft_map['mean'][order][:, order]
        im = flat_ax[idx].imshow(ordered_ft_map, cmap='viridis', vmin=0, vmax=1)
        flat_ax[idx].set_title(ds, fontsize=20, weight='bold')
        flat_ax[idx].axis('off')
        
    cax, _ = mpl.colorbar.make_axes([ax for ax in flat_ax])
    cb = plt.colorbar(im, cax=cax, shrink=0.5)
    cb.outline.set_linewidth(2)
    cb.ax.tick_params(labelsize=15)
    fig.savefig(os.path.join('./fig', 'ft_correlation.pdf'))
    plt.close(fig) 

def dataset_statistics(datasets):
    
    for idx, ds in enumerate(datasets):
        feature_path = os.path.join('./feature', ds.lower().replace('-',''))
        with open(os.path.join(feature_path, 'ft_map_train.pkl'), 'rb') as f:
            ft_map = pickle.load(f)
        
        distance_matrix = 1 - ft_map['mean']
        
        Z = hierarchy.linkage(distance_matrix, method='complete')
        
        threshold = np.max(Z[:, 2]) * 0.9
        
        labels = hierarchy.fcluster(Z, threshold, criterion='distance')
        clusters = {}
        
        for i, label in enumerate(labels):
            if label not in clusters:
                clusters[label] = []
            clusters[label].append(i)
        
        sum_mean = torch.sum(ft_map['mean']).item()
        sum_std = torch.sum(ft_map['std'] ** 2).item()
        
        c_mean = torch.sum(torch.diag(ft_map['mean'])).item()
        c_std = torch.sum(torch.diag(ft_map['std']) ** 2).item()
        
        d_mean = 0
        d_std = 0
        
        size_c = ft_map['mean'].size(0)
        size_d = 0
        for c in clusters:
            corr = ft_map['mean'][clusters[c]][:, clusters[c]]
            size_d += len(clusters[c]) ** 2
            
            d_mean += torch.sum(corr[~torch.eye(corr.shape[0], dtype=bool)]).item()
            
            corr = ft_map['std'][clusters[c]][:, clusters[c]]
            
            d_std += torch.sum(corr[~torch.eye(corr.shape[0], dtype=bool)] ** 2).item()
        
        size_d -= size_c
        size_e = size_c ** 2 - size_c - size_d
        e_mean = sum_mean - c_mean - d_mean
        e_std = sum_std - c_std - d_std
        
        
        c_mean /= size_c
        c_std /= size_c
        
        d_mean /= size_d
        d_std /= size_d
        
        e_mean /= size_e
        e_std /= size_e
        print(ds + f'\t&\t{len(clusters)}\t&\t{c_mean:.2f}$\\pm${c_std:.2f}\t&\t{d_mean:.2f}$\\pm${d_std:.2f}\t&\t{e_mean:.2f}$\\pm${e_std:.2f} \\\\')

def draw_dendrogram(datasets):
    with open('label_dict.pkl', 'rb') as f:
        label_dict = pickle.load(f)
    
    for fig_idx, sub_datasets in enumerate([datasets[:3], datasets[3:]]):
        fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(40, 25))
        for idx, ds in enumerate(sub_datasets):
            flat_ax = axes.flat
            feature_path = os.path.join('./feature', ds.lower().replace('-',''))
            with open(os.path.join(feature_path, 'ft_map_train.pkl'), 'rb') as f:
                ft_map = pickle.load(f)
            
            distance_matrix = 1 - ft_map['mean']
            
            Z = hierarchy.linkage(distance_matrix, method='complete')
            
            ax=flat_ax[idx]
            dn = hierarchy.dendrogram(Z, ax=ax, labels=[label_dict[ds.replace('-','')][l] for l in range(len(label_dict[ds.replace('-','')]))])
            ax.set_ylabel('Distance', fontsize=25)
            ax.set_title(ds, fontsize=40, weight='bold')
            if ds.replace('-','') == 'StanfordCars' or ds.replace('-','') == 'Caltech256':
                lsize=12
            else:
                lsize=20
                
            ax.tick_params(axis='both', which='major', labelsize=lsize)
            
            ax.set_xticklabels(ax.get_xticklabels(), rotation=-90)
            ax.spines['top'].set_linewidth(2)
            ax.spines['bottom'].set_linewidth(2)
            ax.spines['left'].set_linewidth(2)
            ax.spines['right'].set_linewidth(2)
        
            plt.tight_layout()
        plt.savefig(f'fig/hierarchical_clustering{fig_idx}.pdf')

    
if __name__ == '__main__':
    datasets = ['CIFAR-100', 'Caltech-101', 'Caltech-256', 'Flowers-102', 'Food-101', 'StanfordCars']
    
    # for ds in datasets:
    #     feature_path = os.path.join('./feature', ds.lower().replace('-',''))
    #     compute_classwise_feature_correlation(feature_path)
    
    dataset_statistics(datasets)
    draw_dendrogram(datasets)
    draw_feature_correlation_heatmap(datasets)