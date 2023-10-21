import pathlib

import numpy as np
import scipy.sparse
import scipy.io
import pandas as pd
import pickle
import networkx as nx
from utils import preprocess
from sklearn.model_selection import train_test_split, KFold
import csv
import torch
from sklearn.decomposition import PCA, KernelPCA, SparsePCA
import os

# output positive and negative samples for training, validation and testing

save_prefix = 'data/preprocessed/MD_processed/'

mi_dis = pd.read_csv('data/raw/MD/new_mi_dis.csv', sep=',', header=None, names=['mi_id', 'dis_id'],
                     keep_default_na=False, encoding='utf-8')
mi_dis = np.array(mi_dis)

mi_data_num = pd.read_csv('data/raw/MD/mi_data_num.csv', sep=',', header=None, names=['mi_id', 'mi_name'],
                          keep_default_na=False, encoding='utf-8')
mi_data_num = np.array(mi_data_num)

dis_data_num = pd.read_csv('data/raw/MD/dis_data_num.csv', sep=',', header=None, names=['dis_id', 'dis_name'],
                           keep_default_na=False, encoding='utf-8')
dis_data_num = np.array(dis_data_num)

rand_seed = 453289

kf = KFold(n_splits=5, shuffle=True, random_state=rand_seed)
neg_edges = []
counter = 0
print(len(mi_data_num),len(dis_data_num),len(mi_dis))
for i in range(len(mi_data_num)):
    for j in range(len(dis_data_num)):
        if counter < len(mi_dis):
            if i == mi_dis[counter, 0] and j == mi_dis[counter, 1]:
                counter += 1
            else:
                neg_edges.append([i, j])
        else:
            neg_edges.append([i, j])
neg_edges = np.array(neg_edges)
for fold_idx, (train_idx, test_idx) in enumerate(kf.split(np.arange(len(mi_dis)))):

    print(f"Fold {fold_idx}")


    test_idx = test_idx
    print(len(train_idx),len(test_idx))
    train_idx.sort()
    test_idx.sort()

    # Extract positive edges for train and test sets
    train_pos_edges = mi_dis[train_idx]
    test_pos_edges=mi_dis[test_idx]
    print(len(train_pos_edges), len(test_pos_edges))

    # Extract negative edges for train, validation and test sets
    train_neg_edges = []
    test_neg_edges=[]
    counter = 0
    for i in range(len(mi_data_num)):
        for j in range(len(dis_data_num)):
            if counter < len(train_pos_edges):
                if i == train_pos_edges[counter, 0] and j == train_pos_edges[counter, 1]:
                    counter += 1
                else:
                    train_neg_edges.append([i, j])
            else:
                train_neg_edges.append([i, j])
    train_neg_edges = np.array(train_neg_edges)
    print(len(train_neg_edges))
    idx = np.random.choice(len(neg_edges), len(test_idx), replace=False)
    test_neg_edges = neg_edges[sorted(idx)]
    counter = 0

    print(
        f"Number of negative edges: Train {len(train_neg_edges)} Test {len(test_neg_edges)}")

    # Save train, validation
    print(
        f"Number of positive edges: Train {len(train_pos_edges)}, Test {len(test_pos_edges)}")
    np.savez(save_prefix + f'fold{fold_idx}_pos_mi_dis.npz',
             train_pos_mi_dis=train_pos_edges,
             test_pos_mi_dis=test_pos_edges)
    np.savez(save_prefix + f'fold{fold_idx}_neg_mi_dis.npz',
             train_neg_mi_dis=train_neg_edges,
             test_neg_mi_dis=test_neg_edges)