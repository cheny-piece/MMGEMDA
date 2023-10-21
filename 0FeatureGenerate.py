import pathlib

import numpy as np
import scipy.sparse
import scipy.io
import pandas as pd
import pickle
import networkx as nx
from sklearn.model_selection import train_test_split
import csv
from sklearn.decomposition import PCA,KernelPCA,SparsePCA,FastICA,FactorAnalysis
from sklearn.random_projection import GaussianRandomProjection
from sklearn.manifold import Isomap,TSNE

def ReadMyCsv2(SaveList, fileName):
    csv_reader = csv.reader(open(fileName))
    for row in csv_reader:          # 注意表头
        SaveList.append(row)
    return

def StorFile(data, fileName):
    with open(fileName, "w", newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerows(data)
    return

def decomposition(features,pca,n):
    a=pca(n_components=n)
    a.fit(features)
    newmatrix=a.fit_transform(features)
    mi_matrix=list(newmatrix)
    return mi_matrix
#read the feature of mirnas
mi_list=[]
mi_list2=[]
ReadMyCsv2(mi_list, "data/raw/MD/mi_cs.csv")
ReadMyCsv2(mi_list2, "data/raw/MD/mi_gs.csv")

features_mi=np.array(mi_list)
features_mi=features_mi.astype("float").tolist()
features_mi2=np.array(mi_list2)
features_mi2=features_mi2.astype("float").tolist()

# features_mi=np.concatenate((features_mi,mi_list2),axis=1)
print(len(features_mi))


mi_matrix1=decomposition(features_mi,PCA,64)
mi_matrix2=decomposition(features_mi2,PCA,64)

mi_matrix=np.concatenate((mi_matrix1,mi_matrix2),axis=1)

StorFile(mi_matrix, "data/raw/MD/mi_features.csv")

#read the feature of  diseases
dis_list=[]
dis_list2=[]
ReadMyCsv2(dis_list, "data/raw/MD/d_gs.csv")
ReadMyCsv2(dis_list2, "data/raw/MD/d_ss.csv")
features_dis=np.array(dis_list)
features_dis=features_dis.astype("float").tolist()
features_dis2=np.array(dis_list2)
features_dis2=features_dis2.astype("float").tolist()

print(len(features_dis))


matrix_dis=decomposition(features_dis,PCA,64)
matrix_dis2=decomposition(features_dis2,PCA,64)

matrix_dis=np.concatenate((matrix_dis,matrix_dis2),axis=1)

StorFile(matrix_dis, "data/raw/MD/dis_features.csv")
