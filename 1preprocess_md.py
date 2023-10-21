import pathlib
import pickle

import numpy as np
import scipy.sparse
import scipy.io
import pandas as pd
import csv

save_prefix = 'data/preprocessed/MD_processed/'


def ReadMyCsv(SaveList, fileName):
    csv_reader = csv.reader(open(fileName))
    for row in csv_reader:  # 注意表头
        SaveList.append(row)
    return


def ReadMyCsv2(SaveList, fileName):
    csv_reader = csv.reader(open(fileName))
    next(csv_reader)
    for row in csv_reader:  # 注意表头
        SaveList.append(row)
    return


save_prefix = 'data/preprocessed/MD_processed/'
num_ntypes = 4
np.random.seed(453286)
mi_data_num = pd.read_csv('data/raw/MD/mi_data_num.csv', sep=',', header=None, names=['mi_id', 'mi_name'],
                          keep_default_na=False, encoding='utf-8')
dis_data_num = pd.read_csv('data/raw/MD/dis_data_num.csv', sep=',', header=None, names=['dis_id', 'dis_name'],
                           keep_default_na=False, encoding='utf-8')
lnc_data_num = pd.read_csv('data/raw/MD/lnc_data_num.csv', sep=',', header=None, names=['lnc_id', 'lnc_name'],
                           keep_default_na=False, encoding='utf-8')
mi_dis = pd.read_csv('data/raw/MD/new_mi_dis.csv', sep=',', header=None, names=['mi_id', 'dis_id'],
                     keep_default_na=False, encoding='utf-8')
mi_lnc = pd.read_csv('data/raw/MD/new_mi_lnc.csv', sep=',', header=None, names=['mi_id', 'lnc_id'],
                     keep_default_na=False, encoding='utf-8')
dis_lnc = pd.read_csv('data/raw/MD/new_dis_lnc.csv', sep=',', header=None, names=['dis_id', 'lnc_id'],
                      keep_default_na=False, encoding='utf-8')


num_mi = len(mi_data_num)
num_dis = len(dis_data_num)
num_lnc = len(lnc_data_num)
print(num_mi, num_dis, num_lnc)

# build the adjacency matrix
# 0 for mi, 1 for dis, 2 for lnc
dim = num_mi + num_dis + num_lnc

type_mask = np.zeros((dim), dtype=int)
type_mask[num_mi:num_mi + num_dis] = 1
type_mask[num_mi + num_dis:] = 2

adjM = np.zeros((dim, dim), dtype=int)
print(dim)
for _, row in mi_dis.iterrows():
    mid = row['mi_id']
    did = num_mi + row['dis_id']
    adjM[mid, did] = 1
    adjM[did, mid] = 1
for _, row in mi_lnc.iterrows():
    mid = row['mi_id']
    lid = num_mi + num_dis + row['lnc_id']
    print(row['lnc_id'])
    adjM[mid, lid] = 1
    adjM[lid, mid] = 1
for _, row in dis_lnc.iterrows():
    did = num_mi + row['dis_id']
    tid = num_mi + num_dis + row['lnc_id']
    adjM[did, tid] = 1
    adjM[tid, did] = 1


dim = num_mi + num_dis + num_lnc
type_mask = np.zeros((dim), dtype=int)
type_mask[num_mi:num_mi + num_dis] = 1
type_mask[num_mi + num_dis:] = 2

mi_dis_list = {i: adjM[i, num_mi:num_mi + num_dis].nonzero()[0] for i in range(num_mi)}
dis_mi_list = {i: adjM[num_mi + i, :num_mi].nonzero()[0] for i in range(num_dis)}
mi_lnc_list = {i: adjM[i, num_mi + num_dis:].nonzero()[0] for i in range(num_mi)}
lnc_mi_list = {i: adjM[num_mi + num_dis + i, :num_mi].nonzero()[0] for i in range(num_lnc)}
dis_lnc_list = {i: adjM[num_mi + i, num_mi + num_dis:].nonzero()[0] for i in range(num_dis)}
lnc_dis_list = {i: adjM[num_mi + num_dis + i, num_mi:num_mi + num_dis].nonzero()[0] for i in range(num_lnc)}

# 0-1-0
m_d_m = []
for d, m_list in dis_mi_list.items():
    m_d_m.extend([(m1, d, m2) for m1 in m_list for m2 in m_list])
m_d_m = np.array(m_d_m)
m_d_m[:, 1] += num_mi
sorted_index = sorted(list(range(len(m_d_m))), key=lambda i: m_d_m[i, [0, 2, 1]].tolist())
m_d_m = m_d_m[sorted_index]
print(m_d_m)
# 0-2-0
m_l_m = []
for l, m_list in lnc_mi_list.items():
    m_l_m.extend([(m1, l, m2) for m1 in m_list for m2 in m_list])
m_l_m = np.array(m_l_m)
m_l_m[:, 1] += num_mi + num_dis
sorted_index = sorted(list(range(len(m_l_m))), key=lambda i: m_l_m[i, [0, 2, 1]].tolist())
m_l_m = m_l_m[sorted_index]
print(m_l_m)
# 1-0-1
d_m_d = []
for m, d_list in mi_dis_list.items():
    d_m_d.extend([(d1, m, d2) for d1 in d_list for d2 in d_list])
d_m_d = np.array(d_m_d)
d_m_d[:, [0, 2]] += num_mi
sorted_index = sorted(list(range(len(d_m_d))), key=lambda i: d_m_d[i, [0, 2, 1]].tolist())
d_m_d = d_m_d[sorted_index]
print(d_m_d)
# 1-2-1
d_l_d = []
for l, d_list in lnc_dis_list.items():
    d_l_d.extend([(d1, l, d2) for d1 in d_list for d2 in d_list])
d_l_d = np.array(d_l_d)
d_l_d[:, [0, 2]] += num_mi
d_l_d[:, 1] += num_mi + num_dis
sorted_index = sorted(list(range(len(d_l_d))), key=lambda i: d_l_d[i, [0, 2, 1]].tolist())
d_l_d = d_l_d[sorted_index]
print(d_l_d)
# 0-1-2-1-0
m_d_l_d_m = []
for d1, l, d2 in d_l_d:
    if len(dis_mi_list[d1 - num_mi]) == 0 or len(dis_mi_list[d2 - num_mi]) == 0:
        continue
    candidate_m1_list = np.random.choice(len(dis_mi_list[d1 - num_mi]), int(0.5 * len(dis_mi_list[d1 - num_mi])),
                                         replace=False)
    candidate_m1_list = dis_mi_list[d1 - num_mi][candidate_m1_list]
    candidate_m2_list = np.random.choice(len(dis_mi_list[d2 - num_mi]), int(0.5 * len(dis_mi_list[d2 - num_mi])),
                                         replace=False)
    candidate_m2_list = dis_mi_list[d2 - num_mi][candidate_m2_list]
    m_d_l_d_m.extend([(m1, d1, l, d2, m2) for m1 in candidate_m1_list for m2 in candidate_m2_list])
m_d_l_d_m = np.array(m_d_l_d_m)
sorted_index = sorted(list(range(len(m_d_l_d_m))), key=lambda i: m_d_l_d_m[i, [0, 4, 1, 2, 3]].tolist())
m_d_l_d_m = m_d_l_d_m[sorted_index]

# 1-0-2-0-1
d_m_l_m_d = []
for m1, l, m2 in m_l_m:
    if len(mi_dis_list[m1]) == 0 or len(mi_dis_list[m2]) == 0:
        continue
    candidate_d1_list = np.random.choice(len(mi_dis_list[m1]), int(0.8 * len(mi_dis_list[m1])), replace=False)
    candidate_d1_list = mi_dis_list[m1][candidate_d1_list] + num_mi
    candidate_d2_list = np.random.choice(len(mi_dis_list[m2]), int(0.8 * len(mi_dis_list[m2])), replace=False)
    candidate_d2_list = mi_dis_list[m2][candidate_d2_list] + num_mi
    d_m_l_m_d.extend([(d1, m1, l, m2, d2) for d1 in candidate_d1_list for d2 in candidate_d2_list])
d_m_l_m_d = np.array(d_m_l_m_d)
print(d_m_l_m_d)
sorted_index = sorted(list(range(len(d_m_l_m_d))), key=lambda i: d_m_l_m_d[i, [0, 4, 1, 2, 3]].tolist())  # ddmlm
d_m_l_m_d = d_m_l_m_d[sorted_index]

expected_metapaths = [
    [(0, 1, 0), (0, 2, 0), (0, 1, 2, 1, 0)],
    [(1, 0, 1), (1, 2, 1), (1, 0, 2, 0, 1)]
]
# create the directories if they do not exist
for i in range(len(expected_metapaths)):
    pathlib.Path(save_prefix + '{}'.format(i)).mkdir(parents=True, exist_ok=True)

metapath_indices_mapping = {(0, 1, 0): m_d_m,
                            (0, 2, 0): m_l_m,
                            (0, 1, 2, 1, 0): m_d_l_d_m,
                            (1, 0, 1): d_m_d,
                            (1, 2, 1): d_l_d,
                            (1, 0, 2, 0, 1): d_m_l_m_d}

# write all things
target_idx_lists = [np.arange(num_mi), np.arange(num_dis)]
offset_list = [0, num_mi]
for i, metapaths in enumerate(expected_metapaths):
    for metapath in metapaths:
        edge_metapath_idx_array = metapath_indices_mapping[metapath]

        with open(save_prefix + '{}/'.format(i) + '-'.join(map(str, metapath)) + '_idx.pickle', 'wb') as out_file:
            target_metapaths_mapping = {}
            left = 0
            right = 0
            for target_idx in target_idx_lists[i]:
                while right < len(edge_metapath_idx_array) and edge_metapath_idx_array[right, 0] == target_idx + \
                        offset_list[i]:
                    right += 1
                target_metapaths_mapping[target_idx] = edge_metapath_idx_array[left:right, ::-1]
                left = right
            pickle.dump(target_metapaths_mapping, out_file)


        with open(save_prefix + '{}/'.format(i) + '-'.join(map(str, metapath)) + '.adjlist', 'w') as out_file:
            left = 0
            right = 0
            for target_idx in target_idx_lists[i]:
                while right < len(edge_metapath_idx_array) and edge_metapath_idx_array[right, 0] == target_idx + \
                        offset_list[i]:
                    right += 1
                neighbors = edge_metapath_idx_array[left:right, -1] - offset_list[i]
                neighbors = list(map(str, neighbors))
                if len(neighbors) > 0:
                    out_file.write('{} '.format(target_idx) + ' '.join(neighbors) + '\n')
                else:
                    out_file.write('{}\n'.format(target_idx))
                left = right

scipy.sparse.save_npz(save_prefix + 'adjM.npz', scipy.sparse.csr_matrix(adjM))
np.save(save_prefix + 'node_types.npy', type_mask)

mi_list = []
ReadMyCsv(mi_list, "data/raw/MD/mi_features.csv")

dis_list = []
ReadMyCsv(dis_list, "data/raw/MD/dis_features.csv")
np.save(save_prefix + 'features_0.npy', mi_list)
np.save(save_prefix + 'features_1.npy', dis_list)
