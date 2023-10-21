import networkx as nx
import numpy as np
import scipy
import pickle


def add_noise(x, noise_scale=0.25):
    return x + noise_scale * np.random.randn(*x.shape)

def load_MD_data(prefix='data/preprocessed/MD_processed'):
    in_file = open(prefix + '/0/0-1-0.adjlist', 'r')
    adjlist00 = [line.strip() for line in in_file]
    adjlist00 = adjlist00
    in_file.close()


    in_file = open(prefix + '/1/1-0-1.adjlist', 'r')
    adjlist10 = [line.strip() for line in in_file]
    adjlist10 = adjlist10
    in_file.close()


    in_file = open(prefix + '/0/0-1-0_idx.pickle', 'rb')
    idx00 = pickle.load(in_file)
    in_file.close()

    in_file = open(prefix + '/1/1-0-1_idx.pickle', 'rb')
    idx10 = pickle.load(in_file)
    in_file.close()


    features_0 = np.load(prefix + '/features_0.npy')
    features_0 =np.array(features_0,dtype=np.float_)
    features_0 =np.array([add_noise(x) for x in features_0])
    features_1 = np.load(prefix + '/features_1.npy')
    features_1 =np.array(features_1,dtype=np.float_)
    features_1 =np.array([add_noise(x) for x in features_1])
    features_2= np.eye(2801,dtype=np.float32)

    adjM = scipy.sparse.load_npz(prefix + '/adjM.npz')
    type_mask = np.load(prefix + '/node_types.npy')


    return [[adjlist00],[adjlist10]],\
           [[idx00],[idx10]],\
           [features_0, features_1,features_2], \
           adjM, \
           type_mask


def load_MD_data2(prefix='data/preprocessed/MD_processed'):
    in_file = open(prefix + '/0/0-1-0.adjlist', 'r')
    adjlist00 = [line.strip() for line in in_file]
    adjlist00 = adjlist00
    in_file.close()

    in_file = open(prefix + '/0/0-2-0.adjlist', 'r')
    adjlist01 = [line.strip() for line in in_file]
    adjlist01 = adjlist01
    in_file.close()

    in_file = open(prefix + '/1/1-0-1.adjlist', 'r')
    adjlist10 = [line.strip() for line in in_file]
    adjlist10 = adjlist10
    in_file.close()
    in_file = open(prefix + '/1/1-2-1.adjlist', 'r')
    adjlist11 = [line.strip() for line in in_file]
    adjlist11 = adjlist11
    in_file.close()

    in_file = open(prefix + '/0/0-1-0_idx.pickle', 'rb')
    idx00 = pickle.load(in_file)
    in_file.close()
    in_file = open(prefix + '/0/0-2-0_idx.pickle', 'rb')
    idx01 = pickle.load(in_file)
    in_file.close()
    in_file = open(prefix + '/1/1-0-1_idx.pickle', 'rb')
    idx10 = pickle.load(in_file)
    in_file.close()
    in_file = open(prefix + '/1/1-2-1_idx.pickle', 'rb')
    idx11 = pickle.load(in_file)
    in_file.close()


    features_0 = np.load(prefix + '/features_0.npy')
    features_0 =np.array(features_0,dtype=np.float_)
    features_0 =np.array([add_noise(x) for x in features_0])
    features_1 = np.load(prefix + '/features_1.npy')
    features_1 =np.array(features_1,dtype=np.float_)
    features_1 =np.array([add_noise(x) for x in features_1])
    features_2= np.eye(2801,dtype=np.float32)

    adjM = scipy.sparse.load_npz(prefix + '/adjM.npz')
    type_mask = np.load(prefix + '/node_types.npy')
    train_val_test_pos_mi_dis = np.load(prefix + '/train_val_test_pos_mi_dis.npz')
    train_val_test_neg_mi_dis = np.load(prefix + '/train_val_test_neg_mi_dis.npz')

    return [[adjlist00,adjlist01],[adjlist10,adjlist11]],\
           [[idx00,idx01],[idx10,idx11]],\
           [features_0, features_1,features_2], \
           adjM, \
           type_mask,\
           train_val_test_pos_mi_dis,\
           train_val_test_neg_mi_dis



