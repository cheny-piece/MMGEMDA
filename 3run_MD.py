import time
import argparse

import torch
import torch.nn.functional as F
import numpy as np

from utils.pytorchtools import EarlyStopping
from utils.data import load_MD_data2
from utils.tools import index_generator,parse_minibatch_MD
from model.MAGNN_lp import MAGNN_lp
from sklearn.metrics import auc, roc_curve, average_precision_score, precision_recall_curve, accuracy_score, \
    precision_score, recall_score, f1_score
import sklearn.metrics as metrics
import torch.optim.lr_scheduler as lr_scheduler
import matplotlib.pyplot as plt
import os
import warnings

warnings.filterwarnings("ignore")
save_prefix = 'checkpoint_ablation/0none/'
# Params
out_dim = 4
dropout_rate = 0.5
lr = 0.00001
weight_decay = 0.01
etypes_list = [[[0, 1], [2, 3], ],  # 01-0,10-1,12-2,21-3, [0, 2, 3, 1]
               [[1, 0], [4, 5]]]  # , [2, 3]
expected_metapaths = [
    [(0, 1, 0), (0, 2, 0)],  # , (0, 1, 2, 1, 0)
    [(1, 0, 1), (1, 2, 1)]  # , (1, 2, 1)
]
use_masks = [[True] * 2,
             [True] * 2]
no_masks = [[False] * 2, [False] * 2]
prefix = 'data/preprocessed/MD_processed/'


def contrastive_loss(neg_out, pos_out, margin=0.2):
    loss = torch.clamp(margin + neg_out - pos_out, min=0).mean()
    return loss


def run_model_MD( hidden_dim, num_heads, attn_vec_dim, rnn_type,
                 num_epochs, patience, batch_size, neighbor_samples, save_postfix):
    adjlists, edge_metapath_indices_list, features_list, adjM, type_mask, train_val_test_pos_mi_dis, train_val_test_neg_mi_dis = load_MD_data2()
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    in_dims = [features.shape[1] for features in features_list]
    # print(in_dims)
    features_list = [torch.FloatTensor(features).to(device) for features in features_list]
    num_mi = len(adjlists[0][0])
    # print(num_mi)
    fold_auc = []
    fold_ap = []
    fold_acc = []
    fold_prec = []
    fold_rec = []
    fold_f1 = []
    for i in range(1):
        auc_list = []
        ap_list = []
        acc_list = []
        prec_list = []
        rec_list = []
        f1_list = []
        for repeat in range(5):
            print("K fold ",  repeat)
            train_val_test_pos_mi_dis = np.load(prefix + f'fold{repeat}_pos_mi_dis.npz')
            train_val_test_neg_mi_dis = np.load(prefix + f'fold{repeat}_neg_mi_dis.npz')
            train_pos_mi_dis = train_val_test_pos_mi_dis['train_pos_mi_dis']
            test_pos_mi_dis = train_val_test_pos_mi_dis['test_pos_mi_dis']
            train_neg_mi_dis = train_val_test_neg_mi_dis['train_neg_mi_dis']
            test_neg_mi_dis = train_val_test_neg_mi_dis['test_neg_mi_dis']
            y_true_test = np.array([1] * len(test_pos_mi_dis) + [0] * len(test_neg_mi_dis))
            net = MAGNN_lp(
                [2, 2], 6, etypes_list, in_dims, hidden_dim, hidden_dim, num_heads, attn_vec_dim, rnn_type,
                dropout_rate)
            net.to(device)
            optimizer = torch.optim.NAdam(net.parameters(), lr=lr, weight_decay=weight_decay)
            # optimizer = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=weight_decay)
            scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5, verbose=True)
            # training loop
            net.train()
            early_stopping = EarlyStopping(patience=patience, verbose=True,
                                           save_path=save_prefix + 'repeat{}_checkpoint_{}.pt'.format(repeat,
                                                                                                      save_postfix))
            dur1 = []
            dur2 = []
            dur3 = []
            train_pos_idx_generator = index_generator(batch_size=batch_size, num_data=len(train_pos_mi_dis))

            for epoch in range(num_epochs):
                # training
                train_loss_list = []
                net.train()
                for iteration in range(train_pos_idx_generator.num_iterations()):
                    # forward
                    t0 = time.time()

                    train_pos_idx_batch = train_pos_idx_generator.next()
                    train_pos_idx_batch.sort()
                    train_pos_mi_dis_batch = train_pos_mi_dis[train_pos_idx_batch].tolist()
                    train_neg_idx_batch = np.random.choice(len(train_neg_mi_dis), len(train_pos_idx_batch))
                    train_neg_idx_batch.sort()
                    train_neg_mi_dis_batch = train_neg_mi_dis[train_neg_idx_batch].tolist()
                    # print(train_pos_mi_dis_batch,train_neg_mi_dis_batch)
                    train_pos_g_lists, train_pos_indices_lists, train_pos_idx_batch_mapped_lists = parse_minibatch_MD(
                        adjlists, edge_metapath_indices_list, train_pos_mi_dis_batch, device, neighbor_samples,
                        use_masks,
                        num_mi)
                    # print(train_pos_indices_lists,train_pos_idx_batch_mapped_lists)
                    train_neg_g_lists, train_neg_indices_lists, train_neg_idx_batch_mapped_lists = parse_minibatch_MD(
                        adjlists, edge_metapath_indices_list, train_neg_mi_dis_batch, device, neighbor_samples,
                        no_masks,
                        num_mi)

                    t1 = time.time()
                    dur1.append(t1 - t0)


                    pos_out = net(
                        (train_pos_g_lists, features_list, type_mask, train_pos_indices_lists,
                         train_pos_idx_batch_mapped_lists))
                    neg_out = net(
                        (train_neg_g_lists, features_list, type_mask, train_neg_indices_lists,
                         train_neg_idx_batch_mapped_lists))

                    pos_train = np.array([1] * len(pos_out))
                    pos_train = torch.tensor(pos_train).to(device)
                    neg_train = np.array([0] * len(neg_out))
                    neg_train = torch.tensor(neg_train).to(device)
                    train_loss = F.nll_loss(torch.log(pos_out), pos_train) + F.nll_loss(torch.log(neg_out),
                                                                                        neg_train)
                    train_loss_list.append(train_loss.cpu().detach().numpy())
                    t2 = time.time()
                    dur2.append(t2 - t1)

                    # autograd
                    optimizer.zero_grad()
                    train_loss.backward()
                    optimizer.step()

                    t3 = time.time()
                    dur3.append(t3 - t2)

                    # print training info
                    if iteration % 20 == 0:
                        print(
                            'ONone | Epoch {:05d} | Iteration {:05d} | Train_Loss {:.4f} | Time1(s) {:.4f} | Time2(s) {:.4f} | Time3(s) {:.4f}'.format(
                                epoch, iteration, train_loss.item(), np.mean(dur1), np.mean(dur2), np.mean(dur3)))
                scheduler.step(np.mean(train_loss_list))
                early_stopping(np.mean(train_loss_list), net)
                if early_stopping.early_stop:
                    print('Early stopping!')
                    break
                plt.switch_backend('Agg')
                plt.figure()
                plt.plot(train_loss_list, 'b',
                         label='loss')
                plt.ylabel('loss')
                plt.xlabel('epoch')
                plt.savefig(os.path.join(save_prefix + "train/repeat{}_epoch{}_loss.jpg".format(repeat, epoch)))


            test_idx_generator = index_generator(batch_size=8, num_data=len(test_pos_mi_dis), shuffle=False)
            net.load_state_dict(torch.load(save_prefix + 'repeat{}_checkpoint_{}.pt'.format(repeat, save_postfix)))
            net.eval()
            pos_proba_list = []
            neg_proba_list = []
            with torch.no_grad():
                for iteration in range(test_idx_generator.num_iterations()):
                    # forward
                    test_idx_batch = test_idx_generator.next()
                    test_pos_mi_dis_batch = test_pos_mi_dis[test_idx_batch].tolist()
                    test_neg_mi_dis_batch = test_neg_mi_dis[test_idx_batch].tolist()
                    test_pos_g_lists, test_pos_indices_lists, test_pos_idx_batch_mapped_lists = parse_minibatch_MD(
                        adjlists, edge_metapath_indices_list, test_pos_mi_dis_batch, device, neighbor_samples, no_masks,
                        num_mi)
                    test_neg_g_lists, test_neg_indices_lists, test_neg_idx_batch_mapped_lists = parse_minibatch_MD(
                        adjlists, edge_metapath_indices_list, test_neg_mi_dis_batch, device, neighbor_samples, no_masks,
                        num_mi)

                    pos_out = net(
                        (test_pos_g_lists, features_list, type_mask, test_pos_indices_lists,
                         test_pos_idx_batch_mapped_lists))
                    neg_out = net(
                        (test_neg_g_lists, features_list, type_mask, test_neg_indices_lists,
                         test_neg_idx_batch_mapped_lists))
                    pos_proba_list.append(pos_out[:, 1])
                    neg_proba_list.append(neg_out[:, 1])
                y_proba_test = torch.cat(pos_proba_list + neg_proba_list)
                y_proba_test = y_proba_test.cpu().numpy()
                plt.clf()
                fpr, tpr, thresholds = metrics.roc_curve(y_true_test, y_proba_test, pos_label=1)
                auc_score = auc(fpr, tpr)
                plt.plot(fpr, tpr, label='AUC = %0.4f' % auc_score)
                plt.xlabel('False Positive Rate')
                plt.ylabel('True Positive Rate')
                plt.title('ROC Curve')
                plt.legend(loc='lower right')
                plt.savefig(save_prefix + 'test/fold{}_repeat{}_auc_curve.png'.format(i, repeat))

                plt.clf()
                ap_score = average_precision_score(y_true_test, y_proba_test)
                precision, recall, _ = precision_recall_curve(y_true_test, y_proba_test)
                plt.plot(recall, precision, label='AP = %0.4f' % ap_score)
                plt.xlabel('Recall')
                plt.ylabel('Precision')
                plt.title('Precision-Recall Curve')
                plt.legend(loc='lower right')
                plt.savefig(save_prefix + 'test/fold{}_repeat{}_ap_curve.png'.format(i, repeat))
                pred_val = [0 if j < 0.97 else 1 for j in y_proba_test]
                acc = accuracy_score(y_true_test, pred_val)
                prec = precision_score(y_true_test, pred_val)
                rec = recall_score(y_true_test, pred_val)
                f1 = f1_score(y_true_test, pred_val)

                print('Link Prediction Test')
                print('AUC = {}'.format(auc_score))
                print('AP = {}'.format(ap_score))
                print('ACC = {}'.format(acc))
                print('PREC = {}'.format(prec))
                print('REC = {}'.format(rec))
                print('F1 = {}'.format(f1))
                auc_list.append(auc_score)
                ap_list.append(ap_score)
                acc_list.append(acc)
                prec_list.append(prec)
                rec_list.append(rec)
                f1_list.append(f1)
                with open(save_prefix + 'test/fold{}_repeat{}_metrics.txt'.format(i, repeat), 'w') as f:
                    f.write('AUC score: {:.4f}\n'.format(auc_score))
                    f.write('AP score: {:.4f}\n'.format(ap_score))
                    f.write('Accuracy: {:.4f}\n'.format(acc))
                    f.write('Precision: {:.4f}\n'.format(prec))
                    f.write('Recall: {:.4f}\n'.format(rec))
                    f.write('F1 score: {:.4f}\n'.format(f1))
                    if repeat == 4:
                        f.write('AUC_mean = {:.4f}, AUC_std = {:.4f}\n'.format(np.mean(auc_list), np.std(auc_list)))
                        f.write('AP_mean = {:.4f}, AP_std = {:.4f}\n'.format(np.mean(ap_list), np.std(ap_list)))
                        f.write('ACC_mean = {:.4f}, ACC_std = {:.4f}\n'.format(np.mean(acc_list), np.std(acc_list)))
                        f.write('PREC_mean = {:.4f}, PREC_std = {:.4f}\n'.format(np.mean(prec_list), np.std(prec_list)))
                        f.write('REC_mean = {:.4f}, REC_std = {:.4f}\n'.format(np.mean(rec_list), np.std(rec_list)))
                        f.write('F1_mean = {:.4f}, F1_std = {:.4f}\n'.format(np.mean(f1_list), np.std(f1_list)))

            print('----------------------------------------------------------------')
            print('Link Prediction Tests Summary')
            print('AUC_mean = {}, AUC_std = {:.8f}'.format(np.mean(auc_list), np.std(auc_list)))
            print('AP_mean = {}, AP_std = {}'.format(np.mean(ap_list), np.std(ap_list)))
            print('ACC_mean = {}, ACC_std = {}'.format(np.mean(acc_list), np.std(acc_list)))
            print('PREC_mean = {}, PREC_std = {}'.format(np.mean(prec_list), np.std(prec_list)))
            print('REC_mean = {}, REC_std = {}'.format(np.mean(rec_list), np.std(rec_list)))
            print('F1_mean = {}, F1_std = {}'.format(np.mean(f1_list), np.std(f1_list)))
            fold_auc.append(np.mean(auc_list))
            fold_ap.append(np.mean(ap_list))
            fold_acc.append(np.mean(acc_list))
            fold_prec.append(np.mean(prec_list))
            fold_rec.append(np.mean(rec_list))
            fold_f1.append(np.mean(f1_list))
            if i == 4:
                with open(save_prefix + 'test/fold{}_repeat{}_metrics.txt'.format(i, 'w')) as f:
                    f.write('AUC_mean = {:.4f}\n'.format(np.mean(fold_auc)))
                    f.write('AP_mean = {:.4f}\n'.format(np.mean(fold_ap), ))
                    f.write('ACC_mean = {:.4f}\n'.format(np.mean(fold_acc)))
                    f.write('PREC_mean = {:.4f}\n'.format(np.mean(fold_prec)))
                    f.write('REC_mean = {:.4f}\n'.format(np.mean(fold_rec)))
                    f.write('F1_mean = {:.4f}\n'.format(np.mean(fold_f1)))


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description='Set the args for training')
    ap.add_argument('--feats-type', type=int, default=0,
                    help='Type of the node features used. ' +
                         '0 - all id vectors; ' +
                         '1 - all zero vector. Default is 0.')
    ap.add_argument('--hidden-dim', type=int, default=64, help='Dimension of the node hidden state. Default is 64.')
    ap.add_argument('--num-heads', type=int, default=8, help='Number of the attention heads. Default is 8.')
    ap.add_argument('--attn-vec-dim', type=int, default=128, help='Dimension of the attention vector. Default is 128.')
    ap.add_argument('--rnn-type', default='RotatE0', help='Type of the aggregator. Default is RotatE0.')
    ap.add_argument('--epoch', type=int, default=20, help='Number of epochs. Default is 100.')
    ap.add_argument('--patience', type=int, default=30, help='Patience. Default is 5.')
    ap.add_argument('--batch-size', type=int, default=64, help='Batch size. Default is 8.')
    ap.add_argument('--samples', type=int, default=50, help='Number of neighbors sampled. Default is 100.')
    ap.add_argument('--repeat0', type=int, default=5, help='Repeat the training and testing for N times. Default is 1.')
    ap.add_argument('--save-postfix', default='MD', help='Postfix for the saved model and result. Default is MD.')

    args = ap.parse_args()
    run_model_MD( 256, 16, 256, 'RotatE0', args.epoch,
                 args.patience, args.batch_size, 50, args.save_postfix)
