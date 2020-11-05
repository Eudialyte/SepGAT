from __future__ import division
from __future__ import print_function

import time
import argparse
import numpy as np
import os

import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

from utils import load_data, accuracy, f1
from models import GCN, GCN1, GCN_Linear, Linear_GCN, Linear, Linear2, SGAT, SGAT_multi # , SGAT1p, SGAT_multi

# Training settings
parser = argparse.ArgumentParser()
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='Disables CUDA training.')
parser.add_argument('--fastmode', action='store_true', default=False,
                    help='Validate during training pass.')
parser.add_argument('--seed', type=int, default=0, help='Random seed.')
parser.add_argument('--epochs', type=int, default=10000,
                    help='Number of epochs to train.')
parser.add_argument('--early_stopping', type=int, default=None,
                    help='Number of selected nonzero eigenvectors')
parser.add_argument('--lr', type=float, default=0.005,
                    help='Initial learning rate.')
parser.add_argument('--weight_decay', type=float, default=5e-4,
                    help='Weight decay, not implemented for lbfgs.')
parser.add_argument('--hidden', type=int, default=8,
                    help='Number of hidden units per head.')
parser.add_argument('--head', type=int, default=8,
                    help='Number of attention heads in SepGAT and SimpGAT.')
parser.add_argument('--head2', type=int, default=1,
                    help='Number of attention heads on layer 2.')
parser.add_argument('--bases', type=int, default=8,
                    help='Number of base functions in SepGAT.')
parser.add_argument('--dropout', type=float, default=0.6,
                    help='Dropout rate (1 - keep probability).')
parser.add_argument('--node_dropout', type=float, default=0.6,
                    help='Node dropout rate for SGAT (1 - keep probability).')
parser.add_argument('--edge_dropout', type=float, default=0.6,
                    help='Edge dropout rate for SGAT (1 - keep probability).')
parser.add_argument('--attn_clip', type=float, default=15.,
                    help='Clip on exponents in attention (Unused).')
parser.add_argument('--grad_clip', type=float, default=None,
                    help='Gradient clip based on value.')
parser.add_argument('--repeat', type=int, default=1,
                    help='Number of repeated runs.')
parser.add_argument('--dataset', type=str, default='cora',
                    help='Dataset ("cora", "citeseer", "pubmed", "reddit").')
parser.add_argument('--label_rate', type=float, default=None,
                    help='Label rate, None for original split.')
parser.add_argument('--save', action='store_true', default=False,
                    help='Whether to save model for early-stopping.')
parser.add_argument('--model', type=str, default='SepGAT',
                    help='Model ("SepGAT", "SimpGAT", "GCN", "SGC").')
parser.add_argument('--pre_attn_order', type=int, default=1,
                    help='Orders of adjacency matrix.')
parser.add_argument('--post_attn_order', type=int, default=1,
                    help='Orders of adjacency matrix.')
parser.add_argument('--pre_attn_appnp', action='store_true', default=False,
                    help='Orders of adjacency matrix.')
parser.add_argument('--pre_appnp_alpha', type=float, default=0.1,
                    help='Orders of adjacency matrix.')
parser.add_argument('--post_attn_appnp', action='store_true', default=False,
                    help='Orders of adjacency matrix.')
parser.add_argument('--post_appnp_alpha', type=float, default=0.1,
                    help='Orders of adjacency matrix.')
parser.add_argument('--poly', action='store_true', default=False,
                    help='Model ("Whether to use polynomial SepGAT").')
parser.add_argument('--order', type=int, default=2,
                    help='Orders of adjacency matrix for SGC.')
parser.add_argument('--optimizer', type=str, default='adam',
                    help='Optimizer. ("adam", "lbfgs (for SGC)")')
parser.add_argument('--loss_threshold', type=float, default=2.5,
                    help='Loss threshold to avoid early-stopping at beginning.')
parser.add_argument('--print', type=int, default=1,
                    help='Print loss and accuracy every n epochs.')
parser.add_argument('--criteria', type=str, default='loss',
                    help='Criteria for early-stopping. ("loss", "acc")')
parser.add_argument('--transductive', action='store_true', default=False,
                    help='Apply transductive learning on Reddit dataset.')

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()
device = 'cuda' if args.cuda else 'cpu'
if args.model == 'SepGAT':
    args.model = 'SGAT_multi'
elif args.model == 'SimpGAT':
    args.model = 'SGAT'

run_id = hex(hash((str(args), time.time())))

np.random.seed(args.seed)
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

inductive = (args.dataset == 'reddit' and not args.transductive)
loop = not args.pre_attn_appnp
if args.poly:
    norm = 'max'
elif args.model != 'SGAT' and args.model != 'SGAT_multi':
    norm = 'sym'
else:
    norm = None

# Train model
acc_list = []
valacc_list = []
mean_time_list = []
total_time_list = []

# repeat multiple runs
for i in range(args.repeat):
    t_total = time.time()
        
    # Load data
    adj_train, adj_val, adj_test, features, labels, idx_train, idx_val, \
        idx_test = load_data(args.dataset, args.label_rate, inductive, loop, norm)
    
    # Model and optimizer
    if args.model == 'GCN':
        model = GCN(nfeat=features.shape[1],
                    nhid=args.hidden,
                    nclass=labels.max().item() + 1,
                    dropout=args.dropout,
                    node_dropout=args.node_dropout,
                    edge_dropout=args.edge_dropout)
    elif args.model == 'GCN1':
        model = GCN1(nfeat=features.shape[1],
                     nhid=args.hidden,
                     nclass=labels.max().item() + 1,
                     dropout=args.dropout,
                     edge_dropout=args.edge_dropout)
    elif args.model == 'GCN_Linear':
        model = GCN_Linear(nfeat=features.shape[1],
                           nhid=args.hidden,
                           nclass=labels.max().item() + 1,
                           dropout=args.dropout)
    elif args.model == 'Linear_GCN':
        model = Linear_GCN(nfeat=features.shape[1],
                           nhid=args.hidden,
                           nclass=labels.max().item() + 1,
                           dropout=args.dropout)
    elif args.model == 'Linear' or args.model == 'SGC':
        model = Linear(nfeat=features.shape[1],
                       nhid=args.hidden,
                       nclass=labels.max().item() + 1,
                       dropout=args.dropout)
    elif args.model == 'Linear2' or args.model == 'SGC2':
        model = Linear2(nfeat=features.shape[1],
                        nhid=args.hidden,
                        nclass=labels.max().item() + 1,
                        dropout=args.dropout)
    elif args.model == 'SGAT':
        model = SGAT(nfeat=features.shape[1],
                     nhid=args.hidden,
                     nhead=args.head,
                     nhead2=args.head2,
                     nclass=labels.max().item() + 1,
                     dropout=args.dropout,
                     node_dropout=args.node_dropout,
                     edge_dropout=args.edge_dropout,
                     pre_attn_order=args.pre_attn_order,
                     post_attn_order=args.post_attn_order,
                     pre_attn_appnp=args.pre_attn_appnp,
                     pre_appnp_alpha=args.pre_appnp_alpha,
                     post_attn_appnp=args.post_attn_appnp,
                     post_appnp_alpha=args.post_appnp_alpha,
                     device=device)
    elif args.model == 'SGAT_multi':
        model = SGAT_multi(nfeat=features.shape[1],
                           nhid=args.hidden,
                           nhead=args.head,
                           nhead2=args.head2,
                           nbase=args.bases,
                           nclass=labels.max().item() + 1,
                           dropout=args.dropout,
                           node_dropout=args.node_dropout,
                           edge_dropout=args.edge_dropout,
                           pre_attn_order=args.pre_attn_order,
                           post_attn_order=args.post_attn_order,
                           pre_attn_appnp=args.pre_attn_appnp,
                           pre_appnp_alpha=args.pre_appnp_alpha,
                           post_attn_appnp=args.post_attn_appnp,
                           post_appnp_alpha=args.post_appnp_alpha,
                           device=device)        
    else:
        model = Linear(nfeat=features.shape[1],
                       nhid=args.hidden,
                       nclass=labels.max().item() + 1,
                       dropout=args.dropout)        
    
    if args.optimizer == 'adam':
        optimizer = optim.Adam(model.parameters(),
                               lr=args.lr, weight_decay=args.weight_decay)
    elif args.optimizer == 'lbfgs':
        optimizer = optim.LBFGS(model.parameters())       
    
    if args.cuda:
        model.cuda()
        features = features.cuda()
        adj_test = adj_test.cuda()
        labels = labels.cuda()
        idx_train = idx_train.cuda()
        idx_val = idx_val.cuda()
        idx_test = idx_test.cuda()
        if inductive:
            pass
            adj_train = adj_train.cuda()
            adj_val = adj_val.cuda()
        else:
            adj_train = adj_test
            adj_val = adj_test
    elif not inductive:
        adj_train = adj_test
        adj_val = adj_test        
            
    if args.model == 'SGC' or args.model == 'SGC2':
        for _ in range(args.order):
            features = torch.spmm(adj_test, features)
    
    # features, labels = Variable(features), Variable(labels)    
    
    def train(epoch):
        
        def closure():
            optimizer.zero_grad()
            output = model(features, adj_train)
            loss_train = F.cross_entropy(output[idx_train], labels[idx_train])
            loss_train.backward()
        
        t = time.time()
        model.train()
        optimizer.zero_grad()
        output = model(features, adj_train)
        loss_train = F.cross_entropy(output[idx_train], labels[idx_train])
        if args.dataset == 'reddit':
            acc_train = f1(output[idx_train], labels[idx_train])
        else:  
            acc_train = accuracy(output[idx_train], labels[idx_train])
        if args.optimizer == 'lbfgs':
            optimizer.step(closure)
        else:
            loss_train.backward()
            if args.grad_clip:
                torch.nn.utils.clip_grad_value_(model.parameters(), args.grad_clip)
            optimizer.step()
    
        if not args.fastmode:
            # Evaluate validation set performance separately,
            # deactivates dropout during validation run.
            model.eval()
            output = model(features, adj_val)
    
        loss_val = F.cross_entropy(output[idx_val], labels[idx_val])
        if args.dataset == 'reddit':
            acc_val = f1(output[idx_val], labels[idx_val])
        else:
            acc_val = accuracy(output[idx_val], labels[idx_val])
        epoch_time = time.time() - t
        if args.print and epoch % args.print == args.print - 1:
            print('Epoch: {:04d}'.format(epoch+1),
                  'loss_train: {:.4f}'.format(loss_train.item()),
                  'acc_train: {:.4f}'.format(acc_train.item()),
                  'loss_val: {:.4f}'.format(loss_val.item()),
                  'acc_val: {:.4f}'.format(acc_val.item()),
                  'time: {:.4f}s'.format(time.time() - t))
        return loss_val.item(), acc_val.item(), epoch_time
    
    
    def test():
        model.eval()
        output = model(features, adj_test)
        loss_test = F.cross_entropy(output[idx_test], labels[idx_test])
        if args.dataset == 'reddit':
            acc_test = f1(output[idx_test], labels[idx_test])
        else:
            acc_test = accuracy(output[idx_test], labels[idx_test])
        print("Test set results:",
              "loss= {:.4f}".format(loss_test.item()),
              "accuracy= {:.4f}".format(acc_test.item()))
        return acc_test.item()
    
    
    acc_val = None
    criterias = []
    epoch_time_list = []
    bad_counter = 0
    best = np.inf
    best_epoch = 0
    for epoch in range(args.epochs):
        loss_val, acc_val, epoch_time = train(epoch)
        epoch_time_list.append(epoch_time)
        criteria = -acc_val if args.criteria == 'acc' else loss_val
        criterias.append(criteria)
        
        if args.early_stopping:
            if criterias[-1] < best:
                best = criterias[-1]
                best_epoch = epoch
                bad_counter = 0
                if args.save:
                    torch.save(model.state_dict(), '{}.pkl'.format(run_id))
            else:
                bad_counter += 1
        
            if bad_counter >= args.early_stopping and loss_val < args.loss_threshold:
                print("Early stopping...")
                break
        
    print("Optimization Finished!")
    total_time = time.time() - t_total
    mean_time = np.mean(epoch_time_list)
    print("Total time elapsed: {:.4f}s".format(total_time))
    print("Time per epoch: {:.4f}s".format(mean_time))
    
    if args.early_stopping and args.save:
        print('Loading {}th epoch'.format(best_epoch))
        model.load_state_dict(torch.load('{}.pkl'.format(run_id)))
    
    # Testing
    acc = test()
    valacc_list.append(acc_val)
    acc_list.append(acc)
    total_time_list.append(total_time)
    mean_time_list.append(mean_time)
avgvalacc = np.mean(valacc_list)
avgacc = np.mean(acc_list)
avg_total_time = np.mean(total_time_list)
avg_mean_time = np.mean(mean_time_list)
stdvalacc = np.std(valacc_list)
stdacc = np.std(acc_list)

print("mean validation accuracy =  {:.4f}".format(avgvalacc),
      "std of validation accuracy =  {:.4f}".format(stdvalacc),
      "mean accuracy =  {:.4f}".format(avgacc), 
      "std of accuracy =  {:.4f}".format(stdacc),)
print("mean total time =  {:.4f}s".format(avg_total_time),
      "mean epoch time =  {:.4f}s".format(avg_mean_time),)
print()
if args.early_stopping and args.save:
    os.remove('{}.pkl'.format(run_id))