from __future__ import division
from __future__ import print_function

import os
import glob
import time
import random
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from sklearn.metrics import mean_squared_log_error, mean_absolute_error, mean_squared_error
from scipy.stats import pearsonr
from utils import load_data, accuracy
from models import GAT

# Training settings
nocuda = False
fastmode = False
sparse = False
seed = 72
epochs = 10000
lr = 0.0005
weight_decay = 5e-3
hidden = 8
nb_heads = 8
dropout = 0.2
alpha = 0.2
patience = 1000

cuda = not nocuda and torch.cuda.is_available()

random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
if cuda:
    torch.cuda.manual_seed(seed)

# Load data
adj, features, labels, idx_train, idx_val, idx_test = load_data()

# Model and optimizer
model = GAT(nfeat=features.shape[1],
                nhid=hidden,
                nclass=int(4),
                dropout=dropout,
                nheads=nb_heads,
                alpha=alpha)
optimizer = optim.Adam(model.parameters(), 
                       lr=lr, 
                       weight_decay=weight_decay)

if cuda:
    model.cuda()
    features = features.cuda()
    adj = adj.cuda()
    labels = labels.cuda()
    idx_train = idx_train.cuda()
    idx_val = idx_val.cuda()
    idx_test = idx_test.cuda()

features, adj, labels = Variable(features), Variable(adj), Variable(labels)
loss = nn.L1Loss()

def train(epoch):
    t = time.time()
    model.train()
    optimizer.zero_grad()
    output = model(features, adj)
    loss_train = loss(output[idx_train], labels[idx_train])
    acc_train = pearsonr(output[idx_train].detach().numpy(), labels[idx_train].detach().numpy())[0]
    loss_train.backward()
    optimizer.step()

    if not fastmode:
        # Evaluate validation set performance separately,
        # deactivates dropout during validation run.
        model.eval()
        output = model(features, adj)

    loss_val = loss(output[idx_val], labels[idx_val])
    acc_val = pearsonr(output[idx_val].detach().numpy(), labels[idx_val].detach().numpy())[0]
    print('Epoch: {:04d}'.format(epoch+1),
          'loss_train: {:.4f}'.format(loss_train.data.item()),
          'acc_train: {:.4f}'.format(acc_train),
          'loss_val: {:.4f}'.format(loss_val.data.item()),
          'acc_val: {:.4f}'.format(acc_val),
          'time: {:.4f}s'.format(time.time() - t))

    return loss_val.data.item()


def compute_test():
    model.eval()
    output = model(features, adj)
    loss_test = loss(output[idx_test], labels[idx_test])
    acc_test = pearsonr(output[idx_test].detach().numpy(), labels[idx_test].detach().numpy())[0]
    print("Test set results:",
          "loss= {:.4f}".format(loss_test.item()),
          "accuracy= {:.4f}".format(acc_test.item()))
# Train model
t_total = time.time()
loss_values = []
bad_counter = 0
best = epochs + 1
best_epoch = 0
for epoch in range(epochs):
    loss_values.append(train(epoch))
    torch.save(model.state_dict(), '{}.pkl'.format(epoch))
    if loss_values[-1] < best:
        best = loss_values[-1]
        best_epoch = epoch
        bad_counter = 0
    else:
        bad_counter += 1

    if bad_counter == patience:
        break

    files = glob.glob('*.pkl')
    for file in files:
        epoch_nb = int(file.split('.')[0])
        if epoch_nb < best_epoch:
            os.remove(file)

files = glob.glob('*.pkl')
for file in files:
    epoch_nb = int(file.split('.')[0])
    if epoch_nb > best_epoch:
        os.remove(file)

print("Optimization Finished!")
print("Total time elapsed: {:.4f}s".format(time.time() - t_total))

# Restore best model
print('Loading {}th epoch'.format(best_epoch))
model.load_state_dict(torch.load('{}.pkl'.format(best_epoch)))

# Testing
compute_test()

#get the results
# output.max(1)[1]