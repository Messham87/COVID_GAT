from __future__ import division
from __future__ import print_function

import os
import glob
import time
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from sklearn.metrics import mean_squared_log_error, mean_absolute_error, mean_squared_error
from utils import load_data
from models import GATMLP, OneLayerGAT, TwoLayerGAT

# Training settings
nocuda = False
fastmode = False
sparse = False
seed = 72
epochs = 1000
lr = 0.005
weight_decay = 5e-3
hidden = 8
nb_heads = 8
dropout = 0.2
alpha = 0.2
patience = 30
nclass = int(1)

cuda = not nocuda and torch.cuda.is_available()

random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
if cuda:
    torch.cuda.manual_seed(seed)

# Load data
adj, train_features, train_labels, valid_features, valid_labels, test_features, test_labels = load_data()

# Model and optimizer
model = TwoLayerGATMLP(nfeat=train_features.shape[1],
                nhid=hidden,
                nclass=nclass,
                dropout=dropout,
                nheads=nb_heads,
                alpha=alpha)
optimizer = optim.Adam(model.parameters(), 
                       lr=lr, 
                       weight_decay=weight_decay)
print(model)

if cuda:
    model.cuda()
    adj = adj.cuda()
    train_features = train_features.cuda()
    valid_features = valid_features.cuda()
    test_features = test_features.cuda()
    train_labels = train_labels.cuda()
    valid_labels = valid_labels.cuda()
    test_labels = test_labels.cuda()

adj, train_features, valid_features, test_features, train_labels, valid_labels, test_labels = Variable(adj), Variable(train_features), Variable(valid_features), Variable(test_features), Variable(train_labels), Variable(valid_labels), Variable(test_labels)
loss = nn.L1Loss()
acc = nn.MSELoss()
def train(epoch):
    t = time.time()
    model.train()
    optimizer.zero_grad()
    train_output = model(train_features, adj)
    train_output
    train_labels.detach().cpu().numpy()
    loss_train = loss(train_output, train_labels)
    acc_train = torch.sqrt(acc(torch.log(train_output+1), torch.log(train_labels+1)))
    loss_train.backward()
    optimizer.step()
    with torch.no_grad():
        model.eval()

        valid_output = model(valid_features, adj)

        loss_val = loss(valid_output, valid_labels)
        acc_val = torch.sqrt(acc(torch.log(valid_output+1), torch.log(valid_labels+1)))
        print('Epoch: {:04d}'.format(epoch+1),
              'loss_train: {:.4f}'.format(loss_train.data.item()),
              'acc_train: {:.4f}'.format(acc_train),
              'loss_val: {:.4f}'.format(loss_val.data.item()),
              'acc_val: {:.4f}'.format(acc_val),
              'time: {:.4f}s'.format(time.time() - t))

    return loss_val.data.item()


def compute_test():
    model.eval()
    test_output = model(test_features, adj)
    loss_test = loss(test_output, test_labels)
    acc_test = torch.sqrt(acc(torch.log(test_output+1), torch.log(test_labels+1)))
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
            open(file, 'w').close() #overwrite and make the file blank instead - ref: https://stackoverflow.com/a/4914288/3553367
            os.remove(file)

files = glob.glob('*.pkl')
for file in files:
    epoch_nb = int(file.split('.')[0])
    if epoch_nb > best_epoch:
        open(file, 'w').close() #overwrite and make the file blank instead - ref: https://stackoverflow.com/a/4914288/3553367
        os.remove(file)

print("Optimization Finished!")
print("Total time elapsed: {:.4f}s".format(time.time() - t_total))

# Restore best model;
print('Loading epoch {}'.format(best_epoch))
model.load_state_dict(torch.load('{}.pkl'.format(best_epoch)))

# Testing
compute_test()