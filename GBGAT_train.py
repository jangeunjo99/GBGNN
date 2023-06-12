import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import scipy.sparse as sp
import time
import argparse
import numpy as np
import pandas as pd
import easydict

# Training settings
args = easydict.EasyDict({
    "no_cuda": False,
    "fastmode": False,
    "sparse": False,
    "seed": 1234,
    "epoch": 100,
    "lr": 0.01,
    "weight_decay": 5e-4,
    "hidden": 64,  
    "dropout": 0.1,
    # "batch_size":512,
    "alpha":0.2,
    "nb_heads":8
})
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)

args.cuda = not args.no_cuda and torch.cuda.is_available()

adj, features, labels, idx_train,idx_test, one_hot_labels   = load_data()



def train(d_epoch, d_model,d_labels,classification = True):  
 
    d_model.train()
    optimizer.zero_grad()
    output = d_model(features.to(device), adj.to(device))
    
    if classification:
        loss_train = F.cross_entropy(output[idx_train], d_labels[idx_train])
        acc_train = accuracy(output[idx_train], d_labels[idx_train])
        loss_train.backward()
        optimizer.step()

    else :
        loss_train = torch.sqrt(F.mse_loss(output[idx_train], d_labels[idx_train]))
        mse_train = MSELOSS(output[idx_train], d_labels[idx_train])
        loss_train.backward()
        optimizer.step()


def test(d_model, d_labels,classification = True):
    d_model.eval()
    output = d_model(features.to(device), adj.to(device))
    if classification:
        loss_test = F.cross_entropy(output[idx_test], d_labels[idx_test])
        acc_test = accuracy(output[idx_test], d_labels[idx_test])
        print("Test set results:",
              "loss= {:.4f}".format(loss_test.item()),
              "accuracy= {:.4f}".format(acc_test.item()))
    else:
        loss_test = torch.sqrt(F.mse_loss(output[idx_test], d_labels[idx_test]))
        mse_test = MSELOSS(output[idx_test], d_labels[idx_test])
        print("Test set results:",
              "loss= {:.4f}".format(loss_test.item()),
              "mse= {:.4f}".format(mse_test.item()))
        
MSELOSS = nn.MSELoss()

# Model and optimizer 

model0 = SpGATClassification(nfeat=features.shape[1], 
                nhid=args.hidden, 
                nclass=int(labels.max()) + 1, 
                dropout=args.dropout, 
                nheads=args.nb_heads, 
                alpha=args.alpha)

optimizer = optim.Adam(model0.parameters(), 
                       lr=args.lr, 
                       weight_decay=args.weight_decay)

if args.cuda:
    model.cuda()
    features = features.cuda()
    adj = adj.cuda()
    labels = labels.cuda()
    idx_train = idx_train.cuda()
    idx_test = idx_test.cuda()

features, adj, labels = Variable(features), Variable(adj), Variable(labels)

# Train model 
for epoch in range(args.epoch):
    train(d_epoch = epoch, d_model = model.to(device) ,d_labels = labels.to(device) ,classification = True)
    
one_hot_labels = torch.tensor(one_hot_labels, dtype=torch.float32).to(device)

iterations = 50
test_acc_list = []

for i in tqdm(range(iterations)):
    res = one_hot_labels - output.clone().detach().to(device)
    
    # Model and optimizer
    model = SpGATRegression(nfeat=features.shape[1], 
                    nhid=args.hidden, 
                    nclass=int(labels.max()) + 1, 
                    dropout=args.dropout, 
                    nheads=args.nb_heads, 
                    alpha=args.alpha).to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    
    start_time = time.time()  
    
    for epoch in range(args.epoch):
        train(d_epoch = epoch, d_model = model.to(device) ,d_labels = res.to(device),classification = False)  # 
        
    model.eval()
    #val_loss = 0
    
    with torch.no_grad():
        res_hat = model(features.to(device), adj.to(device))
        output = output.to(device) + 0.05* res_hat
    
        # acc_val = accuracy(output[idx_val], labels[idx_val]) 
    
    elapsed_time = time.time() - start_time  
    print(f"Iteration {i+1}: Elapsed time: {elapsed_time:.2f} seconds")

    test_acc = accuracy(output[idx_test], labels[idx_test])
    test_acc_list.append(test_acc)
    
acc_test_df = pd.DataFrame(torch.tensor(test_acc_list).cpu().numpy())
acc_test_df