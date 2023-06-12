
import time
import numpy as np
import torch
from torch.nn import Dropout, ELU
import torch.nn.functional as F
from torch import nn
import layers

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# GAT Classification model
class SpGATClassification(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout, alpha, nheads):
        """Sparse version of GAT."""
        super(SpGATClassification, self).__init__()
        self.dropout = dropout

        self.attentions = [layers.SpGraphAttentionLayer(nfeat, 
                                                 nhid, 
                                                 dropout=dropout, 
                                                 alpha=alpha, 
                                                 concat=True) for _ in range(nheads)]
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)

        self.out_att = layers.SpGraphAttentionLayer(nhid * nheads, 
                                             nclass, 
                                             dropout=dropout, 
                                             alpha=alpha, 
                                             concat=False)

    def forward(self, x, adj):
        x = F.dropout(x, self.dropout, training=self.training)
        x = torch.cat([att(x, adj) for att in self.attentions], dim=1)
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.elu(self.out_att(x, adj))
        return F.log_softmax(x, dim=1)

# GAT Regression model
class SpGATRegression(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout, alpha, nheads):
        """Sparse version of GAT."""
        super(SpGATRegression, self).__init__()
        self.dropout = dropout

        self.attentions = [layers.SpGraphAttentionLayer(nfeat, 
                                                 nhid, 
                                                 dropout=dropout, 
                                                 alpha=alpha, 
                                                 concat=True) for _ in range(nheads)]
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)

        self.out_att = layers.SpGraphAttentionLayer(nhid * nheads, 
                                             nclass, 
                                             dropout=dropout, 
                                             alpha=alpha, 
                                             concat=False)

    def forward(self, x, adj):
        x = F.dropout(x, self.dropout, training=self.training)
        x = torch.cat([att(x, adj) for att in self.attentions], dim=1)
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.elu(self.out_att(x, adj))
        return x
    
class GCN_classification(nn.Module):
    def __init__(self, nfeat, nhid,  nclass, dropout):
        super(GCN_classification, self).__init__()

        self.gc1 = layers.GraphConvolution(nfeat, nhid)  
        self.gc2 = layers.GraphConvolution(nhid, nclass) 
        self.dropout = dropout

    def forward(self, x, adj):
        
        x = x.to(device)
        adj = adj.to(device)

        x = F.relu(self.gc1(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gc2(x, adj)
        
        return F.softmax(x, dim=1)

class GCN_Regression(nn.Module):
    def __init__(self, nfeat, nhid,  nclass, dropout):
        super(GCN_Regression, self).__init__()

        self.gc1 = layers.GraphConvolution(nfeat, nhid)
        self.gc2 = layers.GraphConvolution(nhid, nclass)  
        self.dropout = dropout

    def forward(self, x, adj):
        
        x = x.to(device)
        adj = adj.to(device)

        x = F.relu(self.gc1(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gc2(x, adj)
        return x   