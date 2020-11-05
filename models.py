import torch.nn as nn
import torch.nn.functional as F
from layers import GraphConvolution, SGATLayer, SGATMultiLayer # , SGAT1pLayer, SGATMultiLayer


class GCN(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout, node_dropout, edge_dropout):
        super(GCN, self).__init__()

        self.gc1 = GraphConvolution(nfeat, nhid)
        self.gc2 = GraphConvolution(nhid, nclass)
        self.dropout = dropout
        self.node_dropout = node_dropout
        self.edge_dropout = edge_dropout

    def forward(self, x, adj):
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.relu(self.gc1(x, adj, self.node_dropout, self.edge_dropout))
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gc2(x, adj, self.node_dropout, self.edge_dropout)
        return x
    

class GCN1(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout):
        super(GCN1, self).__init__()

        self.gc1 = GraphConvolution(nfeat, nclass)
        self.dropout = dropout

    def forward(self, x, adj):
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gc1(x, adj)
        return x
    
    
class GCN_Linear(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout):
        super(GCN_Linear, self).__init__()

        self.gc1 = GraphConvolution(nfeat, nhid)
        self.linear2 = nn.Linear(nhid, nclass, bias=True)
        self.dropout = dropout

    def forward(self, x, adj):
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.relu(self.gc1(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.linear2(x)
        return x
    
    
class Linear_GCN(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout):
        super(Linear_GCN, self).__init__()

        self.linear1 = nn.Linear(nfeat, nhid, bias=True)
        self.gc2 = GraphConvolution(nhid, nclass)        
        self.dropout = dropout

    def forward(self, x, adj):
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.relu(self.linear1(x))
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gc2(x)
        return x
    
    
class Linear(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout):
        super(Linear, self).__init__()

        self.linear1 = nn.Linear(nfeat, nclass, bias=True)
        self.dropout = dropout

    def forward(self, x, adj=None):
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.linear1(x)
        return x


class Linear2(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout):
        super(Linear2, self).__init__()

        self.linear1 = nn.Linear(nfeat, nhid, bias=True)
        self.linear2 = nn.Linear(nhid, nclass, bias=True)        
        self.dropout = dropout

    def forward(self, x, adj=None):
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.relu(self.linear1(x))
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.linear2(x)
        return x
    
    
class SGAT(nn.Module):
    def __init__(self, nfeat, nhid, nhead, nhead2, nclass, dropout=0.0,
                 node_dropout=0.0, edge_dropout=0.0,
                 pre_attn_order=1, post_attn_order=1, 
                 pre_attn_appnp=False, pre_appnp_alpha=0.1, 
                 post_attn_appnp=False, post_appnp_alpha=0.1, device='cpu'):
        super(SGAT, self).__init__()

        self.layer1 = SGATLayer(nfeat, nhid, nhead, 
                                node_dropout=node_dropout,
                                edge_dropout=edge_dropout, 
                                pre_attn_order=pre_attn_order,
                                post_attn_order=post_attn_order,
                                pre_attn_appnp=pre_attn_appnp,
                                pre_appnp_alpha=pre_appnp_alpha,
                                post_attn_appnp=post_attn_appnp,
                                post_appnp_alpha=post_appnp_alpha,
                                bias=True, mean=False, device=device)
        self.layer2 = SGATLayer(nhid * nhead, nclass, nhead2, 
                                node_dropout=node_dropout,
                                edge_dropout=edge_dropout, 
                                pre_attn_order=pre_attn_order,
                                post_attn_order=post_attn_order,
                                pre_attn_appnp=pre_attn_appnp,
                                pre_appnp_alpha=pre_appnp_alpha,
                                post_attn_appnp=post_attn_appnp,
                                post_appnp_alpha=post_appnp_alpha,
                                bias=False, mean=True, device=device)        
        self.dropout = dropout

    def forward(self, x, adj):
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.elu(self.layer1(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.layer2(x, adj)
        return x
    
    
class SGAT_multi(nn.Module):
    def __init__(self, nfeat, nhid, nhead, nhead2, nbase, nclass, dropout=0.0,
                 node_dropout=0.0, edge_dropout=0.0,
                 pre_attn_order=1, post_attn_order=1, 
                 pre_attn_appnp=False, pre_appnp_alpha=0.1, 
                 post_attn_appnp=False, post_appnp_alpha=0.1, device='cpu'):
        super(SGAT_multi, self).__init__()

        self.layer1 = SGATMultiLayer(nfeat, nhid, nhead, nbase,
                                     node_dropout=node_dropout,
                                     edge_dropout=edge_dropout, 
                                     pre_attn_order=pre_attn_order,
                                     post_attn_order=post_attn_order,
                                     pre_attn_appnp=pre_attn_appnp,
                                     pre_appnp_alpha=pre_appnp_alpha,
                                     post_attn_appnp=post_attn_appnp,
                                     post_appnp_alpha=post_appnp_alpha,
                                     bias=True, mean=False, device=device)
        self.layer2 = SGATMultiLayer(nhid * nhead, nclass, nhead2, nbase, 
                                     node_dropout=node_dropout,
                                     edge_dropout=edge_dropout, 
                                     pre_attn_order=pre_attn_order,
                                     post_attn_order=post_attn_order,
                                     pre_attn_appnp=pre_attn_appnp,
                                     pre_appnp_alpha=pre_appnp_alpha,
                                     post_attn_appnp=post_attn_appnp,
                                     post_appnp_alpha=post_appnp_alpha,
                                     bias=False, mean=True, device=device)        
        self.dropout = dropout

    def forward(self, x, adj):
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.elu(self.layer1(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.layer2(x, adj)
        return x    