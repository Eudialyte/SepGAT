import math

import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module


class sparse_dropout(Module):
    """
    Sparse dropout implementation
    """
    
    def __init__(self):
        super(sparse_dropout, self).__init__()
        
    def forward(self, input, p=0.5):
        if self.training == True and p > 0.:
            random = torch.rand_like(input._values())
            mask = random.ge(p)
            new_indices = torch.masked_select(input._indices(), mask).reshape(2, -1)
            new_values = torch.masked_select(input._values(), mask)
            output = torch.sparse.FloatTensor(new_indices, new_values, input.shape)
            return output
        else:
            return input


class GraphConvolution(Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, node_dropout=0.0, edge_dropout=0.0, bias=True):
        super(GraphConvolution, self).__init__()
        self.sparse_dropout = sparse_dropout()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.zeros(in_features, out_features))
        stdv = 1. / math.sqrt(self.weight.size(1))
        nn.init.uniform_(self.weight, -stdv, stdv)
        if bias:
            self.bias = Parameter(torch.zeros(1, out_features))
            nn.init.uniform_(self.bias, -stdv, stdv)
        else:
            self.register_parameter('bias', None)
        self.node_dropout = node_dropout
        self.edge_dropout = edge_dropout

    def forward(self, input, adj):
        adj = self.sparse_dropout(adj, self.edge_dropout)
        support = torch.mm(input, self.weight)
        support = F.dropout(support, self.node_dropout, training=self.training)
        output = torch.sparse.mm(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'
        
class SGATLayer(Module):
    def __init__(self, in_features, out_features, n_head=8, node_dropout=0.0,
                 edge_dropout=0.0, pre_attn_order=1, post_attn_order=1, 
                 pre_attn_appnp=False, pre_appnp_alpha=0.1, 
                 post_attn_appnp=False, post_appnp_alpha=0.1,
                 bias=True, mean=False, device='cpu'):
        super(SGATLayer, self).__init__()
        self.sparse_dropout = sparse_dropout()
        self.in_features = in_features
        self.out_features = out_features
        self.n_head = n_head
        self.weight = Parameter(torch.zeros(in_features, out_features * n_head))
        nn.init.xavier_normal_(self.weight, gain=1.414)
        if bias:
            self.bias = Parameter(torch.zeros(1, out_features, n_head))
            nn.init.xavier_normal_(self.bias, gain=1.414)
        else:
            self.register_parameter('bias', None)
        # self.a1 = Parameter(torch.zeros(out_features, n_head))
        # nn.init.xavier_normal_(self.a1, gain=1.414)
        self.a2 = Parameter(torch.zeros(out_features, n_head))
        nn.init.xavier_normal_(self.a2, gain=1.414)

        # self.attn_clip = attn_clip
        self.pre_attn_order = pre_attn_order
        self.post_attn_order = post_attn_order
        self.pre_attn_appnp = pre_attn_appnp
        self.pre_appnp_alpha = pre_appnp_alpha
        self.post_attn_appnp = post_attn_appnp
        self.post_appnp_alpha = post_appnp_alpha
        self.mean = mean
        self.node_dropout = node_dropout
        self.edge_dropout = edge_dropout
        self.device = device

    def forward(self, input, adj):
        adj = self.sparse_dropout(adj, self.edge_dropout)
        support0 = torch.mm(input, self.weight).view(input.shape[0], self.out_features, self.n_head)      
        support0 = F.dropout(support0, self.node_dropout, training=self.training)
        ones = torch.ones(input.shape[0], 1, self.n_head).to(self.device)
        # mask = F.dropout(ones, self.dropout, training=self.training)
        mask = ones
        support = support0 * mask # ones
        support = torch.cat([support, mask], 1) # (n_nodes, out_features+1, n_heads)
        support_old = support
        
        attn2 = torch.einsum('aij,ij->aj', support[:, :-1, :], self.a2)
        # attn2 = F.dropout(attn2, self.dropout, training=self.training)
        attn2 = attn2 + torch.sqrt(attn2 * attn2 + 1.)
        # attn2 = attn2.clamp(-self.attn_clip, self.attn_clip) # (n_nodes, out_features+1, n_heads)
        for _ in range(self.post_attn_order):
            attn_support = torch.einsum('aij,aj->aij', support, attn2)
            attn_support = attn_support.view(input.shape[0], -1)
            if self.pre_attn_appnp:
                attn_support_old = attn_support
                for _ in range(self.pre_attn_order):
                    attn_support = torch.sparse.addmm(attn_support_old,
                                                      adj, attn_support,
                                                      beta=1-self.pre_appnp_alpha,
                                                      alpha=self.pre_appnp_alpha)
            else:
                for _ in range(self.pre_attn_order):
                    attn_support = torch.sparse.mm(adj, attn_support)
            attn_support = attn_support.view(input.shape[0], -1, self.n_head)
            
            if self.post_attn_appnp:
                support = support_old * self.post_appnp_alpha + attn_support * (1 - self.post_appnp_alpha)
            else:
                support = attn_support
            
            if self.post_attn_appnp:
                support = support_old * self.post_appnp_alpha + attn_support * (1 - self.post_appnp_alpha)
                support = support.contiguous()
                support = support / (support[:, -1:, :] + 1e-9)
            else:
                support = attn_support
                support = support.contiguous()
        
        output = support[:, :-1, :] / (support[:, -1:, :] + 1e-9)
        
        if self.bias is not None:
            output += self.bias
        
        if self.mean:
            return output.mean(dim=2)
        else:
            return output.view(input.shape[0], -1)

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ' * '\
               + str(self.n_head) + ')'    
        
class SGATMultiLayer(Module):
    def __init__(self, in_features, out_features, n_head=8,  bases=4,
                 node_dropout=0.0, edge_dropout=0.0, pre_attn_order=1,
                 post_attn_order=1, pre_attn_appnp=False, pre_appnp_alpha=0.1, 
                 post_attn_appnp=False, post_appnp_alpha=0.1,
                 bias=True, mean=False, device='cpu'):
        super(SGATMultiLayer, self).__init__()
        self.sparse_dropout = sparse_dropout()
        self.in_features = in_features
        self.out_features = out_features
        self.n_head = n_head
        self.bases = bases
        self.weight = Parameter(torch.zeros(in_features, out_features * n_head))
        nn.init.xavier_normal_(self.weight, gain=1.414)
        if bias:
            self.bias = Parameter(torch.zeros(1, out_features, n_head))
            nn.init.xavier_normal_(self.bias, gain=1.414)
        else:
            self.register_parameter('bias', None)
        self.a1 = Parameter(torch.zeros(out_features, bases, n_head))
        nn.init.xavier_normal_(self.a1, gain=1.414)
        self.a2 = Parameter(torch.zeros(out_features, bases, n_head))
        nn.init.xavier_normal_(self.a2, gain=1.414)

        # self.attn_clip = attn_clip
        self.pre_attn_order = pre_attn_order
        self.post_attn_order = post_attn_order
        self.pre_attn_appnp = pre_attn_appnp
        self.pre_appnp_alpha = pre_appnp_alpha
        self.post_attn_appnp = post_attn_appnp
        self.post_appnp_alpha = post_appnp_alpha
        self.mean = mean
        self.node_dropout = node_dropout
        self.edge_dropout = edge_dropout
        self.device = device

    def forward(self, input, adj):
        adj = self.sparse_dropout(adj, self.edge_dropout)
        support0 = torch.mm(input, self.weight).view(input.shape[0], self.out_features, self.n_head)      
        support0 = F.dropout(support0, self.node_dropout, training=self.training)
        ones = torch.ones(input.shape[0], 1, self.n_head).to(self.device)
        # mask = F.dropout(ones, self.dropout, training=self.training)
        mask = ones
        support = support0 * mask # ones
        support = torch.cat([support, mask], 1) # (n_nodes, out_features+1, n_heads)
        support_old = support
        
        attn2 = torch.einsum('aij,ikj->akj', support[:, :-1, :], self.a2) # (n_nodes, bases, n_head)
        # attn2 = F.dropout(attn2, self.dropout, training=self.training)
        attn2 = attn2 + torch.sqrt(attn2 * attn2 + 1.)
        # attn2 = attn2.clamp(-self.attn_clip, self.attn_clip) # (n_nodes, out_features+1, n_heads)
        attn1 = None
        
        for i in range(self.post_attn_order):
            attn_support = torch.einsum('aij,akj->aikj', support, attn2)
            attn_support = attn_support.view(input.shape[0], -1)
            if self.pre_attn_appnp:
                attn_support_old = attn_support
                for _ in range(self.pre_attn_order):
                    attn_support = torch.sparse.addmm(attn_support_old,
                                                      adj, attn_support,
                                                      beta=self.pre_appnp_alpha,
                                                      alpha=1-self.pre_appnp_alpha)
            else:
                for _ in range(self.pre_attn_order):
                    attn_support = torch.sparse.mm(adj, attn_support)
            attn_support = attn_support.view(input.shape[0], -1, self.bases, self.n_head)
            
            if i == 0:
                attn1 = torch.einsum('aij,ikj->akj', support[:, :-1, :], self.a1) # (n_nodes, bases, n_head)
                attn1 = attn1 + torch.sqrt(attn1 * attn1 + 1.)
            attn_support = torch.einsum('aikj,akj->aij', attn_support, attn1)
            if self.post_attn_appnp:
                support = support_old * self.post_appnp_alpha + attn_support * (1 - self.post_appnp_alpha)
                support = support.contiguous()
                support = support / (support[:, -1:, :] + 1e-9)
            else:
                support = attn_support
                support = support.contiguous()
        
        output = support[:, :-1, :] / (support[:, -1:, :] + 1e-9)
        
        if self.bias is not None:
            output += self.bias
        
        if self.mean:
            return output.mean(dim=2)
        else:
            return output.contiguous().view(input.shape[0], -1)

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ' * '\
               + str(self.n_head) + ')'    

