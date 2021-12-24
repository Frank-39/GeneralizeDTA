import torch
import random
from torch_geometric.utils import negative_sampling, structured_negative_sampling
import numpy as np
from math import sqrt
from scipy import stats
def rmse(y,f):
    rmse = sqrt(((y - f)**2).mean(axis=0))
    return rmse
def mse(y,f):
    mse = ((y - f)**2).mean(axis=0)
    return mse
def pearson(y,f):
    rp = np.corrcoef(y, f)[0,1]
    return rp
def spearman(y,f):
    rs = stats.spearmanr(y, f)[0]
    return rs
def ci(y,f):
    ind = np.argsort(y)
    y = y[ind]
    f = f[ind]
    i = len(y)-1
    j = i-1
    z = 0.0
    S = 0.0
    while i > 0:
        while j >= 0:
            if y[i] > y[j]:
                z = z+1
                u = f[i] - f[j]
                if u > 0:
                    S = S + 1
                elif u == 0:
                    S = S + 0.5
            j = j - 1
        i = i - 1
        j = i-1
    ci = S/z
    return ci
class TaskConstruction:
    def __init__(self, args):
        """
        construct tasks
        """
        self.args = args

    def __call__(self, data):
        num_nodes = data.num_nodes
        num_edges = data.num_edges
        # print('data',data)

        # sample support set and query set for each data/task/graph
        num_sampled_edges = self.args.node_batch_size * (self.args.support_set_size + self.args.query_set_size)


        perm = np.random.randint(num_edges, size=num_sampled_edges)

        # print('perm:',perm)
        # assert False
        pos_edges = data.edge_index[:, perm]

        x = 1 - 1.1 * (data.edge_index.size(1) / (num_nodes * num_nodes) )
        if x != 0:
            alpha = 1 / (1 - 1.1 * (data.edge_index.size(1) / (num_nodes * num_nodes) ))
        else:
            alpha = 0
        if alpha > 0:
            neg_edges = negative_sampling(data.edge_index, num_nodes, num_sampled_edges)

        else:
            i, _, k = structured_negative_sampling(data.edge_index)
            neg_edges = torch.stack((i,k), 0)
        cur_num_neg = neg_edges.shape[1]
        if cur_num_neg != num_sampled_edges:
            perm = np.random.randint(cur_num_neg, size=num_sampled_edges)
            neg_edges = neg_edges[:, perm]

        # print(pos_edges)
        # print(neg_edges)
        # print('*************')
        # print(pos_edges.size())
        # print(neg_edges.size())
        # assert False

        data.pos_sup_edge_index = pos_edges[:, :self.args.node_batch_size * self.args.support_set_size]
        data.neg_sup_edge_index = neg_edges[:, :self.args.node_batch_size * self.args.support_set_size]
        data.pos_que_edge_index = pos_edges[:, self.args.node_batch_size * self.args.support_set_size:]
        data.neg_que_edge_index = neg_edges[:, self.args.node_batch_size * self.args.support_set_size:]

        # print(data.pos_que_edge_index.shape)
        # print(data.pos_sup_edge_index.shape)
        # print(data.neg_que_edge_index.shape)
        # print(data.neg_sup_edge_index.shape)
        #
        # assert False

        return data

