import random
import time
from collections import OrderedDict
import torch
import torch.nn as nn
from torchvision import models
from model_chem import MetaGCN, MetaGIN, MetaPool, MetaGAT, MetaGraphSAGE
from torchvision.models.resnet import *
import torch.nn.functional as F
from embinitialization import EmbInitial, EmbInitial_DBLP, EmbInitial_CHEM

from torch_geometric.nn import global_max_pool as gmp
from model_chem import MetaGCN, MetaGIN, MetaPool, MetaGAT, MetaGraphSAGE,fc
import plus.model.plus_tfm as plus_tfm
from plus.preprocess import preprocess_seq_for_tfm
from plus.model.plus_tfm import get_loss
from torch_geometric.nn import global_mean_pool


#args.model_cfg
class Pooling(torch.nn.Module):
    def __init__(self):
        super(Pooling, self).__init__()
        self.dense = nn.Linear(768, 768)
        self.activation = nn.Tanh()

    def forward(self, hidden_states):
        # We "pool" the model by simply taking the hidden state corresponding
        # to the first token.
        first_token_tensor = hidden_states[:, 0]
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output
#args.model_cfg
class MetaPre(torch.nn.Module):
    def __init__(self, args,model_cfg,run_cfg):
        super(MetaPre, self).__init__()
        self.args = args
        self.sup_size = args.support_set_size
        self.que_size = args.query_set_size
        self.emb_initial = EmbInitial_CHEM(args.emb_dim, args.node_fea_dim)

        self.gnn = MetaGCN(args.emb_dim, args.edge_fea_dim, args.dropout_ratio)
        self.pool = MetaPool(args.emb_dim)
        self.loss = nn.BCEWithLogitsLoss()

########################## protein 用到的##########################
        self.tfm = plus_tfm.PLUS_TFM(model_cfg)
        self.pooling = Pooling()
        self.fc2 = nn.Linear(768,256)
############################## pre chem用到的###############################
        self.fc = fc(556, 1024, 512,1) # 最后一维batch_size
        self.y_lf = nn.MSELoss()
    def cycle_index(self, num, shift):
        arr = torch.arange(num) + shift
        arr[-shift:] = torch.arange(shift)

        return arr

    def meta_gradient_step(self,batch_data, cp_batch_data ,optimizer):


        torch.autograd.set_detect_anomaly(True)


        batch_data = batch_data.to(self.args.device)
        x = self.emb_initial(batch_data.x)
        sup_task_nodes_emb, que_task_nodes_emb = [], []

        # node-level
        node_loss = []
        node_acc = []
        for idx in range(self.args.node_batch_size):

            cur_pos_sup_e_idx = batch_data.pos_sup_edge_index[:,idx * self.sup_size * self.args.graph_batch_size:
                                                                (idx + 1) * self.sup_size * self.args.graph_batch_size]
            cur_neg_sup_e_idx = batch_data.neg_sup_edge_index[:,idx * self.sup_size * self.args.graph_batch_size:
                                                                (idx + 1) * self.sup_size * self.args.graph_batch_size]
            cur_pos_que_e_idx = batch_data.pos_que_edge_index[:,idx * self.que_size * self.args.graph_batch_size:
                                                                (idx + 1) * self.que_size * self.args.graph_batch_size]
            cur_neg_que_e_idx = batch_data.neg_que_edge_index[:,idx * self.que_size * self.args.graph_batch_size:
                                                                (idx + 1) * self.que_size * self.args.graph_batch_size]


            fast_weights = OrderedDict(self.gnn.named_parameters())

            for step in range(self.args.node_update):
                node_emb = self.gnn(x, batch_data.edge_index, batch_data.edge_attr, fast_weights)
                pos_score = torch.sum(node_emb[cur_pos_sup_e_idx[0]] *
                                      node_emb[cur_pos_sup_e_idx[1]], dim=1)  # ([n_batch*#sup_set])
                neg_score = torch.sum(node_emb[cur_neg_sup_e_idx[0]] *
                                      node_emb[cur_neg_sup_e_idx[1]], dim=1)
                loss = self.loss(pos_score, torch.ones_like(pos_score)) + \
                       self.loss(neg_score, torch.zeros_like(neg_score))
                gradients = torch.autograd.grad(loss, fast_weights.values(), create_graph=True)
                # update weights manually
                fast_weights = OrderedDict(
                    (name, param - self.args.node_lr * grad)
                    for ((name, param), grad) in zip(fast_weights.items(), gradients)
                )
            sup_task_nodes_emb.append(
                node_emb[cur_pos_sup_e_idx].reshape(-1, self.args.emb_dim))  # ([(#sup_set, dim), ...])
            que_task_nodes_emb.append(
                node_emb[cur_pos_que_e_idx].reshape(-1, self.args.emb_dim))  # ([(#que_set, dim), ...])

            node_emb = self.gnn(x, batch_data.edge_index, batch_data.edge_attr, fast_weights)
            pos_score = torch.sum(node_emb[cur_pos_que_e_idx[0]] *
                                  node_emb[cur_pos_que_e_idx[1]], dim=1)  # ([n_batch*#sup_set])
            neg_score = torch.sum(node_emb[cur_neg_que_e_idx[0]] *
                                  node_emb[cur_neg_que_e_idx[1]], dim=1)
            loss = self.loss(pos_score, torch.ones_like(pos_score)) + \
                   self.loss(neg_score, torch.zeros_like(neg_score))
            acc = (torch.sum(pos_score > 0) + torch.sum(neg_score < 0)).to(torch.float32) / float(2 * len(pos_score))
            node_loss.append(loss)
            node_acc.append(acc)



        # graph level
        g_fast_weights = OrderedDict(self.pool.named_parameters())
        graph_emb = self.pool(node_emb, weights=g_fast_weights).squeeze()  # (#num_graph, dim)
        neg_graph_emb = graph_emb[self.cycle_index(len(graph_emb), 1)]

        task_emb = torch.cat(([self.pool(ns_e, weights=g_fast_weights)
                               for ns_e in sup_task_nodes_emb]), dim=0)  # (node_batch_size, dim)
        graph_pos_score = torch.sum(task_emb * graph_emb.repeat(self.args.node_batch_size, 1), dim=1)
        graph_neg_score = torch.sum(task_emb * neg_graph_emb.repeat(self.args.node_batch_size, 1), dim=1)
        loss = self.loss(graph_pos_score, torch.ones_like(graph_pos_score)) + \
               self.loss(graph_neg_score, torch.zeros_like(graph_neg_score))
        gradients = torch.autograd.grad(loss, g_fast_weights.values(), create_graph=True)
        g_fast_weights = OrderedDict(
            (name, param - self.args.graph_lr * grad)
            for ((name, param), grad) in zip(g_fast_weights.items(), gradients)
        )
#################################DTA训练过程#######################################################
        
        
        batch = cp_batch_data.batch
        x_chem_protein = self.emb_initial(cp_batch_data.x)
        
        target_token=cp_batch_data.target_token.view(self.args.cp_batch_size,512)
        target_segments=cp_batch_data.segments.view(self.args.cp_batch_size, 512)
        target_input_mask=cp_batch_data.input_mask.view(self.args.cp_batch_size, 512)

        y = cp_batch_data.y # 亲和度的值

        x_chem_emb = self.gnn(x_chem_protein, cp_batch_data.edge_index, cp_batch_data.edge_attr,fast_weights)
        target_emb = self.tfm(target_token,target_segments,target_input_mask,embedding=True)
        
        xt = self.pool(x_chem_emb, g_fast_weights,batch)
 
        target_embbing = self.fc2(self.pooling(target_emb))
        xc = torch.cat((xt, target_embbing), 1)
        out = self.fc(xc)

        loss_y = self.y_lf(out, y.view(-1, 1).float().to('cuda:0')) + 0.5*node_loss[0]
        # optimization
        optimizer.zero_grad()
        loss_y.backward()
        optimizer.step()
        # print(loss_y)
        return loss_y

    # 训练不用到forward，测试时可以加速过程
    def forward(self, cp_batch_data):
        batch = cp_batch_data.batch
        x_chem_protein = self.emb_initial(cp_batch_data.x)
        
        target_token=cp_batch_data.target_token.view(self.args.cp_batch_size,512)
        target_segments=cp_batch_data.segments.view(self.args.cp_batch_size, 512)
        target_input_mask=cp_batch_data.input_mask.view(self.args.cp_batch_size, 512)
        x_chem_emb = self.gnn(x_chem_protein, cp_batch_data.edge_index, cp_batch_data.edge_attr)
        target_emb = self.tfm(target_token,target_segments,target_input_mask,embedding=True)
        
        xt = self.pool(x_chem_emb,batch=batch)
 
        target_embbing = self.fc2(self.pooling(target_emb))
        xc = torch.cat((xt, target_embbing), 1)
        out = self.fc(xc)
        return out






if __name__=='__main__':
    # a = MetaPre().cuda()
    # print(a)
    pass