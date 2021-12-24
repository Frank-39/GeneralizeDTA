from dataloader import DataLoaderAE
from loader_chem import Cp_dataset,MoleculeDataset
import torch.optim as optim
import numpy as np
from metapre_for_alldata import MetaPre
from progressbar import *
from util import *
from torch_geometric.data import DataLoader
import os
import argparse
import torch
from plus.data.alphabets import Protein
from tqdm import tqdm
from util import TaskConstruction
import sys


sys.path.append( "path" )

def mse(y,f):
    mse = ((y - f)**2).mean(axis=0)
    return mse
def predicting(model, device, loader):
    model.eval()
    total_preds = torch.Tensor().to(device)
    total_labels = torch.Tensor().to(device)
    print('Make prediction for {} samples...'.format(len(loader.dataset)))
    with torch.no_grad():

        for data in tqdm(loader, desc="Iteration",ncols=80):
            data = data.to(device)
            output = model(data)
            total_preds = torch.cat((total_preds, output), 0)
            total_labels = torch.cat((total_labels, data.y.view(-1, 1)), 0)
    return total_labels.flatten(), total_preds.flatten()
def train(args, model, chem_loader,cp_loader,valid_loader,test_loader,optimizer):
    # train
    model.train()
    c = []
    best_epoch= -1
    best_mse = 1000
    for epoch in range(1, args.epochs + 1):

        print("====epoch " + str(epoch))
        for step, (batch_cp,batch_chem) in tqdm(enumerate(zip(cp_loader,chem_loader))):
            model.meta_gradient_step(batch_chem,batch_cp,optimizer)
        model_file_name='res3/'+ str(epoch)+'_epoch.model'
        print('predicting for test data')
        if epoch>1:
            G, P = predicting(model, device, test_loader)
            G = G.cpu().numpy()
            P = P.cpu().numpy()
            ret = [rmse(G,P),mse(G,P),pearson(G,P),spearman(G,P),ci(G,P)]
            if ret[1].item() < best_mse:
                best_epoch = epoch + 1
                best_mse = ret[1].item()
                torch.save(model.state_dict(), model_file_name)

                print('\n\n\nrmse improved at epoch ', best_epoch, '; \nbest_test_mse,best_test_ci:', best_mse)
            else:
                print("\n当前mse:",ret[1])
                print("最佳mse:",best_mse,"最佳的epoch:",best_epoch)
                print("\n\n\n")
            f = open(f"test/{epoch}",'w')
            f.write(str(ret[1]))
            f.close()




class Model_cfg:
    def __init__(self,config_dict):
        self.dropout = config_dict["dropout"]
        self.feedforward_dim=    config_dict['feedforward_dim']
        self.hidden_dim =  config_dict[ 'hidden_dim']
        self.idx = config_dict['idx']
        self.input_dim = config_dict['input_dim']
        self.max_len = config_dict['max_len']
        self.model_type = config_dict['model_type']
        self.num_classes = config_dict['num_classes']
        self.num_heads = config_dict['num_heads']
        self.num_layers = config_dict['num_layers']
        self.pos_encode = config_dict['pos_encode']

if __name__ == "__main__":
    print (time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
    parser = argparse.ArgumentParser(
        description='PyTorch implementation of meta-learning-like pre-training of graph neural networks')
    parser.add_argument('--device', type=int, default=0,
                        help='which gpu to use if any (default: 0)')
    parser.add_argument('--graph_batch_size', type=int, default=64,
                        help='input batch size for parent tasks (default: 64)')
    parser.add_argument('--cp_batch_size', type=int, default=64,
                        help='input batch size for parent tasks (default: 64)')
    parser.add_argument('--node_batch_size', type=int, default=1,
                        help='input batch size for parent tasks (default: 3)')
    parser.add_argument('--epochs', type=int, default=100,
                        help='number of epochs to train (default: 100)')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='learning rate (default: 0.001)')
    parser.add_argument('--decay', type=float, default=0,
                        help='weight decay (default: 0)')
    parser.add_argument('--emb_dim', type=int, default=300,
                        help='embedding dimensions (default: 300)')

    parser.add_argument('--num_workers', type=int, default=0, help='number of workers for dataset loading')
    # gnn setting
    parser.add_argument('--gnn_type', type=str, default='gcn')
    parser.add_argument('--num_layer', type=int, default=5,
                        help='number of GNN message passing layers (default: 5).')
    parser.add_argument('--dropout_ratio', type=float, default=0.1,
                        help='dropout ratio (default: 0)')
    parser.add_argument('--graph_pooling', type=str, default="mean")
    parser.add_argument('--model_file', type=str, default='test', help='filename to output the pre-trained model')

    # meta-learning settings
    parser.add_argument('--order', type=int, default=2, help='gradient order')
    parser.add_argument('--node_level', type=int, default=1, help='node-level adaptation')
    parser.add_argument('--graph_level', type=int, default=1, help='graph-level adaptation')
    parser.add_argument('--node_lr', type=float, default=0.0001, help='learning rate for node-level adaptation')
    parser.add_argument('--node_update', type=int, default=1, help='update step for node-level adaptation')
    parser.add_argument('--graph_lr', type=float, default=0.0001, help='learning rate for graph-level adaptation')
    parser.add_argument('--graph_update', type=int, default=1, help='update step for graph-level adaptation')
    parser.add_argument('--support_set_size', type=int, default=10, help='size of support set')
    parser.add_argument('--query_set_size', type=int, default=5, help='size of query set')

    # dataset settings
    parser.add_argument('--dataset', type=str, default='chem')
    parser.add_argument('--node_fea_dim', type=int, default=2)
    parser.add_argument('--edge_fea_dim', type=int, default=2)
    args = parser.parse_args()

    torch.manual_seed(0)
    np.random.seed(0)

    # device = torch.device("cpu")
    device = torch.device("cuda:" + str(args.device)) if torch.cuda.is_available() else torch.device("cpu")
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(0)
    args.device = device
    print('开始训练2')

    alphabet = Protein()
    model_config_dict = {"dropout": 0.1, "feedforward_dim": 3072, "hidden_dim": 768, "idx": "model_config",
                         "input_dim": 25, "max_len": 512,
                         "model_type": "TFM", "num_classes": 2, "num_heads": 12, "num_layers": 12, "pos_encode": True}
    model_cfg = Model_cfg(model_config_dict)

    os.environ['CUDA_VISIBLE_DEVICES'] = 'cuda:0'
    # data_parallel = False

    device, data_parallel = torch.device("cuda" if torch.cuda.is_available() else "cpu"), torch.cuda.device_count() > 1

    print('set up dataset')
    # set up dataset
    root_unsupervised = f'../data/chem/zinc_standard_agent'
    root_cp_dataset = f'../data/davis'
    dataset = MoleculeDataset(root_unsupervised, dataset='zinc_standard_agent', transform=TaskConstruction(args))
    train_data = Cp_dataset(root_cp_dataset,dataset='train')
    valid_data = Cp_dataset(root_cp_dataset,dataset='dev')
    cp_test_dataset = Cp_dataset(root_cp_dataset,dataset='test')

    # loader是预训练的数据
    pretrain_loader = DataLoaderAE(dataset, batch_size=args.graph_batch_size, shuffle=True, num_workers=args.num_workers)
    train_loader =DataLoader(train_data, batch_size=args.cp_batch_size, shuffle=True,drop_last=True)
    valid_loader = DataLoader(valid_data, batch_size=args.cp_batch_size, shuffle=False,drop_last=True)
    test_loader = DataLoader(cp_test_dataset, batch_size=args.cp_batch_size, shuffle=False,drop_last=True)
    print('set up model')
    
    # set up model

    metapre = MetaPre(args,model_cfg,None)
#     print(metapre)
    metapre.tfm.load_weights("../pretrain/PLUS-TFM.pt")
    metapre = metapre.to(args.device)
    # metapre.load_state_dict({k.replace('module.',''):v for k,v in torch.load('QQQQQQQ4_epoch.model').items()})
    for name, p in metapre.named_parameters():
        if name.startswith('tfm'):
            p.requires_grad = False
    # set up optimizer
    # optimizer = optim.Adam(metapre.parameters(), lr=args.lr, weight_decay=args.decay)
    optimizer = optim.Adam(filter(lambda x: x.requires_grad is not False ,metapre.parameters()),lr= 0.001)
    print('begining training')

    train(args, metapre,pretrain_loader,train_loader,valid_loader,test_loader, optimizer)