import torch_geometric
import torch_geometric.nn as nn
from model import GraphEncoder,GraphDecoder
import torch
import numpy as np
from torch_geometric.datasets import TUDataset
from torch_geometric.loader import DataLoader
from torch_geometric import utils
import torch.nn.functional as F
import argparse
import os
from torch.utils.data import random_split
from torch.utils.data.dataset import Subset
from utils import get_adj
import networkx as nx
import matplotlib.pyplot as plt
from torch_geometric.nn import GCNConv,ARMAConv,GraphConv,GATConv,SAGEConv
parser = argparse.ArgumentParser()

parser.add_argument('--seed', type=int, default=777,
                    help='seed')
parser.add_argument('--batch_size', type=int, default=128,
                    help='batch size')
parser.add_argument('--lr', type=float, default=0.0008,
                    help='learning rate')
parser.add_argument('--weight_decay', type=float, default=0.0001,
                    help='weight decay')
parser.add_argument('--pooling_ratio', type=float, default=0.5,
                    help='pooling ratio')
parser.add_argument('--dropout_ratio', type=float, default=0.5,
                    help='dropout ratio')
parser.add_argument('--dataset', type=str, default='BOTDS',
                    help='dataset sub-directory under dir: data. e.g. BOTDS')
parser.add_argument('--epochs', type=int, default=1000,
                    help='maximum number of epochs')
parser.add_argument('--val_epochs', type=int, default=25,
                    help='maximum number of validation epochs')
parser.add_argument('--patience', type=int, default=50,
                    help='patience for earlystopping')
parser.add_argument('--pooling_layer_type', type=str, default='GCNConv',
                    help='type of pooling layer')
parser.add_argument('--use_node_attr', type=bool, default=True,
                    help='node features')
parser.add_argument('--variational', type=bool, default=False,
                    help='Varitional Graph Auto Encoder')
#Load GPU (If present)
args = parser.parse_args()
# args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
args.device = torch.device("cpu")
dataset = TUDataset('data',name = args.dataset,use_node_attr=args.use_node_attr)
args.nhid = 128
args.num_features = dataset.num_features
args.num_node_features = dataset.num_node_features
loader = DataLoader(dataset, batch_size=1, shuffle=False)
args.max_num_nodes = 20
args.latent_embeding_size = args.nhid*2
args.decoder_hidden_size = args.latent_embeding_size*4
args.variational = True
args.lambda_recon = 0.8
args.conv = GCNConv
if args.variational:
    model = nn.VGAE(GraphEncoder(args),GraphDecoder(args))
else:
    model = nn.GAE(GraphEncoder(args),GraphDecoder(args))
model.to(args.device)
optimizer = torch.optim.Adam(model.parameters(),lr = args.lr)
for epoch in range(15):
    j =0
    print("Started Training epoch: ",epoch)
    model.train()
    for i,data in enumerate(loader):
        if data.num_nodes == args.max_num_nodes:
            j=j+1
            optimizer.zero_grad()
            z = model.encode(data)
            adj,node_features= model.decode(z)
            loss = model.decoder.recon_loss(data,adj,node_features)
            if args.variational:
                loss = loss + model.kl_loss()
            print(loss)
            loss.backward()
            optimizer.step()
            if(j>2):
                break
    
j=0
print("Started Validation")
model.eval()
for i,data in enumerate(loader):
        if data.num_nodes == args.max_num_nodes:
            j=j+1
            if j<=2:
                continue
            z = model.encode(data)
            adj,node_features= model.decode(z)
            loss = model.decoder.recon_loss(data,adj,node_features)
            if args.variational:
                loss = loss + model.kl_loss()
            print(f"loss:{loss}")
            print(adj)
            torch.round_(adj)
            adj_actual = get_adj(data)
            adj_actual = adj_actual.numpy()
            G = nx.from_numpy_array(adj_actual)
            print(type(G.edges()))
            nx.draw(G, with_labels=True)
            plt.show()
            # plt.savefig('actual_graph_20.png')
            adj = adj.detach().numpy()
            G1 = nx.from_numpy_array(adj)
            print(G1.edges())
            nx.draw(G1, with_labels=True)
            plt.show()
            print(nx.graph_edit_distance(G,G1))
            # plt.savefig('predicted_graph_20.png')            
            break

