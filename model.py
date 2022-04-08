from platform import node
from numpy import std
import torch
import torch_geometric
from torch_geometric.nn import GCNConv,ARMAConv,GraphConv,GATConv,SAGEConv
from torch_geometric.nn import global_mean_pool as gap, global_max_pool as gmp
from torch_geometric.nn.models import InnerProductDecoder
import torch.nn.functional as F
from layers import SAGPool
from torch_geometric.loader import DataLoader
from utils import get_adj
class GraphEncoder(torch.nn.Module):
    def __init__(self,args):
        super(GraphEncoder, self).__init__()
        self.args = args
        self.num_features = args.num_features
        self.nhid = args.nhid
        self.pooling_ratio = args.pooling_ratio
        self.dropout_ratio = args.dropout_ratio
        self.variational = args.variational
        self.conv = args.conv
        # Encoder Layers
        self.conv1 = self.conv(self.num_features, self.nhid)
        self.pool1 = SAGPool(self.nhid, ratio=self.pooling_ratio)
        self.conv2 = self.conv(self.nhid, self.nhid)
        self.pool2 = SAGPool(self.nhid, ratio=self.pooling_ratio)
        self.conv3 = self.conv(self.nhid, self.nhid)
        self.pool3 = SAGPool(self.nhid, ratio=self.pooling_ratio)
        # #Latent Layer transformation for Graph VAE
        if self.variational:
            self.latent_embeding_size = args.latent_embeding_size
            self.mu_transform = torch.nn.Linear(self.nhid*2, self.latent_embeding_size)
            self.log_var_transform = torch.nn.Linear(self.nhid*2, self.latent_embeding_size)

    def forward(self,data):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        x = F.relu(self.conv1(x, edge_index))
        x, edge_index, _, batch, _ = self.pool1(x, edge_index, None, batch)
        x1 = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)

        x = F.relu(self.conv2(x, edge_index))
        x, edge_index, _, batch, _ = self.pool2(x, edge_index, None, batch)
        x2 = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)

        x = F.relu(self.conv3(x, edge_index))
        x, edge_index, _, batch, _ = self.pool3(x, edge_index, None, batch)
        x3 = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)

        x = x1 + x2 + x3
        # # Latent transformation layers for mu and sigma 
        if self.variational:
            mu = self.mu_transform(x)
            log_var = self.log_var_transform(x)
            return mu,log_var
        else:
            return x

class GraphDecoder(torch.nn.Module):
    def __init__(self,args) -> None:
        super(GraphDecoder,self).__init__()
        self.args = args
        self.num_node_features = args.num_node_features
        self.nhid = args.nhid
        self.decoder_hidden_size = args.decoder_hidden_size
        self.max_num_nodes = args.max_num_nodes
        self.output_dim = args.max_num_nodes*(args.max_num_nodes-1)//2
        self.latent_embeding_size = args.latent_embeding_size
        self.variational = args.variational
        self.linear1 = torch.nn.Linear(self.latent_embeding_size, self.decoder_hidden_size)
        self.linear2 = torch.nn.Linear(self.decoder_hidden_size, self.output_dim)
        self.linear3 = torch.nn.Linear(self.decoder_hidden_size,self.num_node_features*self.max_num_nodes)
        self.lambda_recon = args.lambda_recon

    def forward(self,z):
        x = F.leaky_relu(self.linear1(z))
        adj_upper = self.linear2(x)
        adj_upper = torch.sigmoid(adj_upper)
        node_features= F.leaky_relu(self.linear3(x))
        adj = self.recon_adj(adj_upper)
        node_features = node_features.view(self.max_num_nodes,self.num_node_features)
        node_features = torch.sigmoid(node_features)
        return adj,node_features
    
    def recon_adj(self,adj_upper):
        adj = torch.zeros(self.max_num_nodes,self.max_num_nodes)
        adj[torch.triu(torch.ones(self.max_num_nodes,self.max_num_nodes),diagonal=1)==1] = adj_upper
        diag = torch.diag(torch.diag(adj, 0))
        adj = adj + torch.transpose(adj, 0, 1) - diag
        return adj
    
    def recon_loss(self,data,adj_pred,node_features_pred):
        print
        adj_recon_loss = F.binary_cross_entropy(adj_pred,get_adj(data))
        node_features_recon_loss = F.binary_cross_entropy(node_features_pred,data.x)
        loss = adj_recon_loss*self.lambda_recon + node_features_recon_loss*(1-self.lambda_recon)
        return loss