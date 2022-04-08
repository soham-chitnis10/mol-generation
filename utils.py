import torch
from torch_geometric.loader import DataLoader
def get_adj(data):
    edge_index = data.edge_index
    n = data.x.shape[0]
    adj = torch.zeros([n,n])
    for i in range(edge_index.shape[1]):
        j = edge_index[0,i].item()
        k = edge_index[1,i].item()
    #     print(adj[j,k],adj[k,j])
    #     print(j,k)
        adj[j,k]=1
    #     print("edge_added")
    #     print(adj[j,k],adj[k,j])
    return adj
def recon_loss(data,pred_adj):
    criterion = torch.nn.BCEWithLogitsLoss()
    return criterion(get_adj(data),pred_adj)
