import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch.nn import Linear


class ASDGCN(torch.nn.Module):
    """
    Two layer GCN model for ASD classification. Population based graph, features
    are preprocessed f-mri scans.
    """

    def __init__(self, num_features , num_classes):
        super().__init__()
        self.conv1 = GCNConv(num_features, 200)
        self.conv2 = GCNConv(200,num_classes)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=0.3, training=self.training)  #Training dropout
        x = self.conv2(x, edge_index)
        return x
