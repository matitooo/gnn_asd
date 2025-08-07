import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch.nn import Linear


class ASDGCN(torch.nn.Module):
    """
    Two layer GCN model for ASD classification. Population based graph, features
    are preprocessed f-mri scans.
    """

    def __init__(self, num_features, hidden_channels, num_classes):
        super().__init__()
        self.conv1 = GCNConv(num_features, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, int(hidden_channels/2))
        self.conv3 = GCNConv(int(hidden_channels/2),int(hidden_channels/4))
        self.lin = Linear(int(hidden_channels/4), num_classes)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = self.conv3(x, edge_index)
        x = F.relu(x)
        x = self.lin(x)
        return x
