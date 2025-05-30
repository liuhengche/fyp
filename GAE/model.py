import torch
from torch_geometric.nn import SAGEConv
from torch.nn import Linear


class ODEncoderModel(torch.nn.Module):
    def __init__(self, hidden_channels, out_channels):
        super().__init__()
        self.conv1 = SAGEConv((-1, -1), hidden_channels)
        self.conv2 = SAGEConv((-1, -1), hidden_channels)
        self.conv3 = SAGEConv((-1, -1), hidden_channels)
        self.lin = Linear(hidden_channels, out_channels)

    def forward(self, x_dict, edge_index_dict):
        detector_x = self.conv1(
            x_dict['detector_time'],
            edge_index_dict[('detector_time', 'metapath_0', 'detector_time')],
        ).relu()
        od_x = self.conv2(
            (detector_x, x_dict['od']),
            edge_index_dict[('detector_time', 'rev_assignment', 'od')],
        ).relu()
        od_x = self.conv3(
            (detector_x, od_x),
            edge_index_dict[('detector_time', 'rev_assignment', 'od')],
        ).relu()

        return self.lin(od_x)

class DetectorTimeEncoder(torch.nn.Module):
    def __init__(self, hidden_channels, out_channels):
        super().__init__()
        self.conv1 = SAGEConv(-1, hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, hidden_channels)
        self.lin = Linear(hidden_channels, out_channels)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        x = self.conv2(x, edge_index).relu()
        return self.lin(x)

class EdgeDecoder(torch.nn.Module):
    def __init__(self, hidden_channels):
        super().__init__()
        self.lin1 = Linear(2 * hidden_channels, hidden_channels)
        self.lin2 = Linear(hidden_channels, 1)

    def forward(self, z_src, z_dst, edge_label_index):
        row, col = edge_label_index
        z = torch.cat([z_src[row], z_dst[col]], dim=-1)
        z = self.lin1(z).relu()
        z = self.lin2(z).sigmoid()
        return z.view(-1)

class Model(torch.nn.Module):
    """Model architecture"""
    def __init__(self, hidden_channels, out_channels):
        super().__init__()
        self.od_encoder = ODEncoderModel(hidden_channels, out_channels)
        self.detector_time_encoder = DetectorTimeEncoder(hidden_channels, out_channels)
        self.decoder = EdgeDecoder(out_channels)

    def forward(self, x_dict, edge_index_dict, edge_label_index):
        z_dict = {
            'od': self.od_encoder(x_dict, edge_index_dict),
            'detector_time': self.detector_time_encoder(
                x_dict['detector_time'],
                edge_index_dict[('detector_time', 'metapath_0', 'detector_time')]
            )
        }
        return self.decoder(z_dict['od'], z_dict['detector_time'], edge_label_index)
