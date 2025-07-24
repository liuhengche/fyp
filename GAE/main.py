import os
import torch
from torch_geometric.loader import DataLoader
from torch_geometric.transforms import RandomLinkSplit, ToUndirected, AddMetaPaths
from torch_geometric.nn.conv.gcn_conv import gcn_norm
import torch.nn.functional as F
from datetime import datetime
from tqdm import tqdm
from datetime import datetime
from odDataset import CachedExperimentDataset
from model import Model
from torch_geometric.data import HeteroData
import loguru
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import pearsonr
from train import Main as tmain

DEMAND_TYPES = ['high', 'low', 'mid']
NUM_EXPERIMENTS = 1000
SELECTED_DETECTORS_PATH = r"D:\desktop\ta\202504\dataset\filtered_edges.txt"
BASE_PATH = r"D:\desktop\ta\202504\dataset"


class Runner:
    def __init__(self):
        pass

    def loss_recon_node(self, data: HeteroData, alphas: torch.Tensor):
        """Calculate reconstruction loss for node features."""
        # u, v set (od_veh_num, link_veh_count)
        x = data['od'].veh_num
        y = data['detector_time'].link_count
        assert isinstance(x, torch.Tensor) and isinstance(y, torch.Tensor), "Node features must be tensors"
        
        num_node_u = data['od'].num_nodes
        num_node_v = data['detector_time'].num_nodes

        # edges
        adj = data['detector_time', 'rev_assignment', 'od'].edge_index
        assert adj.shape[-1] == alphas.shape[0], "Edge indices and alphas must match in size"

        if x.dim() == 1:
            x = x.unsqueeze(-1)
        if y.dim() == 1:
            y = y.unsqueeze(-1)
        sparse_alpha = torch.sparse_coo_tensor(indices=adj, values=alphas, 
                                               size=(num_node_v, num_node_u),
                                               dtype=x.dtype, device=x.device)
        y_pred = torch.spmm(sparse_alpha, x).squeeze(-1)
        y = y.squeeze(1)
        # pearson correlation plotting
        y_np = y.detach().cpu().numpy()
        y_pred_np = y_pred.detach().cpu().numpy()
        pearson_r, _ = pearsonr(y_np, y_pred_np)

        # Create scatter plot
        plt.figure(figsize=(8, 8))
        plt.scatter(y_np, y_pred_np, color='blue', alpha=0.7, label='Predicted vs True')
        plt.xlabel('y (True Values)')      # x-axis labeled as 'y'
        plt.ylabel('y_pred (Predicted)')  # y-axis labeled as 'y_pred'
        plt.title(f'Pearson Correlation: r = {pearson_r:.2f}')  # Add Pearson value to title
        x = np.arange(0, max(y_np.max(), y_pred_np.max()) + 1, 0.1)
        x_ = x
        plt.plot(x, x_, color='red', linestyle='--')
        plt.grid(True)
        plt.legend()
        plt.xticks([])
        plt.yticks([])
        plt.savefig(f'pearson_correlation.png')
        plt.show()


        return self.huber_loss(pred=y_pred, target=y, delta=1.0)
    
    def huber_loss(self, pred: torch.Tensor, target: torch.Tensor, delta: float = 0.25):
        if pred.dim() == 1:
            pred = pred.unsqueeze(-1)
        if target.dim() == 1:
            target = target.unsqueeze(-1)
        assert pred.shape == target.shape, f"Prediction and target must have the same shape, prediction shape: {pred.shape}, target shape: {target.shape}"
        return F.huber_loss(pred, target, reduction='mean', delta=delta)
    
    def main(self):
        dataset = CachedExperimentDataset(
            base_path=BASE_PATH,
            demand_types=DEMAND_TYPES,
            num_experiments=NUM_EXPERIMENTS,
            processed_dir='processed_data',
            use_single_cache=True,
            force_rebuild=False
        )
        
        # Create dataloader
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        dataloader = DataLoader(dataset, batch_size=1, shuffle=False, follow_batch=['od', 'detector_time'])
        model = Model(hidden_channels=64, out_channels=64).to(device)
        model_path = 'best_model.pt'
        if os.path.exists(model_path):
            checkpoint = torch.load(model_path)
            model.load_state_dict(checkpoint['model_state_dict'])
        
        for data in tqdm(dataloader):
            
            data = data.to(device)
            data = ToUndirected()(data)
            
            metapaths = [[('detector_time', 'rev_assignment', 'od'), ('od', 'assignment', 'detector_time')]]
            data = AddMetaPaths(metapaths=metapaths)(data)

            # GCN normalization
            _, edge_weight = gcn_norm(
                data['detector_time', 'metapath_0', 'detector_time'].edge_index,
                num_nodes=data['detector_time'].num_nodes,
                add_self_loops=False
            )
            edge_index_metapath = data['detector_time', 'metapath_0', 'detector_time'].edge_index[:, edge_weight > 0.01] # 0.002
            data['detector_time', 'metapath_0', 'detector_time'].edge_index = edge_index_metapath
            
            out = model(
                    data.x_dict,
                    data.edge_index_dict,
                    data['od', 'assignment', 'detector_time'].edge_index
            )

            self.loss_recon_node(data, out)
            break


if __name__ == "__main__":
    trainer = tmain()
    trainer.main()
    runner = Runner()
    runner.main()