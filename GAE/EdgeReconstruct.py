import os
from matplotlib import pyplot as plt
import torch
import pandas as pd
from torch_geometric.loader import DataLoader
from torch_geometric.data import HeteroData
from torch_geometric.transforms import ToUndirected, AddMetaPaths
from torch_geometric.nn.conv.gcn_conv import gcn_norm
from scipy.stats import pearsonr
import numpy as np
from tqdm import tqdm
import torch.nn.functional as F
import logging


from odDataset import CachedExperimentDataset
from model import Model
from mappingStore import OD_index_to_id, Detector_index_to_id
# Constants
DEMAND_TYPES = ['high', 'low', 'mid']
NUM_EXPERIMENTS = 1000
SELECTED_DETECTORS_PATH = r"D:\desktop\ta\202504\dataset\filtered_edges.txt"
BASE_PATH = r"D:\desktop\ta\202504\dataset"
detector_path = r'D:\desktop\ta\202504\dataset\qkv\high\high_0.csv'

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', filename='reconstruct.log', filemode='a')


def loss_recon_node(data: HeteroData, alphas: torch.Tensor):
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
    return huber_loss(pred=y_pred, target=y, delta=1.0)

def huber_loss(pred: torch.Tensor, target: torch.Tensor, delta: float = 0.25):
    if pred.dim() == 1:
        pred = pred.unsqueeze(-1)
    if target.dim() == 1:
        target = target.unsqueeze(-1)
    # 
    assert pred.shape == target.shape, f"Prediction and target must have the same shape, prediction shape: {pred.shape}, target shape: {target.shape}"
    # with open('log', 'a') as f:
    #     pred_list = [str(p.item()) for p in pred.flatten()]
    #     target_list = [str(t.item()) for t in target.flatten()]
    #     f.write('prediction, target:\n')
    #     for p, t in zip(pred_list, target_list):
    #         f.write(f"{p}, {t}\n")
    return F.huber_loss(pred, target, reduction='mean', delta=delta)
    

class Reconstruct:
    def __init__(self):
        self.od_to_id = OD_index_to_id()
        self.detector_to_id = Detector_index_to_id(detector_path, "high_0")

    def to_class_index(self, alpha):
        """Convert 0.1–1.0 float to 0–9 index"""
        return torch.clamp((alpha * 10).long() - 1, min=0, max=9)
    
    def compute_class_weights(self, alpha_values):
        class_counts = torch.bincount(self.to_class_index(alpha_values), minlength=10)
        class_weights = class_counts.max() / class_counts.float()
        return class_weights


    # def save_alpha_csv(self):
    #     dataset = CachedExperimentDataset(
    #         base_path=BASE_PATH,
    #         demand_types=DEMAND_TYPES,
    #         num_experiments=NUM_EXPERIMENTS,
    #         processed_dir='processed_data',
    #         use_single_cache=True,
    #         force_rebuild=False
    #     )
    #     dataloader = DataLoader(dataset, batch_size=1, shuffle=False, follow_batch=['od', 'detector_time'])
    #     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    #     print(f"Using device: {device}")
    #     model = Model(hidden_channels=64, out_channels=64).to(device)

    #     # Resume from checkpoint
    #     model_path = "best_model_huber_loss.pt"

    #     if os.path.exists(model_path):
    #         checkpoint = torch.load(model_path, weights_only=False)
    #         model.load_state_dict(checkpoint['model_state_dict'])
    #         best_f1 = checkpoint['best_f1']
    #         print(f"Resuming from best model with F1: {best_f1:.4f}")
    #     else:
    #         assert False, "Model checkpoint not found. Please train the model first."
    
    #     class_values = torch.linspace(0.1, 1.0, 10).to(device)

    #     all_alphas = []
    #     for data in dataset:
    #         all_alphas.extend(data['od', 'assignment', 'detector_time'].edge_label.tolist())
    #     class_weights = self.compute_class_weights(torch.tensor(all_alphas))
    #     class_weights = class_weights.to(device)


    #     model.eval()
    #     demand_types = ['high', 'low', 'mid']
    #     graph_index = 0
    #     os.makedirs("graph_correlations_predictions", exist_ok=True)
    #     save_folder = "graph_correlations_predictions"
    #     for data in tqdm(dataloader, desc="Plotting Graphs", total=len(dataloader)):
    #         data = data.to(device)
    #         data = ToUndirected()(data)
            
    #         # Remove reverse edge label if it exists
    #         if ('detector_time', 'rev_assignment', 'od') in data.edge_types:
    #             del data['detector_time', 'rev_assignment', 'od'].edge_label

    #         # Reconstruct metapath
    #         metapaths = [[('detector_time', 'rev_assignment', 'od'), ('od', 'assignment', 'detector_time')]]
    #         data = AddMetaPaths(metapaths=metapaths)(data)

    #         # GCN normalization for metapath edges
    #         _, edge_weight = gcn_norm(
    #             data['detector_time', 'metapath_0', 'detector_time'].edge_index,
    #             num_nodes=data['detector_time'].num_nodes,
    #             add_self_loops=False,
    #         )
    #         edge_index_metapath = data['detector_time', 'metapath_0', 'detector_time'].edge_index[:, edge_weight > 0.002]
    #         data['detector_time', 'metapath_0', 'detector_time'].edge_index = edge_index_metapath

    #         # Use all edges for prediction
    #         edge_index = data['od', 'assignment', 'detector_time'].edge_index
    #         edge_label = data['od', 'assignment', 'detector_time'].edge_label
    #         edge_attr = data['od', 'assignment', 'detector_time'].edge_attr

    #         # Predict on all edges
    #         with torch.no_grad():
    #             logits = model(data.x_dict, data.edge_index_dict, edge_index).clamp(min=0)
                
    #         predicted_prob = F.softmax(logits, dim=1)
    #         class_values = torch.linspace(0.1, 1.0, 10).to(device)
    #         predicted_values = torch.sum(predicted_prob * class_values, dim=1).cpu().numpy()
    #         true_values = edge_label.cpu().numpy()
    #         pearson_corr, _ = pearsonr(true_values, predicted_values)

    #         # Plot
    #         plt.figure(figsize=(8, 6))
    #         plt.scatter(true_values, predicted_values, alpha=0.6, s=10, label='Predictions')
    #         plt.plot([0, 1], [0, 1], 'r--', color='black', label='Perfect Fit')
    #         plt.xlabel('True Alpha Values')
    #         plt.ylabel('Predicted Alpha Values')
    #         plt.title(f'graph_{demand_types[graph_index//1000]}_{graph_index%1000} | Pearson: {pearson_corr:.4f}')
    #         plt.xlim(0, 1)
    #         plt.ylim(0, 1)
    #         plt.grid(True, linestyle='--', alpha=0.7)
    #         plt.legend()

    #         # Regression line
    #         m, b = np.polyfit(true_values, predicted_values, 1)
    #         plt.plot(true_values, m * true_values + b, color='blue', label='Regression Line')
    #         plt.legend()

    #         # Save and close figure to prevent memory issues
    #         plt.tight_layout()
    #         plt.savefig(f'graph_{demand_types[graph_index//1000]}_{graph_index%1000}.png', dpi=100, bbox_inches='tight')
    #         plt.close()

    #         # Save to CSV
    #         csv_rows = []
    #         for edge_idx in range(edge_index.shape[1]):
    #             od_idx = edge_index[0, edge_idx].item()
    #             dt_idx = edge_index[1, edge_idx].item()
    #             alpha = edge_label[edge_idx].item()
    #             predicted = predicted_values[edge_idx]

    #             # Convert OD and detector indices back to readable format
    #             i,j,k = self.od_to_id.get_i_j_k(od_idx)
    #             detector_id, interval_begin = self.detector_to_id.get_detector_interval(dt_idx)

    #             csv_rows.append({
    #                 "graph_index": graph_index,
    #                 "i": int(i),
    #                 "j": int(j),
    #                 "k": int(k),
    #                 "detector_id": detector_id,
    #                 "interval_begin": int(interval_begin),
    #                 "alpha": alpha,
    #                 "predicted_value": predicted,
    #                 "ground_truth": alpha
    #             })

    #         df = pd.DataFrame(csv_rows)
    #         file_name = os.path.join(save_folder, f"graph_{demand_types[graph_index//1000]}_{graph_index%1000}_correlations.csv")
    #         df.to_csv(file_name, 
    #                  columns=["graph_index", "i", "j", "k", "detector_id", 
    #                          "interval_begin", "alpha", "predicted_value", "ground_truth"],
    #                  index=False) 
    #         print(f"Saved graph {graph_index} correlations to CSV")

            
    #         graph_index += 1
    #         break # for testing
    #     print("All graphs processed and saved.")


    def save_alpha_csv_old(self):
        dataset = CachedExperimentDataset(
            base_path=BASE_PATH,
            demand_types=DEMAND_TYPES,
            num_experiments=NUM_EXPERIMENTS,
            processed_dir='processed_data',
            use_single_cache=True,
            force_rebuild=False
        )
        dataloader = DataLoader(dataset, batch_size=1, shuffle=False, follow_batch=['od', 'detector_time'])
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = Model(hidden_channels=64, out_channels=64).to(device)

        # Resume from checkpoint
        model_path = "best_model_huber_loss.pt"
        if os.path.exists(model_path):
            checkpoint = torch.load(model_path, weights_only=False)
            model.load_state_dict(checkpoint['model_state_dict'])
            print("Resumed model from checkpoint")

        model.eval()
        demand_types = ['high', 'low', 'mid']
        graph_index = 0
        os.makedirs("graph_correlations_predictions", exist_ok=True)
        save_folder = "graph_correlations_predictions"
        huber_loss_sum = 0.0
        num_graphs = 0
        for data in tqdm(dataloader, desc="Plotting Graphs", total=len(dataloader)):
            data = data.to(device)
            data = ToUndirected()(data)
            
            # Remove reverse edge label if it exists
            if ('detector_time', 'rev_assignment', 'od') in data.edge_types:
                del data['detector_time', 'rev_assignment', 'od'].edge_label

            # Reconstruct metapath
            metapaths = [[('detector_time', 'rev_assignment', 'od'), ('od', 'assignment', 'detector_time')]]
            data = AddMetaPaths(metapaths=metapaths)(data)

            # GCN normalization for metapath edges
            _, edge_weight = gcn_norm(
                data['detector_time', 'metapath_0', 'detector_time'].edge_index,
                num_nodes=data['detector_time'].num_nodes,
                add_self_loops=False,
            )
            edge_index_metapath = data['detector_time', 'metapath_0', 'detector_time'].edge_index[:, edge_weight > 0.002]
            data['detector_time', 'metapath_0', 'detector_time'].edge_index = edge_index_metapath

            # Use all edges for prediction
            edge_index = data['od', 'assignment', 'detector_time'].edge_index
            edge_label = data['od', 'assignment', 'detector_time'].edge_label
            edge_attr = data['od', 'assignment', 'detector_time'].edge_attr

            # Predict on all edges
            with torch.no_grad():
                out = model(data.x_dict, data.edge_index_dict, edge_index).clamp(min=0)

            huber_loss_check = loss_recon_node(data, data['od', 'assignment', 'detector_time'].edge_label)
            assert huber_loss_check.item() == 0, f"huber loss should be 0 for ground truth alpha, but got {huber_loss_check.item()}"
            huber_loss = loss_recon_node(data, out)
            huber_loss_sum += huber_loss.item()
            num_graphs += 1
            true_values = edge_label.cpu().numpy()
            predicted_values = out.cpu().numpy()
            pearson_corr, _ = pearsonr(true_values, predicted_values)
            # logging.info(f"Graph {demand_types[graph_index//1000]}_{graph_index%1000}: Huber Loss: {huber_loss.item():.4f}, Pearson Correlation: {pearson_corr:.4f}")
            # # Plot
            plt.figure(figsize=(8, 6))
            plt.scatter(true_values, predicted_values, alpha=0.6, s=10, label='Predictions')
            plt.plot([0, 1], [0, 1], 'r--', color='black', label='Perfect Fit')
            plt.xlabel('True Alpha Values')
            plt.ylabel('Predicted Alpha Values')
            plt.title(f'graph_{demand_types[graph_index//1000]}_{graph_index%1000} | Pearson: {pearson_corr:.4f}')
            plt.xlim(0, 1)
            plt.ylim(0, 1)
            plt.grid(True, linestyle='--', alpha=0.7)
            plt.legend()

            # Regression line
            m, b = np.polyfit(true_values, predicted_values, 1)
            plt.plot(true_values, m * true_values + b, color='blue', label='Regression Line')
            plt.legend()

            # Save and close figure to prevent memory issues
            plt.tight_layout()
            plt.savefig(f'graph_{demand_types[graph_index//1000]}_{graph_index%1000}.png', dpi=100, bbox_inches='tight')
            plt.close()

            # Save the alpha predictions
            csv_rows = []
            for edge_idx in range(edge_index.shape[1]):
                od_idx = edge_index[0, edge_idx].item()
                dt_idx = edge_index[1, edge_idx].item()
                alpha = edge_attr[edge_idx].item()

                i,j,k = self.od_to_id.get_i_j_k(od_idx)
                detector_id, interval_begin = self.detector_to_id.get_detector_interval(dt_idx)

                # print(f"Alpha={alpha}: i-j-k={i}-{j}-{k}, detector={detector_id}, interval={interval_begin}")

                csv_row = {
                    "graph_index": graph_index,
                    "i": int(i),
                    "j": int(j),
                    "k": int(k),
                    "detector_id": detector_id,
                    "interval_begin": int(interval_begin),
                    "alpha": alpha,
                    "predicted_value": out[edge_idx].item(),
                    "ground_truth": edge_label[edge_idx].item() if edge_label is not None else None
                }

                csv_rows.append(csv_row)
            df = pd.DataFrame(csv_rows)
            file_name = os.path.join(save_folder, f"graph_{demand_types[graph_index//1000]}_{graph_index%1000}_correlations.csv")
            df.to_csv(file_name, 
                     columns=["graph_index", "i", "j", "k", "detector_id", 
                             "interval_begin", "alpha", "predicted_value", "ground_truth"],
                     index=False) 
            print(f"Saved graph {graph_index} correlations to CSV")

            
            graph_index += 1
            break # for testing
        logging.info(f"Average Huber Loss: {huber_loss_sum / num_graphs:.4f} over {num_graphs} graphs")
        print("All graphs processed and saved.")

    def reconstruct(self):
        # Placeholder for the reconstruction logic
        # This should be replaced with the actual implementation
        reconstructed_data = self.data  # Replace with actual reconstruction logic
        return reconstructed_data
    



if __name__ == "__main__":
    reconstruct = Reconstruct()
    reconstruct.save_alpha_csv_old()
    # reconstructed_data = reconstruct.reconstruct()
    # print(reconstructed_data)