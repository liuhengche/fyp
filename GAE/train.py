import os
import random
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
from torch.utils.data import SubsetRandomSampler
import logging
import torch.nn as nn


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', filename='training_gcn_norm.log', filemode='a')
# loguru.logger.add("training.log", rotation="1 MB", level="INFO", backtrace=True, diagnose=True)
# Constants
DEMAND_TYPES = ['high', 'low', 'mid']
NUM_EXPERIMENTS = 1000
SELECTED_DETECTORS_PATH = r"D:\desktop\ta\202504\dataset\filtered_edges.txt"
BASE_PATH = r"D:\desktop\ta\202504\dataset"
MODEL_PATH = "best_model.pt"
class ModelSaver:
    """Handles model saving based on validation performance"""
    def __init__(self, model, path=MODEL_PATH):
        self.model = model
        self.path = path
        self.best_loss = float('inf')
        
    def check_and_save(self, val_loss, new_path=MODEL_PATH):
        self.path = new_path
        if val_loss < self.best_loss:
            self.best_loss = val_loss
            torch.save(self.model.state_dict(), self.path)
            # print(f"Saved new best model with avg validation huber loss: {val_loss:.4f}")
        return self.best_loss

class Main:
    def __init__(self):
        pass

    def loss_recon_demand(self, data: HeteroData, alphas: torch.Tensor):
        """calculate reconstruction loss for demand features."""
        x = data['od'].veh_num
        y = data['detector_time'].link_count
        assert isinstance(x, torch.Tensor) and isinstance(y, torch.Tensor), "Demand features must be tensors"

        num_node_u = data['od'].num_nodes
        num_node_v = data['detector_time'].num_nodes

        # Ensure y is a 1D tensor
        y = y.flatten()

        # edges
        adj = data['detector_time', 'rev_assignment', 'od'].edge_index
        assert adj.shape[-1] == alphas.shape[0], "Edge indices and alphas must match in size"

        sparse_alpha = torch.sparse_coo_tensor(indices=adj, values=alphas,
                                               size=(num_node_v, num_node_u),
                                               dtype=x.dtype, device=x.device)
        dense_alpha = sparse_alpha.to_dense()

        # pseudoinverse operation
        sparse_alpha_pinv = dense_alpha.pinverse()
    
        x_pred = torch.mm(sparse_alpha_pinv, y.unsqueeze(-1)).squeeze(-1)
        x.squeeze_(-1)  # Ensure x is 2D for loss calculation
        assert x_pred.shape == x.shape, f"Predicted shape {x_pred.shape} does not match original shape {x.shape}"
        
        # RMSE loss
        criterion_mse = nn.MSELoss(reduction='mean')
        rmse_loss = torch.sqrt(criterion_mse(x_pred, x) + 1e-6)
        # checking sparse_alpha_pinv norm 
        with open("matrix_norm.txt", "a") as f:
            f.write(f"Norm of sparse_alpha_pinv: {sparse_alpha_pinv.norm().item()}, RMSE value is: {rmse_loss}\n")
        return rmse_loss



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
        return self.huber_loss(pred=y_pred, target=y, delta=10.0)
    
    def huber_loss(self, pred: torch.Tensor, target: torch.Tensor, delta: float = 0.25):
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
    
    def main(self):
        # Load full dataset
        dataset = CachedExperimentDataset(
            base_path=BASE_PATH,
            demand_types=DEMAND_TYPES,
            num_experiments=NUM_EXPERIMENTS,
            processed_dir='processed_data',
            use_single_cache=True,
            force_rebuild=False
        )

        # Generate train/test indices
        train_indices = []
        test_indices = []

        for i in range(3):
            start = i * 1000
            end = start + 1000
            part_indices = list(range(start, end))
            random.shuffle(part_indices)
            split_idx = int(0.9 * len(part_indices))
            train_part = part_indices[:split_idx]
            test_part = part_indices[split_idx:]
            train_indices.extend(train_part)
            test_indices.extend(test_part)

        # Create samplers
        train_sampler = SubsetRandomSampler(train_indices)
        test_sampler = SubsetRandomSampler(test_indices)

        # Create DataLoaders
        train_dataloader = DataLoader(dataset, batch_size=1, sampler=train_sampler)
        test_dataloader = DataLoader(dataset, batch_size=1, sampler=test_sampler)

        # Setup device and model
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = Model(hidden_channels=64, out_channels=64).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.0003)
        model_saver = ModelSaver(model)

        model_path = MODEL_PATH
        if os.path.exists(model_path):
            checkpoint = torch.load(model_path)
            model.load_state_dict(checkpoint['model_state_dict'])
            model_saver.best_loss = checkpoint['best_loss']
            print(f"Resuming from best model with loss: {model_saver.best_loss:.4f}")
        else:
            print("No checkpoint found. Starting fresh training.")

        logging.info(f"Training started at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        logging.info(f"Model path: {model_path}")
        logging.info("Training results:")
        train_alpha = []
        # Training Loop
        for epoch in tqdm(range(1, 2), desc="Training Progress", leave=False):
            model.train()
            train_loss = 0.0
            huber_loss_sum = 0.0
            num_samples = 0

            for data in tqdm(train_dataloader, desc=f"Epoch {epoch}", leave=False):
            # for data in train_dataloader:
                data = data.to(device)
                data = ToUndirected()(data)

                # Add metapaths
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

                # Forward pass
                optimizer.zero_grad()
                out = model(
                    data.x_dict,
                    data.edge_index_dict,
                    data['od', 'assignment', 'detector_time'].edge_index
                )

                assert self.loss_recon_node(data, data['od', 'assignment', 'detector_time'].edge_label) == 0, f"Huber loss should be zero if alpha is ground truth"
                
                # Loss calculation
                huber_loss = self.loss_recon_node(data, out)
                loss = F.mse_loss(out, data['od', 'assignment', 'detector_time'].edge_label)
                train_alpha.append(loss.item())
                loss += huber_loss
                # Backward pass
                loss.backward()
                optimizer.step()

                train_loss += loss.item()
                huber_loss_sum += huber_loss.item()
                num_samples += 1

            # Evaluate on test set
            model.eval()
            test_loss = 0.0
            test_huber = 0.0
            test_samples = 0

            with torch.no_grad():
                # for data in tqdm(test_dataloader, desc=f"Testing Epoch {epoch}", leave=False):
                for data in test_dataloader:
                    data = data.to(device)
                    data = ToUndirected()(data)
                    data = AddMetaPaths(metapaths=metapaths)(data)

                    _, edge_weight = gcn_norm(
                        data['detector_time', 'metapath_0', 'detector_time'].edge_index,
                        num_nodes=data['detector_time'].num_nodes,
                        add_self_loops=False
                    )
                    edge_index_metapath = data['detector_time', 'metapath_0', 'detector_time'].edge_index[:, edge_weight > 0.01] 
                    data['detector_time', 'metapath_0', 'detector_time'].edge_index = edge_index_metapath

                    out = model(
                        data.x_dict,
                        data.edge_index_dict,
                        data['od', 'assignment', 'detector_time'].edge_index
                    )

                    huber = self.loss_recon_node(data, out)
                    loss = F.mse_loss(out, data['od', 'assignment', 'detector_time'].edge_label).sqrt().item()
                    test_loss += loss
                    test_huber += huber.item()
                    test_samples += 1

            # Save best model
            avg_train_loss = train_loss / num_samples
            avg_test_loss = test_loss / test_samples
            avg_test_huber = test_huber / test_samples
            
            if os.path.exists(model_path):
                try:
                    checkpoint = torch.load(model_path)
                    # if avg_test_huber < checkpoint.get('best_loss', float('inf')):
                    if avg_test_huber < best_loss:
                        best_loss = model_saver.check_and_save(avg_test_huber)
                        torch.save({
                            'model_state_dict': model.state_dict(),
                            'best_loss': model_saver.check_and_save(avg_test_huber),
                            'avg_test_huber': avg_test_huber,
                        }, model_path)
                        logging.info(f"Updated model with improved Huber loss: {avg_test_huber:.4f} at epoch {epoch}")
                except Exception as e:
                    print(f"Error saving model: {e}")
            else:
                torch.save({
                    'model_state_dict': model.state_dict(),
                    'best_loss': model_saver.check_and_save(avg_test_huber),
                    'avg_test_huber': avg_test_huber,
                }, model_path)
                print(f"\nSaved initial model with Huber loss: {avg_test_huber:.4f}")
                best_loss = avg_test_huber

            # Logging
            print(f'\nEpoch: {epoch:04d} | Train Loss: {avg_train_loss:.4f} | Test Loss: {avg_test_loss:.4f} | Test Huber: {avg_test_huber:.4f}')

            logging.info(f'Epoch: {epoch:04d}, Train Loss: {avg_train_loss:.4f}, Test Loss: {avg_test_loss:.4f}, Test Huber: {avg_test_huber:.4f}')
           

if __name__ == "__main__":
    runner = Main()
    runner.main()