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

# Constants
DEMAND_TYPES = ['high', 'low', 'mid']
NUM_EXPERIMENTS = 1000
SELECTED_DETECTORS_PATH = r"D:\desktop\ta\202504\dataset\filtered_edges.txt"
BASE_PATH = r"D:\desktop\ta\202504\dataset"

class ModelSaver:
    """Handles model saving based on validation performance"""
    def __init__(self, model, path="best_model.pt"):
        self.model = model
        self.path = path
        self.best_loss = float('inf')
        
    def check_and_save(self, val_loss, new_path="best_model.pt"):
        self.path = new_path
        if val_loss < self.best_loss:
            self.best_loss = val_loss
            torch.save(self.model.state_dict(), self.path)
            print(f"Saved new best model with loss: {val_loss:.4f}")
        return self.best_loss

class main:
    def __init__(self):
        pass

    def main():
        # Initialize dataset and dataloader
        # dataset = ExperimentDataset(BASE_PATH, DEMAND_TYPES, NUM_EXPERIMENTS)
        # if len(dataset) == 0:
        #     raise RuntimeError("No valid graphs were built. Check file paths and data integrity.")
        
        # dataloader = DataLoader(dataset, batch_size=1, shuffle=True, follow_batch=['od', 'detector_time'])
        dataset = CachedExperimentDataset(
            base_path=BASE_PATH,
            demand_types=DEMAND_TYPES,
            num_experiments=NUM_EXPERIMENTS,
            processed_dir='processed_data',
            use_single_cache=True,
            force_rebuild=False
        )
        
        # Create dataloader
        dataloader = DataLoader(dataset, batch_size=1, shuffle=True, follow_batch=['od', 'detector_time'])

        # Initialize model components
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {device}")
        model = Model(hidden_channels=64, out_channels=64).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.0003)
        model_saver = ModelSaver(model)
        
        # Resume from checkpoint if available
        model_path = "best_model.pt"
        if os.path.exists(model_path):
            checkpoint = torch.load(model_path)
            model.load_state_dict(checkpoint['model_state_dict'])
            model_saver.best_loss = checkpoint['best_loss']
            print(f"Resuming from best model with loss: {model_saver.best_loss:.4f}")
            with open("training_results.txt", 'a') as f:
                f.write("Current datetime: " + str(datetime.now()) + "\n")
                f.write("Resuming from best model with loss: " + str(model_saver.best_loss) + "\n")
        else:
            print("No checkpoint found. Starting fresh training.")
            with open("training_results.txt", 'a') as f:
                f.write("Current datetime: " + str(datetime.now()) + "\n")
                f.write("Starting fresh training.\n")

        with open("training_results.txt", 'a') as f:
            f.write("Training results:\n")
            f.write("Epoch, Loss, Train RMSE, Val RMSE, Test RMSE\n")
        # Training loop
        for epoch in tqdm(range(1, 701), desc="Training Progress"):
            train_loss = 0.0
            train_rmse = 0.0
            val_rmse = 0.0
            test_rmse = 0.0
            num_samples = 0

            for data in tqdm(dataloader, desc=f"Epoch {epoch}", leave=False):
                data = data.to(device)

                # Add reverse edges and metapaths
                data = ToUndirected()(data)
                del data['detector_time', 'rev_assignment', 'od'].edge_label

                # Split data
                transform = RandomLinkSplit(
                    num_val=0.1,
                    num_test=0.1,
                    neg_sampling_ratio=0.0,
                    edge_types=[('od', 'assignment', 'detector_time')],
                    rev_edge_types=[('detector_time', 'rev_assignment', 'od')],
                )
                train_data, val_data, test_data = transform(data)

                # Add metapaths
                metapaths = [[('detector_time', 'rev_assignment', 'od'), ('od', 'assignment', 'detector_time')]]
                train_data = AddMetaPaths(metapaths=metapaths)(train_data)

                # GCN normalization
                _, edge_weight = gcn_norm(
                    train_data['detector_time', 'metapath_0', 'detector_time'].edge_index,
                    num_nodes=train_data['detector_time'].num_nodes,
                    add_self_loops=False,
                )
                edge_index_metapath = train_data['detector_time', 'metapath_0', 'detector_time'].edge_index[:, edge_weight > 0.002]
                for data_split in [train_data, val_data, test_data]:
                    data_split['detector_time', 'metapath_0', 'detector_time'].edge_index = edge_index_metapath

                # Training step
                model.train()
                optimizer.zero_grad()
                out = model(
                    train_data.x_dict,
                    train_data.edge_index_dict,
                    train_data['od', 'assignment', 'detector_time'].edge_label_index,
                )
                loss = F.mse_loss(out, train_data['od', 'assignment', 'detector_time'].edge_label)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
                num_samples += 1

                # Evaluation steps remain unchanged
                model.eval()
                with torch.no_grad():
                    train_out = model(
                        train_data.x_dict,
                        train_data.edge_index_dict,
                        train_data['od', 'assignment', 'detector_time'].edge_label_index,
                    ).clamp(min=0)
                    train_rmse += F.mse_loss(train_out, train_data['od', 'assignment', 'detector_time'].edge_label).sqrt().item()

                    val_out = model(
                        val_data.x_dict,
                        val_data.edge_index_dict,
                        val_data['od', 'assignment', 'detector_time'].edge_label_index,
                    ).clamp(min=0)
                    val_rmse += F.mse_loss(val_out, val_data['od', 'assignment', 'detector_time'].edge_label).sqrt().item()

                    test_out = model(
                        test_data.x_dict,
                        test_data.edge_index_dict,
                        test_data['od', 'assignment', 'detector_time'].edge_label_index,
                    ).clamp(min=0)
                    test_rmse += F.mse_loss(test_out, test_data['od', 'assignment', 'detector_time'].edge_label).sqrt().item()

            # Average metrics
            avg_loss = train_loss / num_samples
            avg_train_rmse = train_rmse / num_samples
            avg_val_rmse = val_rmse / num_samples
            avg_test_rmse = test_rmse / num_samples

            # save best model
            # try:
            #     model_saver.check_and_save(val_loss=avg_val_rmse, new_path="best_model.pt")
            # except Exception as e:
            #     print(f"Error saving model: {e}")
            if os.path.exists(model_path):
                try:
                    checkpoint = torch.load(model_path)
                    if avg_val_rmse < checkpoint.get('val_rmse', float('inf')):
                        torch.save({
                            'model_state_dict': model.state_dict(),
                            'best_loss': model_saver.check_and_save(avg_val_rmse),
                            'epoch': epoch,
                            'val_rmse': avg_val_rmse,
                            'test_rmse': avg_test_rmse,
                            'loss': avg_loss,
                        }, model_path)
                        print(f"Updated best model with improved validation RMSE: {avg_val_rmse:.4f}")
                        with open("training_results.txt", 'a') as f:
                            f.write(f"Updated best model with improved validation RMSE: {avg_val_rmse:.4f}\n")
                except Exception as e:
                    print(f"Error updating model: {e}")
            else:
                torch.save({
                    'model_state_dict': model.state_dict(),
                    'best_loss': model_saver.check_and_save(avg_val_rmse),
                    'epoch': epoch,
                    'val_rmse': avg_val_rmse,
                    'test_rmse': avg_test_rmse,
                    'loss': avg_loss,
                }, model_path)
                with open("training_results.txt", 'a') as f:
                    f.write(f"Saved initial model with loss: {avg_loss:.4f}\n")
                print(f"Saved initial model with loss: {avg_loss:.4f}")

            # Log results
            with open("training_results.txt", 'a') as f:
                f.write(f'Epoch: {epoch:04d}, Loss: {avg_loss:.4f}, Train: {avg_train_rmse:.4f}, Val: {avg_val_rmse:.4f}, Test: {avg_test_rmse:.4f}\n')
            print(f'Epoch: {epoch:04d}, Loss: {avg_loss:.4f}, Train: {avg_train_rmse:.4f}, Val: {avg_val_rmse:.4f}, Test: {avg_test_rmse:.4f}')

