import os
import pandas as pd
import torch
from torch.utils.data import Dataset
from torch_geometric.data import HeteroData
from tqdm import tqdm

# Constants
DEMAND_TYPES = ['high', 'low', 'mid']
NUM_EXPERIMENTS = 1000
SELECTED_DETECTORS_PATH = r"D:\desktop\ta\202504\dataset\filtered_edges.txt"
BASE_PATH = r"D:\desktop\ta\202504\dataset"

class DetectorDataProcessor:
    """Handles detector data processing with time intervals"""
    def __init__(self, selected_detectors):
        self.selected_detectors = selected_detectors

    def process(self, file_path):
        try:
            df = pd.read_csv(file_path)
            df = df[df['edge'].isin(self.selected_detectors)]
            return df.groupby(['edge', 'interval_begin'])
        except Exception as e:
            print(f"Error processing detector file {file_path}: {e}")
            return []

class ODEncoder:
    """Encodes OD pairs with fixed mapping"""
    def __init__(self):
        self.od_id_to_index = {f"{i}-{j}-{k}": idx 
                              for idx, (i, j, k) in enumerate(
                                  [(i,j,k) for i in range(1,8) 
                                   for j in range(1,8) if i != j 
                                   for k in range(20)])}

    def get_features(self, file_path):
        try:
            df = pd.read_csv(file_path)
            features = torch.zeros((840, 1))
            od_dict = {}
            for _, row in df.iterrows():
                od_str = f"{row['i']}-{row['j']}-{row['k']}"
                if od_str in self.od_id_to_index:
                    od_dict[self.od_id_to_index[od_str]] = row['vehsPerHour']
            for idx, value in od_dict.items():
                features[idx] = value
            return features
        except Exception as e:
            print(f"Error reading OD file {file_path}: {e}")
            return torch.zeros((840, 1))

class GraphBuilder:
    """Builds heterogeneous graph from processed data"""
    def __init__(self, od_encoder, detector_processor):
        self.od_encoder = od_encoder
        self.detector_processor = detector_processor

    def _get_detector_time_id(self, detector_id, interval):
        return f"{detector_id}_{interval}"

    def build(self, od_path, detector_path, alpha_path):
        try:
            # Load OD features
            od_features = self.od_encoder.get_features(od_path)

            # Load and group detector data
            detector_groups = self.detector_processor.process(detector_path)

            # Build detector-time nodes
            detector_time_features = []
            detector_time_cache = {}
            for (detector_id, interval), group in detector_groups:
                dt_id = self._get_detector_time_id(detector_id, interval)
                detector_time_cache[dt_id] = len(detector_time_cache)
                flow = group['edge_flow'].iloc[0]
                density = group['edge_density'].iloc[0]
                speed = group['edge_speed'].iloc[0]
                detector_time_features.append([flow, density, speed])

            # Load alpha data
            alpha_df = pd.read_csv(alpha_path)

            # Build edges
            edges = []
            alphas = []
            for _, row in alpha_df.iterrows():
                dt_id = self._get_detector_time_id(row['edge'], row['interval_begin'])
                if dt_id not in detector_time_cache:
                    continue
                od_str = row['i-j-k']
                if od_str not in self.od_encoder.od_id_to_index:
                    continue
                edges.append([
                    self.od_encoder.od_id_to_index[od_str],
                    detector_time_cache[dt_id]
                ])
                alphas.append(row['alpha'])

            # Create HeteroData object
            data = HeteroData()
            data['od'].x = od_features
            data['detector_time'].x = torch.tensor(detector_time_features, dtype=torch.float)
            edge_index = torch.tensor(edges, dtype=torch.long).T if edges else torch.empty((2, 0), dtype=torch.long)
            edge_attr = torch.tensor(alphas, dtype=torch.float) if alphas else torch.empty(0, dtype=torch.float)
            data['od', 'assignment', 'detector_time'].edge_index = edge_index
            data['od', 'assignment', 'detector_time'].edge_attr = edge_attr
            data['od', 'assignment', 'detector_time'].edge_label = edge_attr

            # print(data)
            # assert False, "Debugging HeteroData object"

            return data
        except Exception as e:
            print(f"Failed to build graph from {od_path}, {detector_path}, {alpha_path}: {e}")
            return None

class CachedExperimentDataset(Dataset):
    """Dataset with individual graph caching (high_0.pt, high_1.pt, etc.)"""
    def __init__(self, base_path, demand_types, num_experiments, 
                 processed_dir='processed_data', 
                 use_single_cache=True,
                 force_rebuild=False):
        super().__init__()
        
        # Initialize parameters
        self.base_path = base_path
        self.demand_types = demand_types
        self.num_experiments = num_experiments
        self.processed_dir = processed_dir
        self.use_single_cache = use_single_cache
        
        # Create directories
        os.makedirs(processed_dir, exist_ok=True)
        self.per_graph_dir = os.path.join(processed_dir, 'per_graph')
        os.makedirs(self.per_graph_dir, exist_ok=True)
        
        # Initialize components
        self.selected_detectors = self._load_selected_detectors()
        self.od_encoder = ODEncoder()
        self.detector_processor = DetectorDataProcessor(self.selected_detectors)
        self.graph_builder = GraphBuilder(self.od_encoder, self.detector_processor)
        
        # File tracking
        self.file_paths = []
        self._collect_file_paths()
        
        # Graph paths
        self.graph_paths = []
        self._find_cached_graphs(force_rebuild)
        
        # Build missing graphs
        if len(self.graph_paths) < len(self.file_paths) or force_rebuild:
            self._build_and_cache_graphs()

    def _load_selected_detectors(self):
        selected = []
        try:
            with open(SELECTED_DETECTORS_PATH, "r") as f:
                for _ in range(4): next(f)
                for _ in range(321):
                    line = f.readline().strip()
                    if line: selected.append(line)
        except Exception as e:
            print(f"Error loading detectors: {e}")
        return selected

    def _collect_file_paths(self):
        """Collect all experiment file paths"""
        for demand in self.demand_types:
            od_dir = os.path.join(self.base_path, 'od', demand)
            detector_dir = os.path.join(self.base_path, 'qkv', demand)
            alpha_dir = os.path.join(self.base_path, 'fcd', demand)
            
            for i in range(self.num_experiments):
                filename = f"{demand}_{i}.csv"
                self.file_paths.append({
                    'od': os.path.join(od_dir, filename),
                    'detector': os.path.join(detector_dir, filename),
                    'alpha': os.path.join(alpha_dir, filename),
                    'name': f"{demand}_{i}"
                })

    def _find_cached_graphs(self, force_rebuild):
        """Find existing cached graphs"""
        for item in self.file_paths:
            graph_path = os.path.join(self.per_graph_dir, f"{item['name']}.pt")
            if not force_rebuild and os.path.exists(graph_path):
                self.graph_paths.append(graph_path)
        if self.graph_paths:
            print(f"Found {len(self.graph_paths)} cached graphs.")

    def _build_and_cache_graphs(self):
        """Build and cache missing graphs"""
        print("Building and caching graphs...")
        for item in tqdm(self.file_paths, desc="Building graphs", total=len(self.file_paths)):
            graph_path = os.path.join(self.per_graph_dir, f"{item['name']}.pt")
            
            if os.path.exists(graph_path):
                continue  # Skip existing graphs
                
            try:
                graph = self.graph_builder.build(item['od'], item['detector'], item['alpha'])
                if graph is not None:
                    torch.save(graph, graph_path)
                    self.graph_paths.append(graph_path)
            except Exception as e:
                print(f"Error building {item['name']}: {e}")
                continue

    def __len__(self):
        return len(self.graph_paths)

    def __getitem__(self, idx):
        """Load graph from disk"""
        return torch.load(self.graph_paths[idx], weights_only=False)
