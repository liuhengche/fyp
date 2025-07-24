import pickle
import os

from odDataset import DetectorDataProcessor

# Constants
DEMAND_TYPES = ['high', 'low', 'mid']
NUM_EXPERIMENTS = 1000
SELECTED_DETECTORS_PATH = r"D:\desktop\ta\202504\dataset\filtered_edges.txt"
BASE_PATH = r"D:\desktop\ta\202504\dataset"

selected_detectors = []
try:
    with open(SELECTED_DETECTORS_PATH, "r") as f:
        for _ in range(4): next(f)
        for _ in range(321):
            line = f.readline().strip()
            if line: selected_detectors.append(line)
except Exception as e:
    print(f"Error loading detectors: {e}")
detector_processor = DetectorDataProcessor(selected_detectors)

class OD_index_to_id:
    def __init__(self):

        self.od_id_to_index = {f"{i}-{j}-{k}": idx 
        for idx, (i, j, k) in enumerate(
            [(i,j,k) for i in range(1,8) 
            for j in range(1,8) if i != j 
            for k in range(20)])}
        
        self.od_index_to_id = {v: k for k, v in self.od_id_to_index.items()}

    def get_i_j_k(self, od_index):
        i_j_k = self.od_index_to_id[od_index]
        i, j, k = i_j_k.split('-')
        return int(i), int(j), int(k)
    

class Detector_index_to_id:

    def __init__(self, detector_path, save_name):
        self.detector_processor = DetectorDataProcessor(selected_detectors)
        self.detector_path = detector_path
        self.save_name = save_name
        self.index_to_id = self._detector_handling(detector_path)

    def _pickle_save(self, save_name, detector_path):
        os.makedirs("detector_mapping", exist_ok=True)
        filename = f"{save_name}.pkl"
        index_to_id = self._detector_handling(detector_path)
        with open(os.path.join("detector_mapping", filename), "wb") as f:
            pickle.dump(index_to_id, f)

    def _detector_handling(self, detector_path):
        detector_groups = detector_processor.process(detector_path) 
        index_to_id = {}
        detector_time_cache = {}
        for (detector_id, interval), group in detector_groups:
            dt_id = f"{detector_id}_{interval}"
            index_to_id[len(detector_time_cache)] = dt_id
            detector_time_cache[dt_id] = len(detector_time_cache)
        return index_to_id

    def get_detector_interval(self, detector_index):
        edge, interval = self.index_to_id[detector_index].split('_')
        return edge, int(interval)
    




