import os
import numpy as np
import sys
from scripts.utils import load_config, load_from_pkl
from scripts.optimization import DODME_congested
import scripts.base 
import argparse

scripts_dir = os.path.join(os.path.dirname(__file__), 'scripts')
if scripts_dir not in sys.path:
    sys.path.insert(0, scripts_dir)

CONFIG = load_config('config.yaml')

# load data
link_traffic_counts   = os.path.join(CONFIG['PWD'] , CONFIG['LINK_TRAFFIC_COUNT_PATH'])
link_mean_speed       = os.path.join(CONFIG['PWD'] , CONFIG['LINK_MEAN_SPEED_PATH'])
link_density          = os.path.join(CONFIG['PWD'] , CONFIG['LINK_DENSITY_PATH'])
detector_link_mapping = os.path.join(CONFIG['PWD'] , CONFIG['DETECTOR_LINK_MAPPING_PATH'])
routes                = os.path.join(CONFIG['PWD'] , CONFIG['ROUTE_CHOICE_SAVE_PATH'])
network_loading_gamma = os.path.join(CONFIG['PWD'] , CONFIG['NETWORK_LOADING_GAMMA_PATH'])
network_loading_n     = os.path.join(CONFIG['PWD'] , CONFIG['NETWORK_LOADING_N_PATH'])
travel_time           = os.path.join(CONFIG['PWD'] , CONFIG['TRAVEL_TIME_PATH'])
neighbors            = os.path.join(CONFIG['PWD'] , 'neighbors.pkl')
assert os.path.exists(link_traffic_counts), f'File not found {link_traffic_counts}'
assert os.path.exists(network_loading_gamma), f'File not found {network_loading_gamma}'
assert os.path.exists(network_loading_n), f'File not found {network_loading_n}'
assert os.path.exists(link_mean_speed), f'File not found {link_mean_speed}'
assert os.path.exists(link_density), f'File not found {link_density}'
assert os.path.exists(travel_time), f'File not found {travel_time}'
assert os.path.exists(routes), f'File not found {routes}'
assert os.path.exists(neighbors), f'File not found {neighbors}'

sim_args_dict = {
    'duration': 1800, # Default simulation time
    'period': 180, # Default detector work cycle
    'data': 'simulation_test', # Default sensor XML data path
    'seed': 2025, # Default seed for reproducibility
    'config': 'test.sumocfg', # Default SUMO configuration path
    'mute_warnings': True, # Default to not mute warnings
    'mute_step_logs': False # Default to not mute step logs
}

sim_args = argparse.Namespace(**sim_args_dict)
# load y, gamma, n, ijrl keys
gamma_  = load_from_pkl(network_loading_gamma) # ijrlk
n_      = load_from_pkl(network_loading_n)     # ijrlk
y_hat_  = load_from_pkl(link_traffic_counts)   # l, k
v_      = load_from_pkl(link_mean_speed)
d_      = load_from_pkl(link_density)
t_      = load_from_pkl(travel_time)
routes_ = load_from_pkl(routes)
neighbors_ = load_from_pkl(neighbors)


if __name__ == '__main__': 
    # solve
    dodme = DODME_congested(
        config=CONFIG, 
        g=gamma_, 
        n=n_, 
        v=v_, 
        d=d_, 
        t=t_, 
        y=y_hat_,
        neighbor=neighbors_,
        fd={},  
        sim_args=sim_args 
    )
    dodme.generate_model(x_upper=100, verbose=True, presolve=True)
    dodme.set_objective(params=[1, 0])
    dodme.solve()
    dodme.benchmark()



