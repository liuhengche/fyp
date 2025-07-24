import os
import argparse

from dnl import DynamicNetworkLoading
from data import DataLoader, DataLoader_StrategyMajorRoad, DataLoader_E1_all, load_graph_from_osm, load_demand_gt_from_csv
from utils import load_config, append_config, load_from_pkl, save_as_pkl

def check_config_integrity(config): 
    try: 
        # working directory
        pwd = config['PWD']
        assert os.path.exists(pwd)
        # domain
        time_interval   = config['TIME_INTERVAL']
        time_interval_n = config['TIME_INTERVAL_N']
        origin_n        = config['ORIGIN_N']
        destination_n   = config['DESTINATION_N']
        assert time_interval   > 0
        assert time_interval_n > 0
        assert origin_n      > 0
        assert destination_n > 0
        # misc
        space_mean_effective_vehicle_len = config['SPACE_MEAN_EFFECTIVE_VEHICLE_LEN']
        grb_license_file = config['GRB_LICENSE_FILE']
        assert space_mean_effective_vehicle_len > 0
        assert os.path.exists(grb_license_file)
        # data 
        demand_ground_truth_path        = config['DEMAND_GROUND_TRUTH_PATH']
        demand_ground_truth_save_path   = config['DEMAND_GROUND_TRUTH_SAVE_PATH']
        detector_raw_data_path          = config['DETECTOR_RAW_DATA_PATH']
        detector_link_mapping_path      = config['DETECTOR_LINK_MAPPING_PATH']
        detector_link_mapping_save_path = config['DETECTOR_LINK_MAPPING_SAVE_PATH']
        network_mean_speed_path         = config['NETWORK_MEAN_SPEED_PATH']
        link_traffic_count_path         = config['LINK_TRAFFIC_COUNT_PATH']
        link_mean_speed_path            = config['LINK_MEAN_SPEED_PATH']
        link_density_path               = config['LINK_DENSITY_PATH']
        assert demand_ground_truth_path is not None
        assert demand_ground_truth_save_path is not None
        assert detector_raw_data_path is not None
        assert detector_link_mapping_path is not None
        assert detector_link_mapping_save_path is not None
        assert network_mean_speed_path is not None
        assert link_traffic_count_path is not None
        assert link_mean_speed_path is not None
        assert link_density_path is not None
        # dnl
        road_network_path      = config['ROAD_NETWORK_PATH']
        route_choice_path      = config['ROUTE_CHOICE_PATH']
        travel_time_path       = config['TRAVEL_TIME_PATH']
        route_choice_dict_path = config['ROUTE_CHOICE_SAVE_PATH']
        edge_dict_path         = config['EDGE_DICT_PATH']
        network_loading_n_path = config['NETWORK_LOADING_N_PATH']
        network_loading_gamma_path = config['NETWORK_LOADING_GAMMA_PATH']
        assert road_network_path is not None
        assert route_choice_path is not None
        assert travel_time_path is not None
        assert route_choice_dict_path is not None
        assert edge_dict_path is not None
        assert network_loading_n_path is not None
        assert network_loading_gamma_path is not None
        # optimization
        sampler_seed = config['SAMPLER_SEED']
        trials_n     = config['TRIALS_N']
        jobs_n       = config['JOBS_N']
        max_solve_time = config['MAX_SOLVE_TIME']
        assert sampler_seed > 0
        assert trials_n > 0
        assert jobs_n > 0
        assert max_solve_time > 0
        # log
        log_storage  = config['LOG_STORAGE']
        lp_file_path = config['LP_FILE_PATH']
        benchmark_log_path = config['BENCHMARK_LOG_PATH']
        assert log_storage is not None
        assert lp_file_path is not None
        assert benchmark_log_path is not None
        return True 
    except KeyError: 
        return False
    
def check_data_integrity(config): 
    link_traffic_counts   = os.path.join(config['PWD'] , config['LINK_TRAFFIC_COUNT_PATH'])
    link_mean_speed       = os.path.join(config['PWD'] , config['LINK_MEAN_SPEED_PATH'])
    link_density          = os.path.join(config['PWD'] , config['LINK_DENSITY_PATH'])
    # detector_link_mapping = os.path.join(config['PWD'] , config['DETECTOR_LINK_MAPPING_PATH'])
    routes                = os.path.join(config['PWD'] , config['ROUTE_CHOICE_SAVE_PATH'])
    network_loading_gamma = os.path.join(config['PWD'] , config['NETWORK_LOADING_GAMMA_PATH'])
    network_loading_n     = os.path.join(config['PWD'] , config['NETWORK_LOADING_N_PATH'])
    travel_time           = os.path.join(config['PWD'] , config['TRAVEL_TIME_PATH'])
    assert os.path.exists(link_traffic_counts), f'File not found {link_traffic_counts}'
    assert os.path.exists(network_loading_gamma), f'File not found {network_loading_gamma}'
    assert os.path.exists(network_loading_n), f'File not found {network_loading_n}'
    assert os.path.exists(link_mean_speed), f'File not found {link_mean_speed}'
    assert os.path.exists(link_density), f'File not found {link_density}'
    assert os.path.exists(travel_time), f'File not found {travel_time}'
    assert os.path.exists(routes), f'File not found {routes}'

    # load y, gamma, n, ijrl keys
    gamma_  = load_from_pkl(network_loading_gamma) # ijrlk
    n_      = load_from_pkl(network_loading_n)     # ijrlk
    y_hat_  = load_from_pkl(link_traffic_counts)   # l, k
    v_      = load_from_pkl(link_mean_speed)
    d_      = load_from_pkl(link_density)
    t_      = load_from_pkl(travel_time)
    routes_ = load_from_pkl(routes)

    # customize
    # print(y_hat_)

if __name__ == '__main__': 
    # args
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/odme/test_congested_linear.yaml', help='configuration path')
    # parser.add_argument('--real', action='store_true', default=False, help='use simulation data')
    args = parser.parse_args()
    CONFIG = load_config(config_path=args.config)

    # check config integrity 
    assert check_config_integrity(config=CONFIG)

    # preprocess
    # if args.real:
    #     data_loader = DataLoader_StrategyMajorRoad(config=CONFIG)
    # else:  
    #     data_loader = DataLoader(config=CONFIG)
    data_loader = DataLoader_E1_all(config=CONFIG)
    data_loader.run(verbose=False)
    # load road network
    node_n, edge_n, neighbors = load_graph_from_osm(
        osm_path   = os.path.join(CONFIG['PWD'], CONFIG['ROAD_NETWORK_PATH']), 
        route_path = os.path.join(CONFIG['PWD'], CONFIG['ROUTE_CHOICE_PATH']), 
        edge_save_path  = os.path.join(CONFIG['PWD'], CONFIG['EDGE_DICT_PATH']),
        route_save_path = os.path.join(CONFIG['PWD'], CONFIG['ROUTE_CHOICE_SAVE_PATH']), 
    )
    assert len(neighbors.values()) != 0
    save_as_pkl(data=neighbors, pkl_path=os.path.join(CONFIG['PWD'], 'neighbors.pkl')) # TODO: modify config
    # load ground truth
    x_gt = load_demand_gt_from_csv(gt_path=os.path.join(CONFIG['PWD'], CONFIG['DEMAND_GROUND_TRUTH_PATH']), 
                                   save_path=os.path.join(CONFIG['PWD'], CONFIG['DEMAND_GROUND_TRUTH_SAVE_PATH']))
    
    # dta
    dnl = DynamicNetworkLoading(config=CONFIG)
    min_od, min_r = dnl.run_network_loading()
    assert min_r > 1, f'OD {min_od} fails to satisfy minimum route choices!'
    CONFIG = append_config(origin_config=CONFIG, append_config={'ROUTE_N': min_r}, config_save_path=args.config)

    # check data integrity
    check_data_integrity(config=CONFIG)