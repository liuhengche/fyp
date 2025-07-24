import os
from tqdm import tqdm
from scripts.base import Edge, Route
from utils import load_from_pkl, save_as_pkl

MAX_INF = 0x7ffff

class DynamicNetworkLoading(object):
    def __init__(self, config): 
        self.routes = load_from_pkl(pkl_path=os.path.join(config['PWD'], config['ROUTE_CHOICE_SAVE_PATH']))
        self.links  = load_from_pkl(pkl_path=os.path.join(config['PWD'], config['EDGE_DICT_PATH']))
        self.v      = load_from_pkl(pkl_path=os.path.join(config['PWD'], config['LINK_MEAN_SPEED_PATH']))
        self.v_     = load_from_pkl(pkl_path=os.path.join(config['PWD'], config['NETWORK_MEAN_SPEED_PATH']))
        self.K = [i for i in range(config['TIME_INTERVAL_N'])]
        self.T = config['TIME_INTERVAL']
        
        self.rt = {}
        self.t = {}
        self.n = {}
        self.gamma = {}

        self.v_path = os.path.join(config['PWD'], config['LINK_MEAN_SPEED_PATH'])
        self.n_path = os.path.join(config['PWD'], config['NETWORK_LOADING_N_PATH'])
        self.t_path = os.path.join(config['PWD'], config['TRAVEL_TIME_PATH'])
        self.gamma_path = os.path.join(config['PWD'], config['NETWORK_LOADING_GAMMA_PATH'])

    def _check_v(self, l, k, k_):
        arrive_k = k + k_
        # time out of domain 
        if arrive_k not in self.K: 
            if (l, k) not in self.v or self.v[l, k] == 0:
                assert self.v_[k] > 0 
                return self.v_[k]
            return self.v[l, k]
        # no traffic volume => return network mean speed
        if (l, arrive_k) not in self.v or self.v[l, arrive_k] == 0: 
            assert self.v_[arrive_k] > 0
            return self.v_[arrive_k]
        # return link mean speed
        return self.v[l, arrive_k]

    def run_network_loading(self, verbose=True):
        # calculate travel time
        bar = tqdm(self.routes.items(), desc='calculate travel time...'.ljust(30)) if verbose else self.routes.items()
        min_r = MAX_INF
        for od_pair, routes in bar:
            if verbose: 
                bar.set_postfix(od_pair=od_pair)
            if min_r >= len(routes): 
                min_od, min_r = od_pair, len(routes)
            for r, route in enumerate(routes):
                for k in self.K: 
                    travel_time = 0.0
                    length = len(route.get_links())
                    for idx, l in enumerate(route.get_links()):
                        # route travel time
                        if idx == length - 1: 
                            self.rt[od_pair, r, k] = travel_time
                        # update t
                        self.t[od_pair, r, l, k] = travel_time
                        x = self.links[l].get_len()
                        v = self._check_v(l=l, k=k, k_=int(travel_time/self.T))
                        travel_time += (x / v) * 3600 # h => s
                        self.v[l, k] = v
        # network loading
        self.n     = dict.fromkeys(self.t.keys())
        self.gamma = dict.fromkeys(self.t.keys())
        bar = tqdm(self.t.items(), desc=f'run network loading...'.ljust(30)) if verbose else self.t.items()
        for key, val in bar: 
            tmp = (val / self.T)
            self.n[key] = int(tmp)
            self.gamma[key] = tmp - self.n[key]
            # assert self.gamma[key] > 0 and self.gamma[key] < 1
        # save data
        save_as_pkl(self.v, pkl_path=self.v_path) # override link mean speeds
        save_as_pkl(self.gamma, pkl_path=self.gamma_path)
        save_as_pkl(self.n, pkl_path=self.n_path)
        save_as_pkl(self.rt, pkl_path=self.t_path)
        return min_od, min_r