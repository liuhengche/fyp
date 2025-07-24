import os
import numpy as np
from tqdm import tqdm
from scripts.base import Edge, Route
from math import fmod
from statistics import mean
from utils import sum_df, softmax_df, load_from_pkl, save_as_pkl
# from data_preprocess import DataLoader_v2, load_graph_from_osm, load_detector_from_xml, calc_tt_v2
from data import DataLoader, load_graph_from_osm


class DynamicTrafficAssignment(): 
    def __init__(self, config):
        self.nfd_dot_time_interval = config['NFD_DOT_TIME_INTERVAL']
        self.time_interval = config['TIME_INTERVAL']

        self.tt_path    = os.path.join(config['PWD'], config['TRAVEL_TIME_PATH'])
        self.route_path = os.path.join(config['PWD'], config['ROUTE_CHOICE_DICT_PATH'])
        self.edge_path  = os.path.join(config['PWD'], config['EDGE_DICT_PATH'])

        self.network_loading_gamma_path = os.path.join(config['PWD'], config['NETWORK_LOADING_GAMMA_PATH'])
        self.network_loading_n_path     = os.path.join(config['PWD'], config['NETWORK_LOADING_N_PATH']) 
        self.route_choice_p_path        = os.path.join(config['PWD'], config['ROUTE_CHOICE_P_PATH']) 


        # self.routes = {}
        # self.links  = {}
        # self.tt     = {} # ijrl (kd)
        self.cf     = {} # ijr (kd)
        self.p      = {} # ijr (kd)
        self.phi    = {} # ij (kd)
        self.gamma  = {} # ijrl (kd)
        self.n      = {} # ijrl (kd)

        self.routes = load_from_pkl(self.route_path)
        self.links  = load_from_pkl(self.edge_path)
        self.tt     = load_from_pkl(self.tt_path) # TODO: [deprecated]
        self.od_pairs = self.routes.keys()
    # TODO: [deprecated]
    def run_route_choice(self):
        ''' route choice model '''
        def _calc_phi(self, verbose=False, save=False): 
            ''' phi '''
            bar = tqdm(self.od_pairs, desc=f'calculate phi...'.ljust(30)) if verbose else self.od_pairs
            for od_pair in bar:
                # sum up over r
                avg_list = []
                r = 0
                for _ in self.routes[od_pair]:
                    avg_list.append(self.tt[(od_pair, r)][-1])
                    r += 1
                tmp = sum_df(avg_list)
                tmp = tmp.map(lambda x: r/x)
                self.phi[od_pair] = tmp 
            if save: 
                pass
                # TODO: [Deprecated]
                # save_as_pkl(self.phi, 'average_tt_phi.pkl')
        def _calc_cf(self, verbose=False, save=True): 
            ''' cf (and sigma)'''
            self.cf = dict.fromkeys(self.tt.keys(), 0.0)
            
            bar = tqdm(self.od_pairs, desc=f'calculate cf...'.ljust(30)) if verbose else self.od_pairs
            for od_pair in bar: 
                for r, route in enumerate(self.routes[od_pair]): 
                    # sum up over ijr
                    for link in route.get_links():
                        ll = self.links[link].get_len()
                        rl = route.get_route_len()
                        assert (ll/rl) > 0 and (ll/rl) < 1
                        # indicator
                        sigma = 0
                        for _, h in enumerate(self.routes[od_pair]): 
                            if link in h.get_links(): 
                                sigma += 1
                        # TODO: modify #route 
                        assert sigma > 0 and sigma <= 3
                        # cf
                        self.cf[(od_pair, r)] += (ll / rl) * np.log(sigma)
            if save: 
                pass
                # TODO: [Deprecated]
                # save_as_pkl(self.sigma, 'toy\\bin_indicator_sigma.pkl')
        def _calc_p(self, verbose=False, save=True): 
            self.p = dict.fromkeys(self.tt.keys())
            bar = tqdm(self.od_pairs, desc=f'calculate matrix p...'.ljust(30)) if verbose else self.od_pairs
            for od_pair in bar: 
                r = 0
                softmax_list = []
                for _ in self.routes[od_pair]:
                    tmp = self.phi[od_pair] * self.tt[(od_pair, r)][-1]
                    tmp = tmp.map(lambda x: np.exp( - (x + self.cf[(od_pair, r)])))
                    # tmp = tmp.map(lambda x: np.exp(-x))
                    softmax_list.append(tmp)
                    r += 1
                # [1, ..., r] |R| dataframes (|D||K|)
                for idx, data in enumerate(softmax_df(softmax_list)): 
                    self.p[(od_pair, idx)] = data 
            if save: 
                save_as_pkl(self.p, self.route_choice_p_path)
        print('run route choice model...')
        _calc_phi(self, verbose=False)
        _calc_cf(self,  verbose=False)
        _calc_p(self,   verbose=False)

    # TODO: [deprecated]
    def run_network_loading(self, verbose=True, save=True):
        self.gamma = dict.fromkeys(self.tt.keys())
        self.n = dict.fromkeys(self.tt.keys())
        bar = tqdm(self.tt.items(), desc=f'run network loading...'.ljust(30)) if verbose else self.tt.items()
        for key, val in bar: 
            self.gamma[key] = []
            self.n[key]     = []
            for subroute_tt in val: 
                # t_ijrl
                tmp_ijrl   = subroute_tt.map(lambda x: (x/self.nfd_dot_time_interval))
                gamma_ijrl = tmp_ijrl.map(lambda x: fmod(x, 1))
                n_ijrl     = tmp_ijrl - gamma_ijrl
                n_ijrl     = n_ijrl.map(lambda x: int(x)) # TODO: n = 0, 1
                # flag       = (n_ijrl <= 1)
                # assert flag.all().all(), 'n_ijrl is not in [0, 1]'
                self.gamma[key].append(gamma_ijrl)
                self.n[key].append(n_ijrl)
        # save gamma  
        if save: 
            save_as_pkl(self.gamma, self.network_loading_gamma_path)
            save_as_pkl(self.n, self.network_loading_n_path)

    def test(self): 
        pass

class DynamicTrafficAssignment_v2(DynamicTrafficAssignment): 
    def __init__(self, config):
        super().__init__(config)
        self.link_mean_speed_path = os.path.join(config['PWD'], config['NETWORK_MEAN_SPEED_PATH'])
        self.link_mean_speed = load_from_pkl(self.link_mean_speed_path)[config['WORK_DATE']]
        self.k = [i for i in range(config['N_TIME_INTERVAL'])] # 0, 1, 2, ...
        self.travel_time = {} # override tt
        self.alpha       = {}

        print(self.link_mean_speed.keys())
        assert False

    def run_route_choice(self):
        ''' route choice model '''

        def _calc_tt(self, verbose=False): 
            ''' tt '''
            bar = tqdm(self.routes.items()) if verbose else self.routes.items()
            for od_pair, routes in bar:
                if verbose: 
                    bar.set_description(f"OD Pair {od_pair}")
                for r, route in enumerate(routes):
                    for k in self.k: 
                        travel_time = 0.0
                        for link in route.get_links(): 
                            l = self.links[link].get_len()
                            v = self.link_mean_speed[link, k]
                            assert v > 0
                            travel_time += (l / v) * 3600 # h => s
                            self.travel_time[od_pair, r, link, k] = travel_time
            tt_max_key = max(self.travel_time, key=self.travel_time.key)
            return self.travel_time[tt_max_key]
        
        def _calc_phi(self, verbose=False): 
            ''' phi '''
            bar = tqdm(self.routes.items()) if verbose else self.routes.items()
            for od_pair, routes in bar:
                if verbose: 
                    bar.set_description(f"OD Pair {od_pair}")
                for k in self.k: 
                    avg_list = []
                    for r, route in enumerate(routes):
                        last_link = route.get_last_link()
                        avg_list.append(self.tt[od_pair, r, last_link, k])
                    self.phi[od_pair, k] = mean(avg_list)
        
        def _calc_cf(self, verbose=False): 
            ''' cf (and sigma)'''
            bar = tqdm(self.routes.items()) if verbose else self.routes.items()
            for od_pair, routes in bar:
                if verbose: 
                    bar.set_description(f"OD Pair {od_pair}")
                for r, route in enumerate(routes):
                    for link in route.get_links():
                        ll = self.links[link].get_len()
                        rl = route.get_route_len()
                        assert (ll/rl) > 0 and (ll/rl) < 1
                        # indicator
                        sigma = 0
                        for h in self.routes[od_pair]: 
                            if link in h.get_links(): 
                                sigma += 1
                        assert sigma > 0 and sigma <= 3
                        # cf
                        self.cf.setdefault((od_pair, r), 0.0)
                        self.cf[(od_pair, r)] += (ll / rl) * np.log(sigma)

        def _calc_p(self, verbose=True): 
            def calc_softmax(lis):
                new_lis = [np.exp(i) for i in lis]
                lis_sum = sum(new_lis)
                return [(i / lis_sum) for i in new_lis]
            bar = tqdm(self.routes.items(), desc=f'calculate p_ijrk...'.ljust(30)) if verbose else self.routes.items()
            for od_pair, routes in bar:
                for k in self.k: 
                    if verbose: 
                        bar.set_description(f"(i, j)={od_pair}, k={k})")
                    softmax_list = []
                    for r, route in enumerate(routes):
                        last_link = route.get_last_link()
                        x = self.phi[od_pair, k] * self.travel_time[od_pair, r, last_link, k]
                        softmax_list.append( -(x + self.cf[od_pair, r]) )
                    for r, val in enumerate(calc_softmax(softmax_list)): 
                        self.p[od_pair, r, k] = val
                        for l in self.links.keys(): 
                            self.alpha[od_pair, r, l, k] = self.p[od_pair, r, k] if l in self.routes[od_pair][r].get_links() else 0
            save_as_pkl(self.alpha, pkl_path=self.route_choice_p_path)
        
        tt_max = _calc_tt(self, verbose=False)
        _calc_phi(self, verbose=False)
        _calc_cf(self, verbose=False)
        _calc_p(self, verbose=True)
        return tt_max

    def run_network_loading(self, verbose=True):
        self.gamma = dict.fromkeys(self.tt.keys())
        self.n     = dict.fromkeys(self.tt.keys())
        bar = tqdm(self.travel_time.items(), desc=f'run network loading...'.ljust(30)) if verbose else self.tt.items()
        for key, val in bar: 
            tmp = (val / self.time_interval)
            self.gamma[key] = fmod(tmp, 1)
            self.n[key]     = int(tmp - self.gamma[key])
        save_as_pkl(self.gamma, self.network_loading_gamma_path)
        save_as_pkl(self.n,     self.network_loading_n_path)