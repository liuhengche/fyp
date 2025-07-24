import os
import math
import datetime
import numpy as np
import xml.etree.ElementTree as ET
from tqdm import tqdm
from statistics import mean, pstdev
from utils import load_from_pkl, save_as_pkl, add_log
from visual import plot_missing_data_venn, plot_ape, plot_geh
from benchmark import ape, geh, mae, rmse
# gurobi
import gurobipy as gp
from gurobipy import GRB
# highs
import highspy
from highspy import Highs

# TODO: remove
I = [i for i in range(1, 16)]
J = [j for j in range(1, 16)]
DEMAND_MAX = 20
SOLVE_HOUR_MAX = 2
# PROBLEM_TYPE = 'MILP'
# PROBLEM_TYPE = 'MIQP'
PROBLEM_TYPE = 'NLP'
INF = highspy.kHighsInf

class DODME():
    ''' base class '''
    def __init__(self, config): 
        self.link2detector = {}
        self.routes        = {}
        self.model         = None

        # data
        self.date  = config['WORK_DATE'] 
        self.alpha = None
        self.gamma = None
        self.n     = None
        self.y     = {}

        # result
        self.x     = {}
        self.w     = {}
        self.obj   = 0
        
        # set
        self.ijrl  = None
        self.I     = I # hard code
        self.J     = J # hard code
        self.L     = None
        self.K     = [i for i in range(config['N_TIME_INTERVAL'])] # hard code

        self.log_path      = config['BENCHMARK_LOG_PATH']
        self.period        = config['NFD_DOT_TIME_INTERVAL'] # i.e., 4mins
        
        self.link_traffic_counts   = os.path.join(config['PWD'] , f'link_traffic_count_{config['WORK_DATE']}.pkl')
        self.detector_link_mapping = os.path.join(config['PWD'] , config['DETECTOR_LINK_MAPPING_PATH'])
        self.routes                = os.path.join(config['PWD'] , config['ROUTE_CHOICE_DICT_PATH'])
        self.route_choice_p        = os.path.join(config['PWD'] , config['ROUTE_CHOICE_P_PATH'])
        self.network_loading_gamma = os.path.join(config['PWD'] , config['NETWORK_LOADING_GAMMA_PATH'])
        self.network_loading_n     = os.path.join(config['PWD'] , config['NETWORK_LOADING_N_PATH'])

        self.abnormal_id = set()

        # load static info
        assert os.path.exists(self.detector_link_mapping), f'File not found {self.detector_link_mapping}'
        assert os.path.exists(self.routes), f'File not found {self.routes}'
        
        tree = ET.parse(self.detector_link_mapping)
        root = tree.getroot()
        for loop_detector in root.iter('inductionLoop'): 
            id   = loop_detector.get('id')[:-2]
            link = loop_detector.get('lane')[:-2] 
            if link in self.link2detector: 
                self.link2detector[link].append(id)
            self.link2detector[link] = []
            self.link2detector[link].append(id)
        print(f'load data from {len(self.link2detector)} links...')

        self.routes = load_from_pkl(self.routes)
        
    def _load_data(self):
        # load dynamic info
        assert os.path.exists(self.link_traffic_counts), f'File not found {self.link_traffic_counts}'
        assert os.path.exists(self.route_choice_p), f'File not found {self.route_choice_p}'
        assert os.path.exists(self.network_loading_gamma), f'File not found {self.network_loading_gamma}'
        assert os.path.exists(self.network_loading_n), f'File not found {self.network_loading_n}'
        # load y, alpha, gamma, n, ijrl keys
        self.ijrl  = []
        self.L     = set()
        self.alpha = {}
        self.gamma = {}
        self.n     = {}
        # sigma = load_from_pkl(route_choice_sigma)
        p     = load_from_pkl(self.route_choice_p)
        gamma = load_from_pkl(self.network_loading_gamma)
        n     = load_from_pkl(self.network_loading_n)
        y     = load_from_pkl(self.link_traffic_counts).to_dict()

        for key in gamma.keys(): 
            (i, j) = key[0]
            r      = key[1]
            all_links = self.routes[(i, j)][r].get_links()
            for l, l_id in enumerate(all_links): 
                # TODO: if detected
                if l_id in self.link2detector: 
                    d_ids = self.link2detector[l_id]
                    assert not i == j
                    self.ijrl.append( ((i, j), r, l_id) )
                    self.L.add(l_id)
                    for k in self.K:
                        tmp = []
                        for d_id in d_ids:
                            if d_id in y: 
                                tmp.append(int(y[d_id][k])) 
                            else: 
                                self.abnormal_id.add(d_id)
                        if len(tmp) > 0: 
                            self.y[l_id, k] = mean(tmp)
                        else: 
                            self.y[l_id, k] = 0
        
        # TODO: report warning
        for id in self.abnormal_id:
            msg = f'[warning] no observation from {id}\n' 
            add_log(msg = msg, log_path=self.log_path)

        bar = tqdm(gamma.keys(), desc=f'routes from {len(self.L)} links, {len(self.K)} time intervals'.ljust(50))
        for key in bar: 
            (i, j) = key[0]
            r      = key[1]
            all_links = self.routes[(i, j)][r].get_links()
            for l, l_id in enumerate(all_links): 
                # assert ((i, j), r, l_id) in sigma.keys()
                self.gamma[(i, j), r, l_id] = gamma[key][l][self.date].tolist()
                self.n[(i, j), r, l_id]     = n[key][l][self.date].tolist()
                self.alpha[(i, j), r, l_id] = p[key].loc[0, self.date] # TODO: k=?
                
    def _calc_y_ijrkl(self, X, verbose=True):
        term1, term2, term3, term4 = {}, {}, {}, {}
        bar = tqdm(self.ijrl, desc='calculate y_ijrl...') if verbose else self.ijrl
        for key in bar: 
            (i, j) = key[0]
            r      = key[1]
            l      = key[2]
            assert not i==j
            if verbose: 
                bar.set_postfix(ij=(i,j), r=r, l=l)
            for k in self.K: 
                term1.setdefault((l, k), 0)
                term2.setdefault((l, k), 0)
                term3.setdefault((l, k), 0)
                term4.setdefault((l, k), 0)
                # TODO: drop k
                term1[(l, k)] += self.alpha[(i, j), r, l] * X[i, j, k] * (1 - self.n[(i, j), r, l][k]) * (1 - self.gamma[(i, j), r, l][k])
                term2[(l, k)] += self.alpha[(i, j), r, l] * X[i, j, k-1] * (1 - self.n[(i, j), r, l][k-1]) * self.gamma[(i, j), r, l][k-1] if (k-1) in self.K else 0
                term3[(l, k)] += self.alpha[(i, j), r, l] * X[i, j, k-1] * self.n[(i, j), r, l][k-1] * (1 - self.gamma[(i, j), r, l][k-1]) if (k-1) in self.K else 0
                term4[(l, k)] += self.alpha[(i, j), r, l] * X[i, j, k-2] * self.n[(i, j), r, l][k-2] * self.gamma[(i, j), r, l][k-2] if (k-2) in self.K else 0
        return term1, term2, term3, term4

    def _save_result(self, model): 
        current_time = '{:%Y-%m-%d_%H}'.format(datetime.datetime.now())
        output_dir = 'results'
        os.makedirs(output_dir, exist_ok=True)  # Create directory if it does not exist
        result_path = os.path.join(output_dir, f'{current_time}.pkl')
    
        # Save the optimization results
        results = {
            "Variable_values": {v.VarName: v.X for v in model.getVars()},
            "Objective_value": model.ObjVal
        }
        # for benchmark
        for v in model.getVars(): 
            name = v.VarName
            val  = v.X
            if name[0] == 'X': 
                self.x[name] = val
            elif name[0] == 'W': 
                self.w[name] = val
            else: 
                assert False, 'Invalid variable name!'
        self.obj = model.ObjVal
        
        save_as_pkl(results, result_path)

        vars = results['Variable_values'] 
        # x
        with open('X_ijk.csv', 'w') as f:
            f.write('variable,value\n')
            for key, val in vars.items():

                line_to_write = []
                line_to_write.append(key.replace(',', ' '))
                line_to_write.append('{:.1f}'.format(val))
                f.write(','.join(line_to_write) + '\n')
        f.close()
        # p
        # route_choice = load_from_pkl('route_choice_P.pkl')
        # with open('P_ijr.csv', 'w') as f:
        #     f.write('(i j),r,probability\n')
        #     for key, val in route_choice.items():
        #         line_to_write = []
        #         line_to_write.append(str(key[0]).replace(',', ' '))
        #         line_to_write.append(str(key[1]))
        #         line_to_write.append(str(route_choice[key].loc[0, date]))
        #         f.write(','.join(line_to_write) + '\n')
        # f.close()

    def generate_model(self): 
        self._load_data()

        print('> create model')
        model = gp.Model()
        # TODO: integer variable
        if PROBLEM_TYPE == 'NLP': 
            X = model.addVars(self.I, self.J, self.K, vtype=GRB.CONTINUOUS, name='X', lb=0, ub=DEMAND_MAX) # x_{ijk}
        elif PROBLEM_TYPE == 'MILP' or PROBLEM_TYPE == 'MIQP': 
            X = model.addVars(self.I, self.J, self.K, vtype=GRB.INTEGER, name='X', lb=0, ub=DEMAND_MAX)
        # TODO: remove redundant variable
        for k in self.K: 
            for i in self.I: 
                for j in self.J: 
                    if i == j: 
                        model.remove(X[i, j, k])
        # add w
        W = model.addVars(self.L, self.K, vtype=GRB.CONTINUOUS, name='W') # w_{lk}

        # TODO: bottleneck
        e1, e2, e3, e4 = self._calc_y_ijrkl(X)
        for l in self.L: 
            for k in self.K: 
                try: 
                    model.addConstr( W[l, k] >= self.y[(l, k)] - (e1[(l,k)] + e2[(l,k)] + e3[(l,k)] + e4[(l,k)]) )
                    model.addConstr( W[l, k] >= -self.y[(l, k)] + e1[(l,k)] + e2[(l,k)] + e3[(l,k)] + e4[(l,k)] )
                except KeyError: 
                    assert False
        
        # TODO: multiple solutions
        if PROBLEM_TYPE == 'MILP' or PROBLEM_TYPE == 'MIQP': 
            model.setParam(GRB.Param.PoolSolutions, 2)
            model.setParam(GRB.Param.PoolGap, 0.1)
            model.setParam(GRB.Param.PoolSearchMode, 2)

        # TODO: set time limit
        model.setParam('TimeLimit', 60 * SOLVE_HOUR_MAX)
        model.setParam(GRB.Param.PoolSolutions, 2)
        model.setParam(GRB.Param.PoolGap, 0.1)
        model.setParam(GRB.Param.PoolSearchMode, 2)

        obj = gp.LinExpr()
        for l in self.L:
            for k in self.K:
                # TODO: MIQP
                if PROBLEM_TYPE == 'MIQP' or PROBLEM_TYPE == 'NLP': 
                    obj += W[l, k] * W[l, k]
                elif PROBLEM_TYPE == 'MILP': 
                    obj += W[l, k]
        model.setObjective(obj, GRB.MINIMIZE)
        self.model = model
        return model
    
    def solve(self, output_path): 
        # generate model
        self.generate_model()
        assert not self.model == None, 'model is not defined => run generate_model()'
        # optimize
        print('> optimize problem')
        self.model.optimize()
        self.model.write(output_path)
        self._save_result(self.model)
        print('\n')

    # TODO: benchmark
    def benchmark(self, metrics=['APE']):
        def eval_ltc(y_hat, e1, e2, e3, e4, metric): 
            res = []
            for l in self.L: 
                for k in self.K: 
                    try: 
                        # (observation, estimation)
                        tmp = metric(y_hat[l, k], e1[l, k] + e2[l, k] + e3[l, k] + e4[l, k])
                        assert not math.isnan(tmp)
                        res.append(tmp)
                    except KeyError: 
                        assert False
            return res
        
        x    = {}
        for key, val in self.x.items():
            i, j, k = key[2:-1].split(',')
            i = int(i)
            j = int(j)
            k = int(k)
            x[i, j, k] = val 
        e1, e2, e3, e4 = self._calc_y_ijrkl(x, verbose=False)

        for metric in metrics: 
            # APE
            if metric == 'APE': 
                y_apes = eval_ltc(y_hat=self.y, e1=e1, e2=e2, e3=e3, e4=e4, metric=ape)
                plot_ape(y_apes, var_name='Y')
                add_log(msg = f'[APE] mean: {mean(y_apes)}, std: {pstdev(y_apes)}, min: {min(y_apes)}, max: {max(y_apes)}\n', 
                        log_path=self.log_path)
            # GEH
            elif metric == 'GEH': 
                n = (3600 // self.period) # num of Ks in one hour
                y_gehs = []
                m, c   = [], []
                for l in self.L: 
                    for k in self.K: 
                        try: 
                            # (observation, estimation)
                            m.append((e1[l, k] + e2[l, k] + e3[l, k] + e4[l, k]))
                            c.append(self.y[l, k])
                            # empty cache
                            if len(m) >= n and len(c) >= n: 
                                tmp = geh(sum(m), sum(c))
                                y_gehs.append(tmp)
                                m = []
                                c = []
                        except KeyError: 
                            assert False
                plot_geh(y_gehs, var_name='Y')
                add_log(msg = f'[GEH] mean: {mean(y_gehs)}. std: {pstdev(y_gehs)}, min: {min(y_gehs)}, max: {max(y_gehs)}\n', 
                        log_path=self.log_path)
            # MAE
            elif metric == 'MAE': 
                y_maes = eval_ltc(y_hat=self.y, e1=e1, e2=e2, e3=e3, e4=e4, metric=mae)
                add_log(msg = f'[MAE] {mean(y_maes)}\n', 
                        log_path=self.log_path)
            # RMSE
            elif metric == 'RMSE': 
                y_rmses = eval_ltc(y_hat=self.y, e1=e1, e2=e2, e3=e3, e4=e4, metric=rmse)
                add_log(msg = f'[RMSE] {np.sqrt(mean(y_rmses))}\n', 
                        log_path=self.log_path)
            else: 
                assert False, 'Not implemented!'
    
    def test(self): 
        # TODO: check routes
        all_routes = set()
        all_routes_detected = set()
        for key in self.ijrl: 
            (i, j) = key[0]
            r      = key[1]
            l      = key[2]
            all_routes.add((i, j, r))
            if l in self.link2detector:
                all_routes_detected.add((i, j, r))
        # log
        add_log(msg = f'# of all routes: {len(all_routes)}\n# of all detected routes: {len(all_routes_detected)}\n', 
                log_path=self.log_path)
        # save data
        save_as_pkl(all_routes, pkl_path='set_all_routes.pkl')
        save_as_pkl(all_routes_detected, pkl_path='set_all_routes_detected.pkl')
                
        # self._load_data(date, data_paths=[y, p, gamma, n])
        # set_link_from_routes    = self.L_from_routes
        # set_link_with_detectors = set(self.link2detector)
        # plot_missing_data_venn(
        #     set_a=set_link_from_routes, 
        #     set_b=set_link_with_detectors
        # )
        # # log
        # with open('log.txt', 'w') as f: 
        #     f.write(f'set A (links from routes): {len(set_link_from_routes)}\nset B (links with detectors): {len(set_link_with_detectors)}\n')
        # f.close()
        # # data
        # save_as_pkl(set_link_from_routes, pkl_path='set_link_from_routes.pkl')
        # save_as_pkl(set_link_with_detectors, pkl_path='set_link_with_detectors.pkl')

class DODME_v2(DODME):
    ''' n > 1 '''
    def __init__(self, config):
        super().__init__(config)

    # TODO: debug
    def _calc_y_ijrkl(self, X, verbose=True):
        # In general: 
        # N = max_{ijrt}{n_{ijrt}}
        # y_{jirk}^l = \sum_{t_1, t_1 + n_{ijrt_1} = k} \alpha_{ijrt_1}^l (1 - \gamma_{ijrt_1}^l) x_{ijt_1} + \sum_{t_2, t_2 + n_{ijrt_2} = k-1} \alpha_{ijrt_2}^l \gamma_{ijrt_2}^l x_{ijt_2}
        
        # In basic setting: 
        # n \in {0, 1}

        # (n, l, k) => n: #time intervals needed
        terms = {}
        # term1, term2, term3, term4 = {}, {}, {}, {}
        bar = tqdm(self.ijrl, desc='calculate y_ijrl...') if verbose else self.ijrl
        for key in bar: 
            (i, j) = key[0]
            r      = key[1]
            l      = key[2]
            assert not i==j
            if verbose: 
                bar.set_postfix(ij=(i,j), r=r, l=l)
            for k in self.K: 
                terms.setdefault((l, k), 0)
                for t1 in range(k+1): 
                    assert 0 <= t1 and t1 <= k
                    if t1 + self.n[(i, j), r, l][t1] == k: 
                        # I. non-delay
                        terms[l, k] += self.alpha[(i, j), r, l] * (1 - self.gamma[(i, j), r, l][t1]) * X[i, j, t1] if t1 in self.K else 0
                        # II. delay
                        terms[l, k] += self.alpha[(i, j), r, l] * self.gamma[(i, j), r, l][t1-1] * X[i, j, t1-1] if (t1-1) in self.K else 0
        return terms 
        
    def generate_model(self):
        '''
        n_{ijrt} = 0, 1, ..., n
        '''
        self._load_data()
        print('> create model')
        model = gp.Model()
        # TODO: integer variable
        if PROBLEM_TYPE == 'NLP': 
            X = model.addVars(self.I, self.J, self.K, vtype=GRB.CONTINUOUS, name='X', lb=0, ub=DEMAND_MAX) # x_{ijk}
        elif PROBLEM_TYPE == 'MILP' or PROBLEM_TYPE == 'MIQP': 
            X = model.addVars(self.I, self.J, self.K, vtype=GRB.INTEGER, name='X', lb=0, ub=DEMAND_MAX)
        # TODO: remove redundant variable
        for k in self.K: 
            for i in self.I: 
                for j in self.J: 
                    if i == j: 
                        model.remove(X[i, j, k])
        # add w
        W = model.addVars(self.L, self.K, vtype=GRB.CONTINUOUS, name='W') # w_{lk}

        terms = self._calc_y_ijrkl(X)

        for l in self.L: 
            for k in self.K: 
                try: 
                    # add constraints
                    model.addConstr(W[l, k] >= self.y[(l, k)] - terms[l, k])
                    model.addConstr(W[l, k] >= -self.y[(l, k)] + terms[l, k])      
                except KeyError: 
                    assert False
        
        # TODO: multiple solutions
        if PROBLEM_TYPE == 'MILP' or PROBLEM_TYPE == 'MIQP': 
            model.setParam(GRB.Param.PoolSolutions, 2)
            model.setParam(GRB.Param.PoolGap, 0.1)
            model.setParam(GRB.Param.PoolSearchMode, 2)

        # TODO: set time limit
        model.setParam('TimeLimit', 60 * 60 * SOLVE_HOUR_MAX)
        model.setParam(GRB.Param.PoolSolutions, 2)
        model.setParam(GRB.Param.PoolGap, 0.1)
        model.setParam(GRB.Param.PoolSearchMode, 2)

        obj = gp.LinExpr()
        for l in self.L:
            for k in self.K:
                # TODO: MIQP
                if PROBLEM_TYPE == 'MIQP' or PROBLEM_TYPE == 'NLP': 
                    obj += W[l, k] * W[l, k]
                elif PROBLEM_TYPE == 'MILP': 
                    obj += W[l, k]
        model.setObjective(obj, GRB.MINIMIZE)
        self.model = model
        return model
    
    def benchmark(self, metrics=['APE']):
        def eval_ltc(y_hat, y_est, metric): 
            res = []
            for key in y_est: 
                try: 
                    tmp = metric(y_hat[key], y_est[key])
                    assert not math.isnan(tmp)
                    res.append(tmp)
                except KeyError: 
                    assert False
            return res
        
        x = {}
        for key, val in self.x.items():
            i, j, k = key[2:-1].split(',')
            i = int(i)
            j = int(j)
            k = int(k)
            x[i, j, k] = val 

        terms = self._calc_y_ijrkl(x, verbose=False)

        for metric in metrics: 
            # APE
            if metric == 'APE': 
                y_apes = eval_ltc(y_hat=self.y, y_est=terms, metric=ape)
                plot_ape(y_apes, var_name='Y')
                add_log(msg = f'[APE Y] mean: {mean(y_apes)}, std: {pstdev(y_apes)}, min: {min(y_apes)}, max: {max(y_apes)}\n', 
                        log_path=self.log_path)
            # GEH
            elif metric == 'GEH': 
                n = (3600 // self.period) # num of Ks in one hour
                y_gehs = []
                m, c   = [], []
                for key in self.y: 
                    try: 
                        # (observation, estimation)
                        m.append(terms[key])
                        c.append(self.y[key])
                        # empty cache
                        if len(m) >= n and len(c) >= n: 
                            tmp = geh(sum(m), sum(c))
                            y_gehs.append(tmp)
                            m = []
                            c = []
                    except KeyError: 
                        assert False
                plot_geh(y_gehs, var_name='Y')
                add_log(msg = f'[GEH Y] mean: {mean(y_gehs)}. std: {pstdev(y_gehs)}, min: {min(y_gehs)}, max: {max(y_gehs)}\n', 
                        log_path=self.log_path)
            # MAE
            elif metric == 'MAE': 
                y_maes = eval_ltc(y_hat=self.y, y_est=terms, metric=mae)
                add_log(msg = f'[MAE Y] {mean(y_maes)}\n', 
                        log_path=self.log_path)
            # RMSE
            elif metric == 'RMSE': 
                y_rmses = eval_ltc(y_hat=self.y, y_est=terms, metric=rmse)
                add_log(msg = f'[RMSE Y] {np.sqrt(mean(y_rmses))}\n', 
                        log_path=self.log_path)
            else: 
                assert False, 'Not implemented!'
    
    def test(self, x_gt, metrics=['APE']):
        ''' test benchmark demand ground truth '''
        def eval_ltc(y_hat, y, metric): 
            res = []
            for key in y_hat: 
                try: 
                    tmp = metric(y_hat[key], y[key])
                    assert not math.isnan(tmp)
                    res.append(tmp)
                except KeyError: 
                    assert False, f'unknown key: {key}'
            return res
        
        assert x_gt is not None
 
        x = {}
        for key, val in self.x.items():
            i, j, k = key[2:-1].split(',')
            i = int(i)
            j = int(j)
            k = int(k)
            x[i, j, k] = val 

        for metric in metrics: 
            # APE
            if metric == 'APE': 
                y_apes = eval_ltc(y_hat=x_gt, y=x, metric=ape)
                plot_ape(y_apes, var_name='X')
                add_log(msg = f'[APE X] mean: {mean(y_apes)}, std: {pstdev(y_apes)}, min: {min(y_apes)}, max: {max(y_apes)}\n', 
                        log_path=self.log_path)
            # GEH
            elif metric == 'GEH': 
                n = (3600 // self.period) # num of Ks in one hour
                y_gehs = []
                m, c   = [], []
                for key in x_gt: 
                    try: 
                        # (observation, estimation)
                        m.append(x[key])
                        c.append(x_gt[key])
                        # empty cache
                        if len(m) >= n and len(c) >= n: 
                            tmp = geh(sum(m), sum(c))
                            y_gehs.append(tmp)
                            m = []
                            c = []
                    except KeyError: 
                        assert False
                plot_geh(y_gehs, var_name='X')
                add_log(msg = f'[GEH X] mean: {mean(y_gehs)}. std: {pstdev(y_gehs)}, min: {min(y_gehs)}, max: {max(y_gehs)}\n', 
                        log_path=self.log_path)
            # MAE
            elif metric == 'MAE': 
                y_maes = eval_ltc(y_hat=x_gt, y=x, metric=mae)
                add_log(msg = f'[MAE X] {mean(y_maes)}\n', 
                        log_path=self.log_path)
            # RMSE
            elif metric == 'RMSE': 
                y_rmses = eval_ltc(y_hat=x_gt, y=x, metric=rmse)
                add_log(msg = f'[RMSE X] {np.sqrt(mean(y_rmses))}\n', 
                        log_path=self.log_path)
            else: 
                assert False, 'Not implemented!'

class DODME_v3(DODME_v2): 
    def __init__(self, config):
        self.link2detector = {}
        self.routes        = {}
        self.model         = None

        # data
        self.date  = config['WORK_DATE'] 
        self.alpha = {}
        self.gamma = {}
        self.n     = {}
        self.y     = {}

        # result
        self.x     = {}
        self.w     = {}
        self.obj   = 0
        
        # set
        self.ijrl  = None
        self.I     = I # hard code
        self.J     = J # hard code
        self.L     = None
        self.K     = [i for i in range(config['N_TIME_INTERVAL'])] # hard code

        self.log_path = config['BENCHMARK_LOG_PATH']
        self.period   = config['TIME_INTERVAL'] # i.e., 4mins
        
        self.link_traffic_counts   = os.path.join(config['PWD'] , config['LINK_TRAFFIC_COUNT_PATH'])
        self.detector_link_mapping = os.path.join(config['PWD'] , config['DETECTOR_LINK_MAPPING_PATH'])
        self.routes                = os.path.join(config['PWD'] , config['ROUTE_CHOICE_DICT_PATH'])
        self.route_choice_p        = os.path.join(config['PWD'] , config['ROUTE_CHOICE_P_PATH'])
        self.network_loading_gamma = os.path.join(config['PWD'] , config['NETWORK_LOADING_GAMMA_PATH'])
        self.network_loading_n     = os.path.join(config['PWD'] , config['NETWORK_LOADING_N_PATH'])

        assert os.path.exists(self.routes), f'File not found {self.routes}'
        self.routes = load_from_pkl(self.routes)
        
    def _load_data(self):
        # load dynamic info
        assert os.path.exists(self.link_traffic_counts), f'File not found {self.link_traffic_counts}'
        assert os.path.exists(self.route_choice_p), f'File not found {self.route_choice_p}'
        assert os.path.exists(self.network_loading_gamma), f'File not found {self.network_loading_gamma}'
        assert os.path.exists(self.network_loading_n), f'File not found {self.network_loading_n}'
        # load y, alpha, gamma, n, ijrl keys
        self.ijrl  = []
        self.L     = set()
        self.alpha  = load_from_pkl(self.route_choice_p)                 # ijrlk
        self.gamma  = load_from_pkl(self.network_loading_gamma)          # ijrlk
        self.n      = load_from_pkl(self.network_loading_n)              # ijrlk
        self.y      = load_from_pkl(self.link_traffic_counts)[self.date] # l, k
        # add key
        for key in self.gamma.keys(): 
            (i, j)  = key[0]
            r, l, k = key[1], key[2], key[3]
            assert i in self.I, 'index i out of range'
            assert j in self.J, 'index j out of range'
            assert k in self.K, 'index k out of range'
            self.ijrl.append((i, j), r, l)
            self.L.add(l) 

    def _calc_y_ijrkl(self, X, verbose=True):
        terms = {}
        bar = tqdm(self.ijrl, desc='calculate y_ijrl...') if verbose else self.ijrl
        for key in bar: 
            (i, j) = key[0]
            r      = key[1]
            l      = key[2]
            assert not i==j
            if verbose: 
                bar.set_postfix(ij=(i,j), r=r, l=l)
            for k in self.K: 
                terms.setdefault((l, k), 0)
                for t1 in range(k+1): 
                    assert 0 <= t1 and t1 <= k
                    if t1 + self.n[(i, j), r, l][t1] == k: 
                        # I. non-delay
                        terms[l, k] += self.alpha[(i, j), r, l] * (1 - self.gamma[(i, j), r, l][t1]) * X[i, j, t1] if t1 in self.K else 0
                        # II. delay
                        terms[l, k] += self.alpha[(i, j), r, l] * self.gamma[(i, j), r, l][t1-1] * X[i, j, t1-1] if (t1-1) in self.K else 0
        return terms