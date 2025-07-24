import argparse
import os 
import math
import datetime
import numpy as np
from tqdm import tqdm
from statistics import mean, pstdev
from scripts.utils import load_from_pkl, save_as_pkl, add_log
from scripts.visual import plot_ape
from scripts.benchmark import ape, mae, rmse
from scripts.simulation import SumoSimulation
# gurobi
import gurobipy as gp
from gurobipy import GRB
import pickle
import matplotlib.pyplot as plt
from scipy.stats import pearsonr

WEIGHTED = False
BIG_M = 0x7fff

class DODME(object): 
    def __init__(self, config, g, n, v, d, t, y):
        self.link2detector = {}
        self.routes        = {}
        self.model         = None
        self.pwd           = config['PWD']
        self.time_limit    = config['MAX_SOLVE_TIME']

        # data
        self.alpha = {}
        self.gamma = g
        self.n     = n
        self.v     = v
        self.d     = d
        self.t     = t
        self.y_hat = y

        self.x_gt = load_from_pkl(pkl_path=os.path.join(config['PWD'], config['DEMAND_GROUND_TRUTH_SAVE_PATH']))
        
        self.period     = config['TIME_INTERVAL'] # 180s
        self.interval_n = config['TIME_INTERVAL_N']

        # result
        self.x      = {}
        self.x_ijrk = {}
        self.x_ijk  = {}
        self.w      = {}
        self.obj    = 0

        # variable
        self.X     = None
        self.W     = None
        self.T     = None
        
        # set
        self.ijrl  = []
        self.I     = [i for i in range(1, config['ORIGIN_N'] + 1)] 
        self.J     = [i for i in range(1, config['DESTINATION_N'] + 1)]
        self.R     = [i for i in range(config['ROUTE_N'])] 
        self.L     = set() # set L: link observed
        self.K     = [i for i in range(config['TIME_INTERVAL_N'])] 

        self.ij    = set()
        self.lk    = []

        # path
        self.lp_path  = os.path.join(config['PWD'], config['LP_FILE_PATH'])
        self.log_path = config['BENCHMARK_LOG_PATH']

        # load data
        self._load_data()
        
    def _load_data(self):
        # all links with detectors
        all_links = []
        for key in self.y_hat.keys(): 
            l, k = key
            assert k in self.K
            # avoid duplicate
            if k == 0: 
                all_links.append(l)
        # set I, J, K, L
        for key in self.gamma.keys(): 
            (i, j)  = key[0]
            r, l, k = key[1], key[2], key[3]
            self.ij.add((i, j))
            # only consider |R| routes
            if r in self.R: 
                assert i in self.I, 'index i out of range'
                assert j in self.J, 'index j out of range'
                assert k in self.K, 'index k out of range'
                # avoid duplicate
                if k == 0: 
                    self.ijrl.append((i, j, r, l))
                # if observed
                if l in all_links:
                    self.L.add(l) 

        # set ij
        # for i in self.I: 
        #     for j in self.J: 
        #         if not i == j: 
        #             self.ij.append((i, j))

        # set lk
        for l in self.L:
            for k in self.K: 
                if self.y_hat[l, k] > 0: 
                    self.lk.append((l, k))

        # TODO: calculate alpha weight
        # df = pd.DataFrame(self.d, index=[i for i in range(len(self.d))])
        # df_exp = df.apply(np.exp) # f => exp()
        # df_sum = df_exp.values.sum()
        # dic = df_exp.to_dict()
        # for l, data in dic.items(): 
        #     for k, val in data.items():
        #         self.alpha[l, k] = float(val / df_sum)

    def _calc_y_est(self, verbose=False):
        # initialize
        y_est = dict.fromkeys(self.y_hat, 0)
        # estimation
        bar = tqdm(self.ijrl, desc='calculate y_ijrl...') if verbose else self.ijrl
        for key in bar: 
            i, j, r, l = key
            if verbose: 
                bar.set_postfix(ij=(i,j), r=r, l=l)
            for k in self.K: 
                # if observed
                if (l, k) in self.lk: 
                # if (l, k) in self.y_hat and self.y_hat[l, k] > 0: 
                    t1, t2 = [], []
                    for t in range(k): 
                        if t + self.n[(i, j), r, l, t] == k:
                            t1.append(t)
                        elif t + self.n[(i, j), r, l, t] == k-1:
                            t2.append(t)
                    for t in t1: 
                        y_est[l, k] += self.X[i, j, r, t] * (1 - self.gamma[(i, j), r, l, t])
                    for t in t2: 
                        y_est[l, k] += self.X[i, j, r, t] * self.gamma[(i, j), r, l, t]
        return y_est
    
    # TODO: modify save result
    def _save_result(self): 
        assert self.model is not None
        
        # save original model
        self.model.write(self.lp_path)
        current_time = '{:%Y-%m-%d_%H}'.format(datetime.datetime.now())
        pkl_path = os.path.join(self.pwd, f'results-{current_time}.pkl')
        csv_path = os.path.join(self.pwd, f'results-{current_time}.csv')

        # Save the optimization results
        results = {
            "Variable_values": {v.VarName: v.X for v in self.model.getVars()},
            "Objective_value": self.model.ObjVal
        }
        # pickle
        save_as_pkl(results, pkl_path)
        # csv
        vars = results['Variable_values'] 
        with open(csv_path, 'w') as f:
            f.write('variable,value\n')
            for key, val in vars.items():
                line_to_write = []
                line_to_write.append(key)
                line_to_write.append('{:.3f}'.format(val))
                f.write(','.join(line_to_write) + '\n')
        f.close()

        # Save OD Matrix
        for key, val in self.x.items(): 
            i, j, r, k = key[2:-1].split(',')
            i = int(i)
            j = int(j)
            k = int(k)
            r = int(r)
            self.x_ijrk[i, j, r, k] = val 
        for (i, j) in self.ij: 
            for k in self.K: 
                self.x_ijk.setdefault((i, j, k), 0)
                for r in self.R: 
                    self.x_ijk[i, j, k] += self.x_ijrk[i, j, r, k]
        pkl_path = os.path.join(self.pwd, f'OD-Matrix-{current_time}.pkl')
        csv_path = os.path.join(self.pwd, f'OD-Matrix-{current_time}.csv')
        # pickle
        save_as_pkl(self.x_ijk, pkl_path)
        # csv
        with open(csv_path, 'w') as f:
            f.write('i,j,k,value\n')
            for key, val in self.x_ijk.items():
                i, j, k = key
                line_to_write = []
                line_to_write.append(str(i))
                line_to_write.append(str(j))
                line_to_write.append(str(k))
                line_to_write.append(str(int(val)))
                f.write(','.join(line_to_write) + '\n')
        f.close()
        
    def generate_model(self, verbose=False, presolve=True):
        '''
        n_{ijrt} = 0, 1, ..., n
        '''
        model = gp.Model()
        if not verbose: 
            # mute console 
            model.setParam('LogToConsole', 0)
            model.setParam('OutputFlag', 0)
        if not presolve: 
            model.setParam('Presolve', 0)
        # set time limit
        model.setParam('TimeLimit', self.time_limit)
        # initialize variables
        # self.X = model.addVars(self.I, self.J, self.R, self.K, vtype=GRB.CONTINUOUS, name='X', lb=0) # x_{ijrk}
        self.X = model.addVars(self.ij, self.R, self.K, vtype=GRB.CONTINUOUS, name='X', lb=0) # x_{ijrk}
        # self.T = model.addVars(self.I, self.J, self.K, vtype=GRB.CONTINUOUS, name='T', lb=0) # TODO: tau_{ijrk}
        self.T = model.addVars(self.ij, self.K, vtype=GRB.CONTINUOUS, name='T', lb=0) # TODO: tau_{ijrk}
        # self.W = model.addVars(self.L, self.K, vtype=GRB.CONTINUOUS, name='W') # w_{lk}
        self.W = model.addVars(self.lk, vtype=GRB.CONTINUOUS, name='W') # w_{lk}
        
        # calculate y estimation
        y_est = self._calc_y_est()
        
        # constraint set A {l, k}
        for (l, k) in self.lk: 
            try: 
                model.addConstr(self.W[l, k] >= self.y_hat[(l, k)] - y_est[l, k])
                model.addConstr(self.W[l, k] >= -self.y_hat[(l, k)] + y_est[l, k])  
            except KeyError: 
                assert False
        
        # constraint set B {i, j, k}
        for (i, j) in self.ij: 
            for k in self.K: 
                for r in self.R: 
                    try: 
                        model.addConstr(self.X[i, j, r, k] * self.t[i, j, r, k] - self.X[i, j, r, k] * self.T[i, j, k] == 0)
                        model.addConstr(self.T[i, j, k] <= self.t[i, j, r, k])
                    except KeyError: 
                        assert False
        # initialize and save self.model
        self.model = model
        return model
    
    # TODO: add prior
    def set_objective(self, params): 
        # check model
        assert self.model is not None, 'model is not defined => run generate_model()'
        # objective function
        obj = gp.LinExpr()
        for (l, k) in self.lk: 
            obj += self.W[l, k] * self.W[l, k]
        self.model.setObjective(obj, GRB.MINIMIZE)

    def solve(self, output=True): 
        # check model
        assert self.model is not None, 'model is not defined => run generate_model()'
        # optimize
        self.model.optimize()
        self.obj = self.model.ObjVal
        # output model & results
        self.w = {v.VarName: v.X for v in self.model.getVars() if v.VarName[0] == 'W'}
        self.x = {v.VarName: v.X for v in self.model.getVars() if v.VarName[0] == 'X'}
        if output: 
            self._save_result()
        original_obj = 0.0
        for w in self.w.values(): 
            original_obj += w * w
        return original_obj
    
    # APIs
    def get_model(self): 
        assert self.model is not None
        return self.model
    
    def get_obj(self): 
        return self.obj

    def benchmark(self, metrics=['APE']):
        # recover y estimation
        y_est = dict.fromkeys(self.y_hat, 0)
        for key in self.ijrl: 
            i, j, r, l = key
            for k in self.K: 
                # if observed
                if (l, k) in self.y_hat and self.y_hat[l, k] > 0: 
                    t1, t2 = [], []
                    for t in range(k): 
                        if t + self.n[(i, j), r, l, t] == k:
                            t1.append(t)
                        elif t + self.n[(i, j), r, l, t] == k-1:
                            t2.append(t)
                    for t in t1: 
                        y_est[l, k] += self.x_ijrk[i, j, r, t] * (1 - self.gamma[(i, j), r, l, t])
                    for t in t2: 
                        y_est[l, k] += self.x_ijrk[i, j, r, t] * self.gamma[(i, j), r, l, t]
        # link traffic count
        y_apes = []
        y_mae  = []
        y_rmse = []
        # demand
        x_apes = []
        x_mae  = []
        x_rmse = []
        # for Y
        for (l, k) in self.lk: 
            # [APE]
            y_ape = ape(y_hat=self.y_hat[l, k], y=y_est[l, k])
            assert not math.isnan(y_ape)
            y_apes.append(y_ape)
            # [MAE]
            mae_ = mae(self.y_hat[l, k], y_est[l, k])
            assert not math.isnan(mae_)
            y_mae.append(mae_)
            # [RMSE]
            rmse_ = rmse(self.y_hat[l, k], y_est[l, k])
            assert not math.isnan(rmse_)
            y_rmse.append(rmse_)
        # for X
        n = (3600 // self.period) # num of Ks in one hour
        hour_n = (self.interval_n // n)

        x_agg    = [0 for i in range(hour_n)]
        x_gt_agg = [0 for i in range(hour_n)]

        for (i, j) in self.ij: 
            for k in self.K:
                idx = (k // n)


        for (i, j) in self.ij: 
            for k in self.K: 
                # [APE]
                x_ape = ape(y_hat=self.x_gt[i, j, k], y=self.x_ijk[i, j, k])
                assert not math.isnan(x_ape)
                x_apes.append(x_ape)
                # [MAE]
                mae_ = mae(self.x_gt[i, j, k], self.x_ijk[i, j, k])
                assert not math.isnan(mae_)
                x_mae.append(mae_)
                # [RMSE]
                rmse_ = rmse(self.x_gt[i, j, k], self.x_ijk[i, j, k])
                assert not math.isnan(rmse_)
                x_rmse.append(rmse_)
        # output
        for metric in metrics: 
            if metric == 'APE': 
                plot_ape(y_apes, var_name='Y')
                add_log(msg = f'[APE Y] mean: {mean(y_apes)}, std: {pstdev(y_apes)}, min: {min(y_apes)}, max: {max(y_apes)}\n', 
                        log_path=self.log_path)
                plot_ape(x_apes, var_name='X')
                add_log(msg = f'[APE X] mean: {mean(x_apes)}, std: {pstdev(x_apes)}, min: {min(x_apes)}, max: {max(y_apes)}\n', 
                        log_path=self.log_path)
            elif metric == 'MAE': 
                add_log(msg = f'[MAE Y] {mean(y_mae)}\n', 
                        log_path=self.log_path)
                add_log(msg = f'[MAE X] {mean(x_mae)}\n', 
                        log_path=self.log_path)
            elif metric == 'RMSE': 
                add_log(msg = f'[RMSE Y] {np.sqrt(mean(y_rmse))}\n', 
                        log_path=self.log_path)
                add_log(msg = f'[RMSE X] {np.sqrt(mean(x_rmse))}\n', 
                        log_path=self.log_path)
            else: 
                assert False, 'Not implemented!'
        # # [APE] demand
        # x_apes = []
        # for i in self.I: 
        #     for j in self.J: 
        #         for k in self.K: 
        #             # x: sum over r
        #             x_ijk = 0
        #             for r in self.R: 
        #                 x_ijk += self.x[i, j, r, k]
        #             # x_hat
        #             x_ijk_hat = self.x_gt[(i, j)]
        #             # x_ape
        #             tmp = ape(y_hat=x_ijk_hat, y=x_ijk)
        #             assert not math.isnan(tmp)
        #             x_apes.append(tmp)
        # plot_ape(x_apes, var_name='X')
        # add_log(msg = f'[APE X] mean: {mean(x_apes)}, std: {pstdev(x_apes)}, min: {min(x_apes)}, max: {max(x_apes)}\n', 
        #         log_path=self.log_path)
        # for metric in metrics: 
        #     # APE
        #     if metric == 'APE': 
        #         pass
                            
        #     # TODO: GEH
        #     elif metric == 'GEH':
        #         pass 
        #     #     n = (3600 // self.period) # num of Ks in one hour
        #     #     y_gehs = []
        #     #     m, c   = [], []
        #     #     for l in self.L: 
        #     #         for k in self.K: 
        #     #             try: 
        #     #                 # (observation, estimation)
        #     #                 m.append((e1[l, k] + e2[l, k] + e3[l, k] + e4[l, k]))
        #     #                 c.append(self.y[l, k])
        #     #                 # empty cache
        #     #                 if len(m) >= n and len(c) >= n: 
        #     #                     tmp = geh(sum(m), sum(c))
        #     #                     y_gehs.append(tmp)
        #     #                     m = []
        #     #                     c = []
        #     #             except KeyError: 
        #     #                 assert False
        #     #     plot_geh(y_gehs, var_name='Y')
        #     #     add_log(msg = f'[GEH] mean: {mean(y_gehs)}. std: {pstdev(y_gehs)}, min: {min(y_gehs)}, max: {max(y_gehs)}\n', 
        #     #             log_path=self.log_path)
        #     # TODO: MAE
        #     elif metric == 'MAE': 
        #         # [MAE] link traffic count
        #         lis = []
        #         for l in self.L: 
        #             for k in self.K: 
        #                 if (l, k) in self.y_hat and self.y_hat[l, k] > 0: 
        #                     tmp = ape(y_hat=self.y_hat[l, k], y=self.y_est[l, k])
        #                     assert not math.isnan(tmp)
        #                     y_apes.append(tmp)
        #         plot_ape(y_apes, var_name='Y')
        #         add_log(msg = f'[APE Y] mean: {mean(y_apes)}, std: {pstdev(y_apes)}, min: {min(y_apes)}, max: {max(y_apes)}\n', 
        #                 log_path=self.log_path)
        #     #     y_maes = eval_ltc(y_hat=self.y, e1=e1, e2=e2, e3=e3, e4=e4, metric=mae)
        #     #     add_log(msg = f'[MAE] {mean(y_maes)}\n', 
        #     #             log_path=self.log_path)
        #     # TODO: RMSE
        #     elif metric == 'RMSE': 
        #     #     y_rmses = eval_ltc(y_hat=self.y, e1=e1, e2=e2, e3=e3, e4=e4, metric=rmse)
        #     #     add_log(msg = f'[RMSE] {np.sqrt(mean(y_rmses))}\n', 
        #     #             log_path=self.log_path)
        #     else: 
        #         assert False, 'Not implemented!'

class DODME_congested(DODME): 
    def __init__(self, config, g, n, v, d, t, y, neighbor, fd, 
                 sim_args
                 ): 
        super().__init__(config, g, n, v, d, t, y)
        # fundamental relation parameters
        
       
        self.fd = fd
        # decision variables
        self.L1 = None
        self.L2 = None
        self.L3 = None
        
        self.l1 = None
        self.l2 = None
        self.l3 = None

        self.N  = neighbor

        self.sumo_sim = SumoSimulation(sim_args) 

        # for testing purpose
        self.w_q = 1.0
        self.w_k = 1.0
        self.w_v = 1.0

        #for testing purpose of q, k, v

        # with open('simulation_k.pkl', 'rb') as f:
        #     self.k = pickle.load(f)
        # with open('simulation_q.pkl', 'rb') as f:
        #     self.q = pickle.load(f)
        # with open('simulation_v.pkl', 'rb') as f:
        #     self.linkV = pickle.load(f)

    def simulation(self, demand): 
        def objective_function(q, k, v):
            q_loss = self.w_q * np.sqrt(np.sum((q - self.q) * (q - self.q), axis=None)) / np.sqrt(np.sum(self.q * self.q, axis=None))
            k_loss = self.w_k * np.sqrt(np.sum((k - self.k) * (k - self.k), axis=None)) / np.sqrt(np.sum(self.k * self.k, axis=None))
            v_loss = self.w_v * np.sqrt(np.sum((v - self.linkV) * (v - self.linkV), axis=None)) / np.sqrt(np.sum(self.linkV * self.linkV, axis=None))
            of = q_loss + k_loss + v_loss
            ''' objective function: q, k, v => ndarray (l, k) '''
            return of, q_loss, k_loss, v_loss
        
        q, k, v = self.sumo_sim.run_sumo(od_matrix = demand)
        assert q.shape == self.q.shape, f"q shape is not equal to self.q shape, {q.shape} != {self.q.shape}"
        return objective_function(q, k, v)

    def generate_model(self, x_upper, verbose=False, presolve=True):
        model = gp.Model()
        if not verbose: 
            # mute console 
            model.setParam('LogToConsole', 0)
            model.setParam('OutputFlag', 0)
        if not presolve: 
            model.setParam('Presolve', 0)
        # set time limit
        model.setParam('TimeLimit', self.time_limit)
        # initialize variables
        # self.X = model.addVars(self.I, self.J, self.R, self.K, vtype=GRB.CONTINUOUS, name='X', lb=0) # x_{ijrk}
        self.X  = model.addVars(self.ij, self.R, self.K, vtype=GRB.CONTINUOUS, name='X', lb=10) # x_{ijrk}
        # self.W = model.addVars(self.L, self.K, vtype=GRB.CONTINUOUS, name='W') # w_{lk}
        self.L1 = model.addVars(self.lk, vtype=GRB.CONTINUOUS, name='L1') # w_{lk}
        self.L2 = model.addVars(self.lk, vtype=GRB.CONTINUOUS, name='L2') # w_{lk}
        self.L3 = model.addVars(self.lk, vtype=GRB.CONTINUOUS, name='L3') # w_{lk}
        
        # calculate y estimation
        y_est = self._calc_y_est()
        # calculate d estimation

        # Commented 20250308, a division by 0 occurred
        # d_est = {key: (val/self.v[key]) for key, val in y_est.items()}
        
        # fundamental relation
    
        # Commented 20250308, removed for now
        # a, b, c = self.fd
        # print(self.X.keys())
        # assert (1, 2, 0, 0) in self.X.keys()
        
        # constraint set A {l, k}
        for (l, k) in self.lk: 
            try: 
                model.addConstr(self.L1[l, k] >=   self.y_hat[(l, k)] - y_est[l, k])
                model.addConstr(self.L1[l, k] >= - self.y_hat[(l, k)] + y_est[l, k])  
                model.addConstr(self.L2[l, k] * self.v[l, k] >=   self.d[l, k] * self.v[l, k] - y_est[l, k])
                model.addConstr(self.L2[l, k] * self.v[l, k] >= - self.d[l, k] * self.v[l, k] + y_est[l, k])
                # model.addConstr(self.L3[l, k] >=   a * d_est[l, k] - self.y_hat[(l, k)])
                # model.addConstr(self.L3[l, k] >= - a * d_est[l, k] + self.y_hat[(l, k)])
                # model.addConstr(self.L3[l, k] >=   b * d_est[l, k] + c - self.y_hat[(l, k)])
                # model.addConstr(self.L3[l, k] >= - b * d_est[l, k] - c + self.y_hat[(l, k)])
            except KeyError: 
                assert False
        # others
        for ij in self.ij: 
            for k in self.K: 
                model.addConstr(
                    sum(self.X[ij[0], ij[1], r, k] for r in self.R) <= x_upper
                )
        for i in self.I: 
            for k in self.K:
                model.addConstr( 
                    sum(self.X[ij[0], ij[1], r, k] for ij in self.ij if i == ij[0] for r in self.R ) 
                    >= sum(self.y_hat[(l, k)] for l in self.N[i])
                ) 

        # save model
        self.model = model
        return model

    def set_objective(self, params): 
        # check model
        assert self.model is not None, 'model is not defined => run generate_model()'
        # objective function
        obj = gp.LinExpr()
        obj += params[0] * sum(self.L1[l, k] * self.L1[l, k] for (l, k) in self.lk) / sum(self.y_hat[(l, k)] * self.y_hat[(l, k)] for (l, k) in self.lk)
        obj += params[1] * sum(self.L2[l, k] * self.L2[l, k] for (l, k) in self.lk) / sum(self.d[(l, k)] * self.d[(l, k)] for (l, k) in self.lk)
        # obj += params[2] * sum(self.L3[l, k] * self.L3[l, k] for (l, k) in self.lk) / sum(self.y_hat[(l, k)] * self.y_hat[(l, k)] for (l, k) in self.lk)
        self.model.setObjective(obj, GRB.MINIMIZE)

    def solve(self, output=True): 
        # check model
        assert self.model is not None, 'model is not defined => run generate_model()'
        # optimize
        self.model.optimize()
        self.obj = self.model.ObjVal
        # output model & results
        # self.w = {v.VarName: v.X for v in self.model.getVars() if v.VarName[0] == 'W'}
        self.x = {v.VarName: v.X for v in self.model.getVars() if v.VarName[0] == 'X'}
        if output: 
            self._save_result()

        # TODO
        od_matrix = np.zeros((len(self.I), len(self.J), len(self.K)))
        for key, val in self.x.items(): 
            i, j, r, k = key[2:-1].split(',')
            i = int(i)
            j = int(j)
            k = int(k)
            r = int(r)
            od_matrix[i-1, j-1, k] += val

        # Testing, random od
        # od_matrix = np.random.randint(low = 10, high = 100, size = (15, 15, 10))

        # loss, _, _, _ = self.simulation(demand = od_matrix)
        # assert loss > 0.0
        
        return 0

    def benchmark(self):
        gt_od_matrix = np.zeros((len(self.I), len(self.J), len(self.K)))
        for key, val in self.x_gt.items(): 
            i, j, k = key
            gt_od_matrix[i-1, j-1, k] += val

        est_od_matrix = np.zeros((len(self.I), len(self.J), len(self.K)))
        for key, val in self.x_ijk.items():
            i, j, k = key
            est_od_matrix[i-1, j-1, k] += val
            
        # with open('check.csv', 'w') as f:
        #     f.write('i,j,k,gt,est\n')
        #     for i in range(len(self.I)):
        #         for j in range(len(self.J)):
        #             for k in range(len(self.K)):
        #                 f.write(f'{i+1},{j+1},{k},{gt_od_matrix[i,j,k]},{est_od_matrix[i,j,k]}\n')
        # assert False

        gt_sumo_sim_args = {
            'duration': 1800, # Default simulation time
            'period': 180, # Default detector work cycle
            'data': r'PWD/gt/simulation_test', # Default sensor XML data path
            'seed': 2025, # Default seed for reproducibility
            'config': r'PWD/gt/test.sumocfg', # Default SUMO configuration path
            'mute_warnings': True, # Default to not mute warnings
            'mute_step_logs': False # Default to not mute step logs
        }
        gt_sumo_sim_args = argparse.Namespace(**gt_sumo_sim_args)
        self.sumo_sim = SumoSimulation(gt_sumo_sim_args)
        # run SUMO simulation with ground truth demand
        gt_q, gt_k, gt_v = self.sumo_sim.run_sumo(od_matrix = gt_od_matrix)

        est_sumo_sim_args = {
            'duration': 1800, # Default simulation time
            'period': 180, # Default detector work cycle
            'data': r'PWD/est/simulation_test', # Default sensor XML data path
            'seed': 2025, # Default seed for reproducibility
            'config': r'PWD/est/test.sumocfg', # Default SUMO configuration path
            'mute_warnings': True, # Default to not mute warnings
            'mute_step_logs': False # Default to not mute step logs
        }
        est_sumo_sim_args = argparse.Namespace(**est_sumo_sim_args)
        self.sumo_sim = SumoSimulation(est_sumo_sim_args)
        # run SUMO simulation with estimated demand
        est_q, est_k, est_v = self.sumo_sim.run_sumo(od_matrix = est_od_matrix)

        # draw pearson correlation

        # Flatten the OD matrices for correlation
        gt_flat = gt_od_matrix.flatten()
        est_flat = est_od_matrix.flatten()

        # Calculate Pearson correlation
        r_matrix, p_value = pearsonr(gt_flat, est_flat)

        # Plot scatter with regression line
        plt.figure(figsize=(8, 8))
        plt.scatter(gt_flat, est_flat, color='blue', alpha=0.5, label='Data Points')
        m, b = np.polyfit(gt_flat, est_flat, 1)
        plt.plot(gt_flat, m * gt_flat + b, color='red', label=f'Fit (r={r_matrix:.2f})')
        plt.title('Pearson Correlation: Ground Truth vs Estimated OD Matrix')
        plt.xlabel('Ground Truth')
        plt.ylabel('Estimated')
        plt.legend()
        plt.grid(True)
        plt.xticks([])
        plt.yticks([])
        plt.savefig(os.path.join(self.pwd, 'pearson_correlation_od_matrix.png'))
        plt.show()
        


        gt_flat = gt_q.ravel()  # Flattens the (150, 10) array into a 1D array of length 1500 
        est_flat = est_q.ravel()

        # Calculate Pearson correlation
        r, p = pearsonr(gt_flat, est_flat)

        # Plot scatter with regression line
        plt.figure(figsize=(8, 8))
        plt.scatter(gt_flat, est_flat, color='blue', alpha=0.5, label='Data Points')
        m, b = np.polyfit(gt_flat, est_flat, 1)  # Linear regression [[4]]
        plt.plot(gt_flat, m * gt_flat + b, color='red', label=f'Regression Line (r = {r:.2f})')
        plt.title('Pearson Correlation: Ground Truth vs Estimated')
        plt.xlabel('Ground Truth')
        plt.ylabel('Estimated')
        plt.legend()
        plt.grid(True)
        plt.xticks([])
        plt.yticks([])
        plt.savefig(os.path.join(self.pwd, 'pearson_correlation_q.png'))
        plt.show()
        



        

# TODO: linear constraints
class DODME_linear(DODME): 
    def __init__(self, config, g, n, v, d, t, y): 
        super().__init__(config, g, n, v, d, t, y)

    def generate_model(self, verbose=False, presolve=True):
        model = gp.Model()
        # mute console
        if not verbose: 
            model.setParam('LogToConsole', 0)
            model.setParam('OutputFlag', 0)
        # enable presolve module
        if not presolve: 
            model.setParam('Presolve', 0)
        # set time limit
        model.setParam('TimeLimit', self.time_limit)
        # initialize variables
        # self.X = model.addVars(self.I, self.J, self.R, self.K, vtype=GRB.CONTINUOUS, name='X', lb=0) # x_{ijrk}
        # self.T = model.addVars(self.I, self.J, self.K, vtype=GRB.CONTINUOUS, name='T', lb=0)
        self.X = model.addVars(self.ij, self.R, self.K, vtype=GRB.CONTINUOUS, name='X', lb=0) # x_{ijrk}
        self.T = model.addVars(self.ij, self.K, vtype=GRB.CONTINUOUS, name='T', lb=0)
        # decision variables
        # self.A = model.addVars(self.I, self.J, self.R, self.K, vtype=GRB.BINARY, name='A')
        # self.B = model.addVars(self.I, self.J, self.R, self.K, vtype=GRB.BINARY, name='B')
        self.A = model.addVars(self.ij, self.R, self.K, vtype=GRB.BINARY, name='A')
        self.B = model.addVars(self.ij, self.R, self.K, vtype=GRB.BINARY, name='B')
        # self.W = model.addVars(self.L, self.K, vtype=GRB.CONTINUOUS, name='W') # w_{lk}
        self.W = model.addVars(self.lk, vtype=GRB.CONTINUOUS, name='W') # w_{lk}
        # calculate y estimation
        y_est = self._calc_y_est()
        # constraint set A {l, k}
        for (l, k) in self.lk: 
            try: 
                model.addConstr(self.W[l, k] >= self.y_hat[l, k] - y_est[l, k])
                model.addConstr(self.W[l, k] >= -self.y_hat[l, k] + y_est[l, k])  
            except KeyError: 
                assert False    
        # constraint set B {i, j, k}
        for (i, j) in self.ij: 
            for k in self.K: 
                for r in self.R: 
                    try: 
                        model.addConstr(self.A[i, j, r, k] + self.B[i, j, r, k] >= 1)
                        model.addConstr(self.X[i, j, r, k] <= (1 - self.A[i, j, r, k]) * BIG_M)
                        model.addConstr(-self.T[i, j, k] <= -self.t[(i, j), r, k] + (1 - self.B[i, j, r, k]) * BIG_M)
                        model.addConstr(self.T[i, j, k] <= self.t[(i, j), r, k])
                    except KeyError: 
                        assert False
        # initialize and save self.model
        self.model = model
        return model
