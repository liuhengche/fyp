import os
import sys
import traci
import argparse
import subprocess
import pandas as pd
import numpy as np
import xml.etree.ElementTree as ET

from glob import glob

class SumoSimulation(): 
    def __init__(self, args):
        self.duration   = args.duration
        self.period     = args.period
        self.data_dir   = args.data
        self.seed       = args.seed
        self.config     = args.config
        self.mute_warnings  = args.mute_warnings
        self.mute_step_logs = args.mute_step_logs
        self.interval_n = (args.duration // args.period)

        self.counter = 0
        # check environment variable
        if 'SUMO_HOME' not in os.environ:
            assert False, f'Check SUMO_HOME environment variable'
        else:
            tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
            sys.path.append(tools)

            # self.sumo_bin = os.path.join(os.environ['SUMO_HOME'], 'bin\\sumo-gui')

    def _load_demand(self, x):
        ''' x: origin-destination demand matrix (unit: veh/h, shape: (i, j, k)) '''
        res = []
        i_, j_, k_ = x.shape
        for i in range(i_):
            for j in range(j_): 
                for k in range(k_): 
                    if i != j: 
                        res.append((i+1, j+1, k, x[i, j, k]))
        return res
    
    def _generate_routes(self):
        od_list = [f"OD{i}" for i in range(1, 16)]  # 生成 OD1, OD2, ..., OD15
        for origin in od_list:
            for destination in od_list:
                if origin != destination:  # 排除起点和终点相同的情况
                    route_id = f"{origin}{destination}"  # 生成唯一的 route ID
                    try:
                        traci.route.add(route_id, [origin, destination])  # 添加路线
                    except traci.exceptions.TraCIException as e:
                        assert False, f"Route {route_id} could not be added: {e}"

    def _generate_vehicle_flow(self, route_data):
        for origin, destination, k, v in route_data:
            route_id = f"OD{origin}OD{destination}"  # 使用 CSV 文件中的 origin 和 destination
            # NOTE: value (unit: veh/h)
            veh_n = int((v/3600) * self.period)
            if veh_n > 0:
                # 根据 k 值来确定发车的时间间隔
                interval_start_time = k * self.period  # 计算车辆的发车时间
                time_step_gap = self.period / veh_n  # 每辆车的发车间隔
                for vehicle_index in range(veh_n):
                    depart_time = interval_start_time + vehicle_index * time_step_gap
                    vehicle_id = f"{k}_{route_id}_{vehicle_index + 1}"  # 生成车辆 ID
                    try:
                        traci.vehicle.add(vehicle_id, routeID=route_id, depart=depart_time)
                    except traci.exceptions.TraCIException as e:
                        print(f"Vehicle {vehicle_id} could not be added: {e}")

    def _extract_measurements(self):
        def extract_single_file(filename): 
            data = {}
            tree = ET.parse(filename)
            root = tree.getroot()
            # load
            for interval in root.iter('interval'): 
                v_t, v_s = float(interval.get('speed')), float(interval.get('harmonicMeanSpeed'))
                time_mean_speed  = 0.0 if v_t == -1.00 else v_t 
                space_mean_speed = 0.0 if v_s == -1.00 else v_s 
                begin = float(interval.get('begin'))
                flow  = float(interval.get('flow'))

                k = (begin // self.period)
                
                data.setdefault(k, [])
                data[k].append((time_mean_speed, space_mean_speed, flow))
            # process
            q, k, v = [], [], []
            for _, lanes in data.items(): 
                q_, k_, v_ = 0, 0, 0
                tmp_vt, tmp_vs = 0, 0
                # for m lanes
                for lane in lanes:
                    v_t, v_s, f = lane
                    q_     += f 
                    tmp_vt += f * v_t
                    tmp_vs += f * v_s
                k_ = (q_ * q_) / tmp_vs if tmp_vs > 0 else 0
                v_ = tmp_vt / q_ if q_ > 0 else 0
                q.append(q_)
                k.append(k_)
                v.append(v_)
            return q, k, v

        ''' simulation measurements: {q[l, k], k[l, k], v[l, k]} '''
        q, k, v = [], [], []
        for file in glob(os.path.join(self.data_dir, '*.xml')):
            q_l, k_l, v_l = extract_single_file(file)
            assert len(q_l) == self.interval_n, f"Length of measurements {len(q_l)} does not match expected intervals {self.interval_n}"
            q.append(q_l)
            k.append(k_l)
            v.append(v_l)
        return np.array(q), np.array(k), np.array(v)
    
    def set_config(self, config): 
        ''' simulation configuration '''
        self.config = config

    def set_data_dir(self, data_dir): 
        ''' simulation data save path '''
        self.data_dir = data_dir
    
    def get_simulation_count(self): 
        return self.counter

    def run_sumo(self, od_matrix):
        self.counter += 1 
        # load demand
        demand = self._load_demand(od_matrix)
        # run sumo
        cmd = ["sumo", "-c", self.config]
        # seed for reproductivity
        cmd.extend(['--seed', str(self.seed)])
        # mute warnings
        cmd.extend(['-W', str(self.mute_warnings)])
        # no step log
        cmd.extend(['--no-step-log', str(self.mute_step_logs)])
        # no teleport
        cmd.extend(['--time-to-teleport', str(-1)])
        # # save error logs
        # if error_log is not None: 
        #     cmd.extend(['--error-log', str(error_log)])
        # # save individual data
        # if tripinfo_output is not None:
        #     cmd.extend(['--tripinfo-output', tripinfo_output])
        # if fcd_output is not None:
        #     cmd.extend([ '--fcd-output', fcd_output])
        traci.start(cmd)
        self._generate_routes()
        self._generate_vehicle_flow(demand)
        # TODO: trace down
        for _ in range(self.duration):
            traci.simulationStep()
        traci.close()
        return self._extract_measurements()
    
class SumoSimulationParallel(SumoSimulation):
    def __init__(self, args):
        super().__init__(args)
        self.flow_save_path = args.flow

    def _load_demand(self, x: np.ndarray):
        i_, j_, k_ = x.shape
        res = [[] for _ in range(k_)]
        for i in range(i_):
            for j in range(j_): 
                for k in range(k_): 
                    if i != j: 
                        res[k].append((i + 1, j + 1, x[i, j, k]))
        return res

    def _generate_vehicle_flow(self, route_data: list):
        with open(self.flow_save_path, 'w') as f:
            # write header 
            f.write('<?xml version="1.0"?>\n')
            f.write('<routes>\n')
            # TODO: change cf model & lc model
            f.write('   <vType id="CAV" vClass="passenger" color="yellow" minGap="2.5" accel="2.6" decel="4.5" maxSpeed="33.33" speedFactor="norm(1,0.05)"/>\n')
            for k, flow_list in enumerate(route_data):
                for (origin, destination, v) in flow_list: 
                    if v > 0: 
                        flow_id = f'{origin}-{destination}-{k}'
                        begin = k * self.period
                        end = (k + 1) * self.period
                        f.write(f'    <flow id="{flow_id}" begin="{begin}" end="{end}" vehsPerHour="{v}" from="OD{origin}" to="OD{destination}">\n')
                        f.write('   </flow>\n')
            f.write('</routes>')
        f.close()

    def run_sumo(self, od_matrix):
        self.counter += 1 
        # load demand
        demand = self._load_demand(od_matrix)
        self._generate_vehicle_flow(demand)
        # run sumo
        cmd = ['sumo', '-c', self.config]
        # seed for reproductivity
        cmd.extend(['--seed', str(self.seed)])
        # mute warnings
        cmd.extend(['-W', str(self.mute_warnings)])
        # no step log
        cmd.extend(['--no-step-log', str(self.mute_step_logs)])
        # no teleport
        cmd.extend(['--time-to-teleport', str(-1)])
        
        # TODO: save tripinfo
        # if tripinfo_output is not None:
        #     cmd.extend(['--tripinfo-output', tripinfo_output])
        
        # TODO: save fcd
        # if fcd_output is not None:
        #     cmd.extend([ '--fcd-output', fcd_output])

        subprocess.run(cmd, check=True)
        return self._extract_measurements()
    
# def nrmse(m_sim, m_obs, mse=None):
#     ''' implement normalized root mean square error '''
#     assert m_sim.shape == m_obs.shape
#     L, K = m_sim.shape 
#     # if mse is known
#     if mse is not None: 
#         square_error = L * K * mse
#         return np.sqrt(L * K * square_error) / np.sum(m_obs)
    
#     return np.sqrt(L * K * np.sum((m_sim - m_obs) * (m_sim - m_obs))) / np.sum(m_obs)

# class SumoObjectiveFunction():
#     def __init__(self, 
#                  observations: list):
#         ''' flexible to a series of measurements, i.e., q, k, v '''
#         # self.sim = simulations  # [q_sim, k_sim, v_sim]
#         assert len(observations) > 0
#         self.obs = observations # [q_obs, k_obs, v_obs]
#         self.gof = nrmse

#     def objective_value(self, simulations): 
#         def _weight(m_sim, m_obs):
#             ''' weight of GoF functions '''
#             L, K = m_sim.shape
#             sigma_sim = np.std(m_sim)
#             sigma_obs = np.std(m_obs)
#             mse = np.sum((m_sim - m_obs) * (m_sim - m_obs)) / (L * K) 
#             return (sigma_obs - sigma_sim) ** 2 / mse, mse
        
#         obj = 0
#         for m_sim, m_obs in zip(simulations, self.obs): 
#             w_m, mean_square_error  = _weight(m_sim, m_obs)
#             obj += w_m * self.gof(m_sim, m_obs, mse = mean_square_error)
#         return obj

def sumo_objective_function_nrmse(sim_measurements: list, obs_measurements: list):
    obj = 0
    loss_list = []
    weight_list = []
    # calculate weight, loss
    for sim_measurement, obs_measurement in zip(sim_measurements, obs_measurements): 
        assert sim_measurement.shape == obs_measurement.shape
        L, K = sim_measurement.shape
        sigma_sim, sigma_obs = np.std(sim_measurement), np.std(obs_measurement)
        square_error = np.sum((sim_measurement - obs_measurement) * (sim_measurement - obs_measurement))
        weight = square_error / L * K * (sigma_obs - sigma_sim) ** 2 
        loss = np.sqrt(L * K * square_error) / np.sum(obs_measurement)
        loss_list.append(loss)
        weight_list.append(weight)
    # calculate objective
    for loss, weight in zip(loss_list, weight_list):
        alpha = weight / sum(weight_list)
        obj += alpha * loss
    return obj
        

if __name__ == '__main__': 
    parser = argparse.ArgumentParser()
    parser.add_argument('--config',   type = str, default = 'test.sumocfg', help = 'sumo configuration path')
    parser.add_argument('--data',     type = str, default = 'simulation', help = 'sensor xml data path')
    parser.add_argument('--duration', type = int, default = 3600, help = 'simulation time')
    parser.add_argument('--period',   type = int, default = 900, help = 'detector work cycle')
    parser.add_argument('--seed',     type = int, default = 2025, help = 'for simulation reproductive')    
    parser.add_argument('--mute_warnings',  action='store_true', default=False, help='mute sumo warnings')
    parser.add_argument('--mute_step_logs', action='store_true', default=False, help='mute step logs')
    args = parser.parse_args()
    interval_n = (args.duration // args.period)

    # x = np.random.randint(low = 10, high = 20, size = (15, 15, interval_n))
    # read csv
    demand_csv = pd.read_csv(r'C:\Users\COSI\Desktop\projects\crawlerTesting\test_route_choice_2025_03_19\demand_gt.csv')
    od_demand = np.zeros((15, 15, 60))
    for row in demand_csv.itertuples(): 
        o = int(getattr(row, 'Origin').split('OD')[1])
        d = int(getattr(row, 'Destination').split('OD')[1])
        k = int(getattr(row, 'k'))
        demand = int(getattr(row, 'Value'))
        od_demand[o-1, d-1, k] = demand * 20

    simulation = SumoSimulation(args)
    simulation.run_sumo(od_matrix = od_demand)

