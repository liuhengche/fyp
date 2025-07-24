# @date: 2024.12.04
# @desc: data process module
# @author: xyli45@um.cityu.edu.hk

import os
import glob
import pandas as pd
import xml.etree.ElementTree as ET

from statistics import mean
from tqdm import tqdm
from scripts.base import Edge, Route
from utils import save_as_pkl

class DataLoader(): 
    def __init__(self, config): 
        self.detector_raw_data_path          = os.path.join(config['PWD'], config['DETECTOR_RAW_DATA_PATH'])
        self.detector_link_mapping_path      = os.path.join(config['PWD'], config['DETECTOR_LINK_MAPPING_PATH'])
        self.detector_link_mapping_save_path = os.path.join(config['PWD'], config['DETECTOR_LINK_MAPPING_SAVE_PATH'])

        self.link_density_path     = os.path.join(config['PWD'], config['LINK_DENSITY_PATH'])
        self.link_mean_speed_path    = os.path.join(config['PWD'], config['LINK_MEAN_SPEED_PATH']) 
        self.link_traffic_count_path = os.path.join(config['PWD'], config['LINK_TRAFFIC_COUNT_PATH'])

        self.network_mean_speed_path = os.path.join(config['PWD'], config['NETWORK_MEAN_SPEED_PATH'])
        
        self.time_interval   = config['TIME_INTERVAL']
        self.time_interval_n = config['TIME_INTERVAL_N']
        
        self.s = config['SPACE_MEAN_EFFECTIVE_VEHICLE_LEN']

        os.makedirs(config['PWD'], exist_ok=True)

        # map detector to link
        # self.detector2link = map_detector2link(
        #     map_path=self.detector_link_mapping_path, 
        #     save_path=self.detector_link_mapping_save_path
        # )
        
    def _time2index(self, time):
        ''' time: float, index: int '''
        index = int(time / self.time_interval)
        assert index >= 0
        assert index < self.time_interval_n
        return index
    
    # Deprecated
    # def run(self, verbose=True): 
    #     '''
    #     input: detectors.xml
    #     output: link_mean_speed {l, k}, link_traffic_count {l, k}, link_density {l, k}
    #     '''
        
    #     def _calc_v(speed): 
    #         ''' calculate mean speed '''
    #         if speed == -1.00: 
    #             return 0.0
    #         return 3.6 * speed
        
    #     def _calc_d(occupancy): 
    #         ''' calculate mean density '''
    #         return (occupancy / 100) / self.s
        
    #     # map detector to link
    #     self.detector2link = map_detector2link(
    #         map_path=self.detector_link_mapping_path, 
    #         save_path=self.detector_link_mapping_save_path
    #     )
        
    #     link_density  = {}
    #     link_mean_speed = {}
    #     link_traffic_count = {}

    #     network_mean_speed = {}

    #     tree = ET.parse(self.detector_raw_data_path)
    #     root = tree.getroot()
    #     iterable = root.iter('interval')
    #     bar = tqdm(iterable, desc='run data preprocess...'.ljust(30)) if verbose else iterable
    #     for interval in bar:
    #         # speed
    #         speed = float(interval.get('speed'))
    #         speed = _calc_v(speed)
    #         # id
    #         id    = interval.get('id')
    #         # time
    #         begin = float(interval.get('begin'))
    #         # volume
    #         volume = int(interval.get('nVehEntered'))
    #         # occupancy
    #         occupancy = float(interval.get('occupancy')) # TODO: (%)
    #         density   = _calc_d(occupancy)
            
    #         l = self.detector2link[id]
    #         k = self._time2index(begin)
    #         if verbose: 
    #             bar.set_postfix(l=l, k=k)
    #         # volume
    #         link_traffic_count.setdefault((l, k), 0) 
    #         link_traffic_count[l, k] += volume
    #         # mean speed
    #         link_mean_speed.setdefault((l, k), 0)
    #         link_mean_speed[l, k] += (volume * speed)
    #         # density
    #         link_density.setdefault((l, k), [])
    #         # check if occupancy > 0.0
    #         if density > 0.0: 
    #             link_density[l, k].append(density)
    #     # average
    #     for key, val in link_density.items():
    #         if len(val) <= 1: 
    #             link_density[key] = sum(val)
    #         else: 
    #             link_density[key] = mean(val) 
    #     for key, val in link_mean_speed.items():
    #         l, k = key 
    #         if link_traffic_count[key] > 0:
    #             avg_speed = (val / link_traffic_count[key])
    #         else: 
    #             avg_speed = 0
    #         # link mean speed
    #         link_mean_speed[key] = avg_speed
    #         # network mean speed
    #         network_mean_speed.setdefault(k, [])
    #         network_mean_speed[k].append(avg_speed)
    #     # network mean speed
    #     for key, val in network_mean_speed.items():
    #         network_mean_speed[key] = mean(val)
    #     # save data
    #     save_as_pkl(network_mean_speed, self.network_mean_speed_path)
    #     save_as_pkl(link_traffic_count, self.link_traffic_count_path)
    #     save_as_pkl(link_mean_speed, self.link_mean_speed_path)
    #     save_as_pkl(link_density, self.link_density_path)
    #     return network_mean_speed, link_traffic_count, link_mean_speed, link_density

    def run(self, verbose=True):
        
        def _calc_v(speed): 
            ''' calculate mean speed (unit: km/h) '''
            if speed == -1.00: 
                return 0.0
            return 3.6 * speed
        
        self.detector2link = map_detector2link(
            map_path=self.detector_link_mapping_path, 
            save_path=self.detector_link_mapping_save_path
        )
        data = {}
        link_density  = {}
        link_mean_speed = {}
        link_flow = {}
        network_mean_speed = {}

        tree = ET.parse(self.detector_raw_data_path)
        root = tree.getroot()
        iterable = root.iter('interval')
        bar = tqdm(iterable, desc='run data preprocess...'.ljust(30)) if verbose else iterable
        # data io
        for interval in bar:
            # speed (km/h)
            time_mean_speed  = _calc_v(float(interval.get('speed')))
            space_mean_speed = _calc_v(float(interval.get('harmonicMeanSpeed'))) 
            # id
            id = interval.get('id')
            # time
            begin = float(interval.get('begin'))
            # flow (veh/h)
            flow = float(interval.get('flow'))
            l = self.detector2link[id]
            k = self._time2index(begin)
            data.setdefault((l, k), [])
            # v_t, v_s, f => q = \sum f, k = q^2 / (\sum f * v_s), v = (\sum f * v_t) / q
            data[l, k].append((time_mean_speed, space_mean_speed, flow))
        # process
        for (l, k), lanes in data.items(): 
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
            # for l, k
            link_density[l, k]    = k_
            link_mean_speed[l, k] = v_
            link_flow[l, k]       = q_
            # network mean speed
            network_mean_speed.setdefault(k, [])
            network_mean_speed[k].append(v_)
        # average
        for k, val in network_mean_speed.items(): 
            network_mean_speed[k] = mean(val)

        save_as_pkl(network_mean_speed, self.network_mean_speed_path)
        save_as_pkl(link_flow, self.link_traffic_count_path)
        save_as_pkl(link_mean_speed, self.link_mean_speed_path)
        save_as_pkl(link_density, self.link_density_path)
        return network_mean_speed, link_flow, link_mean_speed, link_density
    
class DataLoader_E1_all(DataLoader): 
    def run(self, verbose=True):
        
        def _calc_v(speed): 
            ''' calculate mean speed (unit: km/h) '''
            if speed == -1.00: 
                return 0.0
            return 3.6 * speed
        
        self.detector2link = map_detector2link(
            map_path=self.detector_link_mapping_path, 
            save_path=self.detector_link_mapping_save_path
        )
        data = {}
        link_density  = {}
        link_mean_speed = {}
        link_flow = {}
        network_mean_speed = {}

        tree = ET.parse(self.detector_raw_data_path)
        root = tree.getroot()
        iterable = root.iter('interval')
        bar = tqdm(iterable, desc='run data preprocess...'.ljust(30)) if verbose else iterable
        # data io
        for interval in bar:
            # speed (km/h)
            time_mean_speed  = _calc_v(float(interval.get('speed')))
            space_mean_speed = _calc_v(float(interval.get('harmonicMeanSpeed'))) 
            # id
            id = interval.get('id')
            # time
            begin = float(interval.get('begin'))
            # flow (veh/h)
            flow = float(interval.get('flow'))
            l = self.detector2link[id]
            k = self._time2index(begin)
            data.setdefault((l, k), [])
            # v_t, v_s, f => q = \sum f, k = q^2 / (\sum f * v_s), v = (\sum f * v_t) / q
            data[l, k].append((time_mean_speed, space_mean_speed, flow))
        # process
        for (l, k), lanes in data.items(): 
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
            # for l, k
            link_density[l, k]    = k_
            link_mean_speed[l, k] = v_
            link_flow[l, k]       = q_
            # network mean speed
            network_mean_speed.setdefault(k, [])
            network_mean_speed[k].append(v_)
        # average
        for k, val in network_mean_speed.items(): 
            network_mean_speed[k] = mean(val)

        save_as_pkl(network_mean_speed, self.network_mean_speed_path)
        save_as_pkl(link_flow, self.link_traffic_count_path)
        save_as_pkl(link_mean_speed, self.link_mean_speed_path)
        save_as_pkl(link_density, self.link_density_path)
        return network_mean_speed, link_flow, link_mean_speed, link_density

# NOTE: implemented for dataset: strategy and major roads
class DataLoader_StrategyMajorRoad(DataLoader):
    def __init__(self, config):
        super().__init__(config)
        # hard code time interval 
        assert self.time_interval == 30 
        # file list
        self.files = []
        for file in glob.glob(os.path.join(self.detector_raw_data_path, '*.csv')): 
            self.files.append(file)
        # check integrity
        assert len(self.files) == self.time_interval_n

    def run(self, verbose=True):
        
        def _calc_d(occupancy): 
            ''' calculate mean density '''
            return (occupancy / 100) / self.s
        
        # map detector to link (real data omit lane index)
        self.detector2link = map_detector2link(
            map_path=self.detector_link_mapping_path, 
            save_path=self.detector_link_mapping_save_path, 
            omit_lane=True
        )
        
        k = 0
        link_density  = {}
        link_mean_speed = {}
        link_traffic_count = {}
        network_mean_speed = {}

        bar = tqdm(self.files, desc='run data preprocess...'.ljust(30)) if verbose else self.files
        for file in bar:
            data = pd.read_csv(file)
            for row in data.itertuples(): 
                detector_id = getattr(row, 'detector_id')
                speed       = getattr(row, 'speed')
                occupancy   = getattr(row, 'occupancy')
                volume      = getattr(row, 'volume')
                density     = _calc_d(occupancy)
                # TODO: [DEBUG] missing link
                if detector_id in self.detector2link: 
                    l = self.detector2link[detector_id]
                else: 
                    print(detector_id)
                    continue
                if verbose: 
                    bar.set_postfix(l=l, k=k)
                # volume
                link_traffic_count.setdefault((l, k), 0) 
                link_traffic_count[l, k] += volume
                # mean speed
                link_mean_speed.setdefault((l, k), 0)
                link_mean_speed[l, k] += (volume * speed)
                # density
                link_density.setdefault((l, k), [])
                # check if occupancy > 0.0
                if density > 0.0: 
                    link_density[l, k].append(density)
            # k++
            k += 1
        # check integrity
        assert k == self.time_interval_n
        # average
        for key, val in link_density.items():
            if len(val) <= 1: 
                link_density[key] = sum(val)
            else: 
                link_density[key] = mean(val) 
        for key, val in link_mean_speed.items():
            l, k = key 
            if link_traffic_count[key] > 0:
                avg_speed = (val / link_traffic_count[key])
            else: 
                avg_speed = 0
            # link mean speed
            link_mean_speed[key] = avg_speed
            # network mean speed
            network_mean_speed.setdefault(k, [])
            network_mean_speed[k].append(avg_speed)
        # network mean speed
        for key, val in network_mean_speed.items():
            network_mean_speed[key] = mean(val)
        # save data
        save_as_pkl(network_mean_speed, self.network_mean_speed_path)
        save_as_pkl(link_traffic_count, self.link_traffic_count_path)
        save_as_pkl(link_mean_speed, self.link_mean_speed_path)
        save_as_pkl(link_density, self.link_density_path)
        return network_mean_speed, link_traffic_count, link_mean_speed, link_density

def load_graph_from_osm(osm_path, route_path, edge_save_path, route_save_path):
    '''
    input: osm.net.xml
    output: {V, E, V_id, edges}
    '''
    
    def load_route_fixed(route_path, edges):
        '''
        input: route.csv
        output: {od_routes}
        '''
        od_routes = {}
        routes    = pd.read_csv(route_path)
        for row in routes.itertuples(): 
            o = int(getattr(row, 'Origin').split('_')[0])
            d = int(getattr(row, 'Destination').split('_')[0])
            route = getattr(row, 'Route').split(' ')
            od_routes.setdefault((o, d), [])
            r = Route(o, d, route)
            r.load_link_len(edges)
            od_routes[(o, d)].append(r)
        return od_routes
    
    def load_route_candidate(route_path, edges):
        '''
        input: route.csv
        output: {od_routes}
        ''' 
        routes = pd.read_csv(route_path)
        od_routes = {}
        for row in routes.itertuples(): 
            o = int(getattr(row, 'ODpair').split('OD')[1])
            d = int(getattr(row, 'ODpair').split('OD')[2])
            route = getattr(row, 'Routechoice').split('->')
            od_routes.setdefault((o, d), [])
            r = Route(o, d, route)
            r.load_link_len(edges)
            od_routes[(o, d)].append(r)
        return od_routes
      
    V = {}
    V_id  = {}
    edges = {}
    to2od = {}
    neighbors = {}
    
    edge_n = 0
    node_n = 0

    # read osm
    tree = ET.parse(osm_path)
    root = tree.getroot()
    # node
    for node in root.iter('junction'): 
        id = node.get('id')
        if V.setdefault(id, -1) == -1: 
            V[id]        = node_n
            V_id[node_n] = id
            node_n += 1
        else: 
            assert False, f'node hash conflict({id})!'
    # edge
    for edge in root.iter('edge'):
        # length (unit: km)
        length = 0
        for lane in edge.iter('lane'): 
            length = lane.get('length')
            break
        if edge.get('function') == 'internal':
            continue
        
        id   = edge.get('id') # name
        f, t = edge.get('from'), edge.get('to')
        assert id not in edges, f'edge hash conflict({id})!'
        if id[:2] == 'OD' and len(id) < 5: 
            id_write = 0
            id_write = int(id.split('OD')[1])
            neighbors.setdefault(id_write, [])
            to2od.setdefault(t, id_write)
        if f in to2od:
            neighbors[to2od[t]].append(id)          
        # if not (f==None and t==None): 
        edges[id] = Edge(f, t, id, float(length) / 1000.0)
        edge_n   += 1
    # load routes
    # od_routes = load_route_fixed(route_path, edges)
    od_routes = load_route_candidate(route_path, edges)
    # save 
    save_as_pkl(edges, edge_save_path)
    save_as_pkl(od_routes, route_save_path)

    return node_n, edge_n, neighbors

def map_detector2link(map_path, save_path, omit_lane=False): 
    tree = ET.parse(map_path)
    root = tree.getroot()
    detector2link = {}
    for loop_detector in root.iter('inductionLoop'): 
        # NOTE: real sensor data omit lane index
        if omit_lane: 
            id = loop_detector.get('id')[:-2]
        else: 
            id = loop_detector.get('id')
        link = loop_detector.get('lane')[:-2]
        detector2link[id] = link 
    save_as_pkl(data=detector2link, pkl_path=save_path)
    return detector2link

def load_demand_gt_from_xml(route_path, save_path, period):
    tree = ET.parse(route_path)
    root = tree.getroot()
    v = {}
    x = {}
    # vehicle
    for v in root.iter('vehicle'): 
        route_id    = v.get('route')
        depart_time = float(v.get('depart'))
        v[route_id] = depart_time
    # route
    for r in root.iter('route'): 
        route_id    = r.get('id')
        edges       = r.get('edges')
        depart_time = v[route_id]
        origin      = int(edges.split(' ')[0][:-2])
        destination = int(edges.split(' ')[-1][:-2])
        time_k      = int(depart_time / period)
        x.setdefault(key = (origin, destination, time_k), default=0)
        x[origin, destination, time_k] += 1
    save_as_pkl(data=x, pkl_path=save_path)
    return x

def load_demand_gt_from_csv(gt_path, save_path, time_dependent=True):
    demand_csv = pd.read_csv(gt_path)
    od_demand = {}
    for row in demand_csv.itertuples(): 
        o = int(getattr(row, 'Origin').split('OD')[1])
        d = int(getattr(row, 'Destination').split('OD')[1])
        k = int(getattr(row, 'k')) if time_dependent else None
        demand = int(getattr(row, 'Value'))
        if time_dependent: 
            od_demand[(o, d, k)] = demand
        else: 
            od_demand[(o, d)] = demand
    save_as_pkl(data=od_demand, pkl_path=save_path)
    return od_demand
