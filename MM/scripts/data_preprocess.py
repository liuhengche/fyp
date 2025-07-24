import os
import glob
import numpy as np
import pandas as pd
import xml.etree.ElementTree as ET

from tqdm import tqdm
from scripts.base import Edge, Route
from statistics import mean
from utils import iter_time, load_from_pkl, save_as_pkl, preprocess_df, sum_df

IS_ARTIFICIAL = True

# 
# input:         
#         'DETECTOR_LINK_MAPPING_PATH': 'detectors.add.30%.xml', 
#         'ROAD_NETWORK_PATH':          'osm.net.xml',           
#         'RAW_ROUTE_CHOICE_PATH':      'route_results.csv',     
# 

class DataLoader():
    def __init__(self, start_date, end_date, config, verbose=False): 
        self.config   = config
        self.data     = {}
        self.detector = {}
        self.start    = start_date
        self.end      = end_date
        # self.duration = iter_time(start_date, end_date)
        
        self.space_mean_effective_vehicle_len = config['SPACE_MEAN_EFFECTIVE_VEHICLE_LEN']
        self.time_interval_n                  = config['N_TIME_INTERVAL']
        self.detector_time_interval           = config['DETECTOR_TIME_INTERVAL']
        self.nfd_dot_time_interval            = config['NFD_DOT_TIME_INTERVAL']

        # self.pwd               = config['PWD']
        self.raw_data_dir      = config['RAW_DATA_DIR']
        # self.detector_loc_path = config['DETECTOR_LOC_PATH']

        self.detector_link_mapping_path      = os.path.join(config['PWD'] , config['DETECTOR_LINK_MAPPING_PATH'])
        self.detector_link_mapping_save_path = os.path.join(config['PWD'] , config['DETECTOR_LINK_MAPPING_SAVE_PATH'])

        self.network_mean_speed_path = os.path.join(config['PWD'], config['NETWORK_MEAN_SPEED_PATH']) 
        self.link_traffic_count_path = os.path.join(config['PWD'], config['LINK_TRAFFIC_COUNT_PATH'])
        
        # self.road_network_path          = config['ROAD_NETWORK_PATH']
        # self.raw_route_choice_path      = config['RAW_ROUTE_CHOICE_PATH']
        # self.network_mean_speed_path    = config['NETWORK_MEAN_SPEED_PATH']
        # self.travel_time_path           = config['TRAVEL_TIME_PATH']
        # self.route_choice_dict_path     = config['ROUTE_CHOICE_DICT_PATH']
        # self.edge_dict_path             = config['EDGE_DICT_PATH'] 

        os.makedirs(config['PWD'], exist_ok=True)

        # map detector to link
        self.detector2link = map_detector2link(
            map_path=self.detector_link_mapping_path, 
            save_path=self.detector_link_mapping_save_path
        )

        # check integrity
        if verbose: 
            print('Check integrity...')
        for date in self.duration: 
            lis = glob.glob(f'{self.raw_data_dir}\\{date[:4]}-{date[4:6]}-{date[6:8]}\\*.csv')
            assert len(lis) > 0, 'Found missing data!'
        # load data
        for date in iter_time(start_date, end_date): 
            self.data[date] = []
            file_lis = glob.glob(f'{self.raw_data_dir}\\{date[:4]}-{date[4:6]}-{date[6:8]}\\*.csv')
            bar = tqdm(file_lis, desc='load data from {:04d}-{:02d}-{:02d}'.format(int(date[:4]),int(date[4:6]),int(date[6:8])).ljust(20)) if verbose else file_lis
            for file in bar: 
                self.data[date].append(file)

    def _clean_df(self, df):
        table = (df['volume'] == 0)
        invalid_rows = []
        for idx, not_valid in enumerate(table): 
            if not_valid: 
                invalid_rows.append(idx)
        invalid_n, total_n = len(invalid_rows), len(table)
        return df.drop(index=invalid_rows), invalid_n, total_n

    def _calc_density(self, occ): 
        return occ / (self.space_mean_effective_vehicle_len * 100)

    # TODO: modify time to index
    def _time_interval2index(self, time):
        h, m, s = int(time[0:2]), int(time[2:4]), int(time[4:6])
        # res = h * 60 * 2 + m * 2 + (s // 30)
        res = h
        assert res >= 0 
        assert res < self.time_interval_n
        return res
    
    def _basename2time(self, basename): 
        return basename.split('-')[3]

    # TODO: deprecated
    # def _load_detector_info(self):
    #     assert os.path.exists(self.detector_loc_path), f'File not found {self.detector_loc_path}'
    #     detector_loc = pd.read_csv(self.detector_loc_path)
    #     for row in detector_loc.itertuples(): 
    #         # print(row)
    #         self.detector[row[1]] = [row[0], row[2], row[3], row[4], row[5], row[6], row[7], row[8], row[9], row[10], row[11]]

    def run_preprocess(self, start_date, end_date, verbose=True): 
        assert int(start_date) >= int(self.start)
        assert int(end_date)   <= int(self.end)
        network_mean_speeds     = []
        all_network_mean_speeds = {}
        all_traffic_counts      = {}
        traffic_counts          = dict.fromkeys(self.detector.keys(), 0)
        for date in iter_time(start_date, end_date):
            data_paths = self.data[date]
            all_network_mean_speeds[date] = [0 for _ in range(self.time_interval_n)]
            all_traffic_counts[date]      = [{} for _ in range(self.time_interval_n)]
            bar = tqdm(data_paths, desc='process data from {:04d}-{:02d}-{:02d}'.format(int(date[:4]),int(date[4:6]),int(date[6:8])).ljust(20)) if verbose else data_paths
            for data_path in bar:
                df = pd.read_csv(data_path)
                time = self._basename2time(os.path.basename(data_path)) # start time 
                id = self._time_interval2index(time) 

                # network mean speed
                nms = 0.0
                tot = df['volume'].sum() # total volume
                for row in df.itertuples(): 
                    detector_id = getattr(row, 'detector_id')
                    speed       = getattr(row, 'speed')
                    # occupancy   = getattr(row, 'occupancy')
                    volume      = getattr(row, 'volume')
                    traffic_counts[detector_id] += volume
                    if volume > 0: 
                        nms += (volume * speed) / tot # weighted average

                network_mean_speeds.append(nms)
                # TODO: dense
                if len(network_mean_speeds) >= (self.nfd_dot_time_interval // self.detector_time_interval): 
                    # empty cache
                    v_nms = mean(network_mean_speeds)
                    all_network_mean_speeds[date][id] = v_nms
                    all_traffic_counts[date][id] = traffic_counts
                    network_mean_speeds = []
                    traffic_counts = dict.fromkeys(self.detector.keys(), 0)
            
        # network mean speed, matrix N: {cols: date, rows: time interval k}
        nms_df = pd.DataFrame(all_network_mean_speeds)
        nms_df = preprocess_df(nms_df)
        nms_df = nms_df.replace(0.0, nms_df.mean().mean())
        save_as_pkl(nms_df, self.network_mean_speed_path)
        # link traffic count, matrix T: {cols: link, rows: time interval k} 
        for date, val in all_traffic_counts.items():
            ltc_df = pd.DataFrame(val)
            ltc_df = preprocess_df(ltc_df)
            ltc_df = ltc_df.replace(np.nan, ltc_df.mean().mean())
            save_as_pkl(ltc_df, os.path.join(self.config['PWD'], f'link_traffic_count_{date}.pkl'))
        return all_network_mean_speeds, all_traffic_counts

# TODO: new data loader
class DataLoader_v2(DataLoader):
    def _time_interval2index(self, time):
        res = (time // self.nfd_dot_time_interval)
        assert res >= 0 
        assert res < self.time_interval_n
        return res
    
    def _basename2time(self, basename):
        return int(basename.split('-')[0])

    def run_preprocess(self, start_date, end_date, verbose=True, unit_convert=False): 
        assert int(start_date) >= int(self.start)
        assert int(end_date)   <= int(self.end)
        network_mean_speeds     = []
        all_network_mean_speeds = {}
        all_traffic_counts      = {}
        # traffic_counts          = dict.fromkeys(self.detector.keys(), 0)
        for date in iter_time(start_date, end_date):
            data_paths = self.data[date]
            all_network_mean_speeds[date] = [0 for _ in range(self.time_interval_n)]
            all_traffic_counts[date]      = [{} for _ in range(self.time_interval_n)]
            bar = tqdm(data_paths, desc='process data from {:04d}-{:02d}-{:02d}'.format(int(date[:4]),int(date[4:6]),int(date[6:8])).ljust(20)) if verbose else data_paths
            for data_path in bar:
                df = pd.read_csv(data_path)
                time = self._basename2time(os.path.basename(data_path)) # start time 
                id = self._time_interval2index(time) 
                detector_lis = df['Detector_ID'].drop_duplicates().to_list()
                traffic_counts = dict.fromkeys(detector_lis, 0)

                # network mean speed
                nms = 0.0
                tot = df['Volume'].sum() # total volume
                for row in df.itertuples(): 
                    detector_id = getattr(row, 'Detector_ID')
                    speed = getattr(row, 'Speed') * 3.60 if unit_convert else getattr(row, 'speed')
                    # occupancy   = getattr(row, 'occupancy')
                    volume      = getattr(row, 'Volume')
                    traffic_counts[detector_id] += volume
                    if volume > 0: 
                        nms += (volume * speed) / tot # weighted average

                network_mean_speeds.append(nms)
                # TODO: dense
                if len(network_mean_speeds) >= (self.nfd_dot_time_interval // self.detector_time_interval): 
                    # empty cache
                    v_nms = mean(network_mean_speeds)
                    all_network_mean_speeds[date][id] = v_nms
                    all_traffic_counts[date][id] = traffic_counts
                    network_mean_speeds = []
                    traffic_counts = dict.fromkeys(self.detector.keys(), 0)
            
        # network mean speed, matrix N: {cols: date, rows: time interval k}
        nms_df = pd.DataFrame(all_network_mean_speeds)
        nms_df = preprocess_df(nms_df)
        nms_df = nms_df.replace(0.0, nms_df.mean().mean())
        save_as_pkl(nms_df, self.network_mean_speed_path)
        # link traffic count, matrix T: {cols: link, rows: time interval k} 
        for date, val in all_traffic_counts.items():
            ltc_df = pd.DataFrame(val)
            ltc_df = preprocess_df(ltc_df)
            ltc_df = ltc_df.replace(np.nan, ltc_df.mean().mean())
            save_as_pkl(ltc_df, os.path.join(self.config['PWD'], f'link_traffic_count_{date}.pkl'))
        return all_network_mean_speeds, all_traffic_counts

class DataLoader_v3(DataLoader): 
    def run_preprocess(self, start_date, end_date, verbose=True, unit_convert=True):
        assert int(start_date) >= int(self.start)
        assert int(end_date)   <= int(self.end)
        link_traffic_count = {}
        link_mean_speed = {}
        for date in iter_time(start_date, end_date):
            data_paths = self.data[date]
            bar = tqdm(data_paths, desc='process data from {:04d}-{:02d}-{:02d}'.format(int(date[:4]),int(date[4:6]),int(date[6:8])).ljust(20)) if verbose else data_paths
            # link traffic count: [date][l, k]
            # link mean speed: [date][l, k]
            link_traffic_count_today = {}
            link_mean_speed_today    = {}
            for data_path in bar:
                df = pd.read_csv(data_path)
                t = self._basename2time(os.path.basename(data_path)) # start time 
                k = self._time_interval2index(t) 
                # for time k
                for row in df.itertuples(): 
                    detector_id = getattr(row, 'detector_id')
                    volume      = getattr(row, 'volume')
                    speed       = getattr(row, 'speed') * 3.60 if unit_convert else getattr(row, 'speed')
                    # occupancy   = getattr(row, 'occupancy') TODO: use occupancy
                    link_id     = self.detector2link[detector_id]
                    # volume
                    if len(link_traffic_count_today.setdefault((link_id, k), [])) == 0:  
                        link_traffic_count_today[link_id, k].append(volume)
                    # mean speed
                    if len(link_mean_speed_today.setdefault((link_id, k), [])) == 0: 
                        link_mean_speed_today[link_id, k].append(volume * speed)
            # average
            for key, val in link_traffic_count_today.items(): 
                link_traffic_count_today[key] = mean(val)
            for key, val in link_mean_speed_today.items(): 
                link_mean_speed_today[key] = mean(val)
            link_traffic_count[date] = link_traffic_count_today
            link_mean_speed[date]    = link_mean_speed_today

        save_as_pkl(link_traffic_count, self.network_mean_speed_path)
        save_as_pkl(link_mean_speed, self.network_mean_speed_path)
        return link_traffic_count, link_mean_speed

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
        
    V = {}
    # E = {}
    V_id  = {}
    # E_id = {}
    edges = {}
    
    edge_n = 0
    node_n = 0

    # edge_conflict = 0
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
        
        id   = edge.get('id') # name
        f, t = edge.get('from'), edge.get('to')
        assert id not in edges, f'edge hash conflict({id})!'
        # TODO: remove from=None and to=None
        if not (f==None and t==None): 
            edges[id] = Edge(f, t, id, float(length) / 1000.0)
            edge_n   += 1
    # print(f'loaded {node_n} nodes, {edge_n} edges...')

    # load routes
    od_routes = load_route_fixed(route_path, edges)
    # save
    save_as_pkl(edges, edge_save_path)
    save_as_pkl(od_routes, route_save_path)
    return node_n, edge_n

# TODO: [deprecated]
# def load_detector_from_xml(detector_path, save_path): 
#     tree = ET.parse(detector_path)
#     root = tree.getroot()
#     cur_time = 0
#     buf = []
#     # check save path
#     if not os.path.exists(save_path): 
#         os.mkdir(save_path)

#     for entry in root.iter('interval'): 
#         begin = int(float(entry.get('begin')))
#         end   = int(float(entry.get('end')))
#         id    = entry.get('id')[:-2]
#         speed = float(entry.get('speed'))
#         occupancy   = float(entry.get('occupancy')) # TODO: (%)
#         nVehEntered = int(entry.get('nVehEntered'))
#         # skip null
#         if speed == -1.00: 
#             continue
#         # m/s => km/h
#         speed *= 3.60

#         if not cur_time == begin: 
#             # clean cache
#             filename = '{:05d}-{:05d}.csv'.format(cur_time, cur_time + (end - begin))
#             with open(os.path.join(save_path, filename), 'w') as f:
#                 f.write('detector_id,speed,occupancy,volume\n')
#                 for tup in buf:
#                     line_to_write = []
#                     line_to_write.append(tup[0])
#                     line_to_write.append(str(tup[1]))
#                     line_to_write.append(str(tup[2]))
#                     line_to_write.append(str(tup[3]))
#                     f.write(','.join(line_to_write)+"\n")
#             f.close() 
#             buf = []
#             cur_time = begin

#         if cur_time == begin: 
#             # buffer
#             buf.append( (id, speed, occupancy, nVehEntered) )
        
    # TODO: empty cache
    # filename = '{:05d}-{:05d}.csv'.format(cur_time, cur_time + (end - begin))
    # with open(os.path.join(save_path, filename), 'w') as f:
    #     f.write('detector_id,speed,occupancy,volume\n')
    #     for tup in buf:
    #         line_to_write = []
    #         line_to_write.append(tup[0])
    #         line_to_write.append(str(tup[1]))
    #         line_to_write.append(str(tup[2]))
    #         line_to_write.append(str(tup[3]))
    #         f.write(','.join(line_to_write)+"\n")
    # f.close() 
    # buf = []

def map_detector2link(map_path, save_path): 
    tree = ET.parse(map_path)
    root = tree.getroot()
    detector2link = {}
    for loop_detector in root.iter('inductionLoop'): 
        id   = loop_detector.get('id')[:-2]
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
    for vehicle in root.iter('vehicle'): 
        route_id    = vehicle.get('route')
        depart_time = float(vehicle.get('depart'))
        v[route_id] = depart_time
    # route
    for route in root.iter('route'): 
        route_id    = route.get('id')
        edges       = route.get('edges')
        depart_time = v[route_id]
        origin      = int(edges.split(' ')[0][:-2])
        destination = int(edges.split(' ')[-1][:-2])
        time_k      = int(depart_time / period)
        x.setdefault((origin, destination, time_k), 0)
        x[origin, destination, time_k] += 1
    save_as_pkl(data=x, pkl_path=save_path)
    return x

# TODO: aggregate_raw_data
def aggregate_raw_data(raw_data_path, save_path): 
    pass

def calc_tt_v2(route_path, nms_path, tt_save_path, verbose=True): 
    # travel time: unit s
    # routes
    od_routes = load_from_pkl(route_path)
    # network mean speed
    network_mean_speeds = load_from_pkl(nms_path)
    # complexity: |O| * |N| * |L| * |D*K|
    tt = {}
    tt_max = -1
    bar = tqdm(od_routes.items()) if verbose else od_routes.items()
    for od_pair, routes in bar:
        if verbose: 
            bar.set_description(f"OD Pair {od_pair}")
        for r, route in enumerate(routes):
            tt[od_pair, r] = []
            for len in route.get_lengths(): 
                # od ij - route r - link l
                # cols: date, rows: time interval k
                tmp = network_mean_speeds.map(lambda v: (len/v) * 3600 if v > 0 else (len/70) * 3600)
                tt[od_pair, r].append(tmp)
                cur_max = tmp.max().max()
                if cur_max > tt_max: 
                    tt_max = cur_max
                bar.set_description(f'max travel time: {tt_max}')
    save_as_pkl(tt, tt_save_path)
    return tt_max