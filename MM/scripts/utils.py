import json
import yaml
import pickle
import datetime
import numpy as np
import pandas as pd
import xml.etree.ElementTree as ET

from scipy.stats import pearsonr
# Time
def iter_time(start, end):
    res = []
    start_y, start_m, start_d = int(start[:4]), int(start[4:6]), int(start[6:8])
    end_y, end_m, end_d = int(end[:4]), int(end[4:6]), int(end[6:8])
    begin_date = datetime.date(start_y, start_m, start_d)
    end_date = datetime.date(end_y, end_m, end_d)
    for i in range((end_date - begin_date).days + 1):
        day = begin_date + datetime.timedelta(days = i)
        date = f'{day.year:04d}{day.month:02d}{day.day:02d}'
        res.append(date)
    return res

def sum_df(df_list): 
    '''
    input: [1, ..., r] list of dataframes (|D||K|)
    output: [1, ..., r] list of softmax dataframes
    '''
    df_shape = df_list[-1].shape
    df_index = df_list[-1].index
    df_columns = df_list[-1].columns 
    sum_df = pd.DataFrame(np.zeros(df_shape), index=df_index, columns=df_columns)
    for i in df_list:
        sum_df += i
    return sum_df

def softmax_df(df_list): 
    '''
    input: [1, ..., r] list of dataframes (|D||K|)
    output: [1, ..., r] list of softmax dataframes
    '''
    sum = sum_df(df_list)
    res = [(i/sum) for i in df_list]
    return res

def preprocess_df(df): 
    # df = df.replace(0.0, np.nan)
    df.interpolate(method='linear', axis=0)
    df.interpolate(method='linear', axis=1)
    return df

def load_config(config_path): 
    ''' support json, yaml '''
    ext = config_path.split('.')[-1]
    with open(config_path) as f:
        if ext == 'yaml': 
            config = yaml.load(f, Loader=yaml.FullLoader)
        elif ext == 'json': 
            config = json.load(f)
    f.close()
    return config

def append_config(origin_config, append_config, config_save_path=None):
    # append
    for key, val in append_config.items(): 
        origin_config[key] = val
    # save local
    if config_save_path is not None:
        with open(config_save_path, 'w') as f: 
            yaml.dump(origin_config, f)
        f.close()
    return origin_config

def load_from_pkl(pkl_path):
    with open(pkl_path, 'rb') as f:
        data = pickle.load(f)
    f.close()
    return data

def save_as_pkl(data, pkl_path): 
    with open(pkl_path, 'wb') as f: 
        pickle.dump(data, f)
    f.close()

def add_log(msg, log_path): 
    with open(log_path, 'a') as f: 
        f.write(msg)
    f.close()

def results2csv(x_path, p_path, x_save_path, p_save_path, date): 
    results = load_from_pkl(x_path)
    vars    = results['Variable_values'] 
    # x
    with open(x_save_path, 'w') as f:
        f.write('variable,value\n')
        for key, val in vars.items():
            if key[0] == 'X': 
                line_to_write = []
                line_to_write.append(key.replace(',', ' '))
                line_to_write.append(str(val))
                f.write(','.join(line_to_write) + '\n')
    f.close()
    # p
    route_choice = load_from_pkl(p_path)
    with open(p_save_path, 'w') as f:
        f.write('(i j),r,probability\n')
        for key, val in route_choice.items():
            line_to_write = []
            line_to_write.append(str(key[0]).replace(',', ' '))
            line_to_write.append(str(key[1]))
            line_to_write.append(str(route_choice[key].loc[0, date]))
            f.write(','.join(line_to_write) + '\n')
    f.close()

def parse_rou_file(params, orig_rou, dest_rou): 
    # Parse the XML file
    tree = ET.parse(orig_rou)
    root = tree.getroot()

    # Find the vType element with id="trial"
    for vtype in root.findall('vType'):
        if vtype.get('id') == 'trial':
            # Update the attributes with the provided parameters
            for key, val in params.items():
                vtype.set(key, str(val))
            break

    tree.write(dest_rou, encoding='UTF-8', xml_declaration=True)

def parse_sumo_cfg_file(orig_sumo_cfg, dest_sumo_cfg, resources): 
    # Parse the XML file
    tree = ET.parse(orig_sumo_cfg)
    root = tree.getroot()
    input_elem = root.find('input')

    rou_file, net_file, add_file = resources

    if input_elem is not None: 
        input_elem.find('route-files').set('value', rou_file)
        input_elem.find('net-file').set('value', net_file)
        input_elem.find('additional-files').set('value', add_file)
    
    tree.write(dest_sumo_cfg, encoding='UTF-8', xml_declaration=True)

# generate x
def generate_ground_truth_matrix(shape, lower = 2, upper = 10):
    M = {}
    i_, j_, k_ = shape 
    for i in i_: 
        for j in j_: 
            for k in k_:
                M[i, j, k] = np.random.uniform(lower, upper)
    return M 


def generate_seed_matrix_uniform(matrix):
    low, medium, high = {}, {}, {}
    for key, val in matrix.items(): 
        low[key]    = val * (0.7 + 0.3*np.random.uniform(0, 1))
        medium[key] = val * (0.8 + 0.3*np.random.uniform(0, 1))
        high[key]   = val * (0.9 + 0.3*np.random.uniform(0, 1))
    return low, medium, high
    

def generate_seed_matrix_gaussian():
    pass

