import argparse
import numpy as np
import xml.etree.ElementTree as ET
from tqdm import tqdm
from simulation import SumoSimulationParallel, sumo_objective_function_nrmse
from utils import add_log, save_as_pkl
from visual import plot_pearson_correlation_scatter, plot_historical_objective_function_value
from datetime import datetime
import shutil
import os
from multiprocessing.pool import ThreadPool
from GeneticAlgorithm import GeneticAlgorithm

def clean_simulation_directory(dir_path):
    """Remove temporary simulation directory"""
    try:
        shutil.rmtree(dir_path)
    except Exception as e:
        print(f"Warning: Failed to remove directory {dir_path}: {str(e)}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default=r'simulation1\sim1.sumocfg')
    parser.add_argument('--data', type=str, default=r'simulation1\simulation1')
    parser.add_argument('--flow',     type = str, default = r'simulation1\sim.rou.xml', help = 'demand xml data path')
    parser.add_argument('--duration', type=int, default=3600)
    parser.add_argument('--period', type=int, default=180)
    parser.add_argument('--seed', type=int, default=2025)
    parser.add_argument('--mute_warnings', action='store_true', default=True)
    parser.add_argument('--mute_step_logs', action='store_true', default=False)
    sim_args = parser.parse_args()
    
    interval_n = (sim_args.duration // sim_args.period)
    x_gt = np.random.randint(low=20, high=100, size=(15, 15, interval_n))
    
    # Initialize ground truth simulation
    ground_truth_sim = SumoSimulationParallel(sim_args)
    print("Generating Ground Truth Data...")
    q_gt, k_gt, v_gt = ground_truth_sim.run_sumo(x_gt)
    
    cur_time = datetime.now().strftime("%Y%m%d-%H%M%S")
    save_path = os.path.join('results', cur_time)
    os.makedirs(save_path, exist_ok=True)
    save_as_pkl(data=x_gt, pkl_path=os.path.join(save_path, 'x_gt.pkl'))
    save_as_pkl(data=q_gt, pkl_path=os.path.join(save_path, 'q_gt.pkl'))
    save_as_pkl(data=k_gt, pkl_path=os.path.join(save_path, 'k_gt.pkl'))
    save_as_pkl(data=v_gt, pkl_path=os.path.join(save_path, 'v_gt.pkl'))
    # Configure GA parameters
    ga_params = {
        'pop_size': 1<<6,  
        'max_gen': 100,
        'elite_rate': 0.1,
        'mutation_rate': 0.05,
        'crossover_rate': 0.8
    }
    
    # Run Genetic Algorithm
    ga = GeneticAlgorithm(q_gt, k_gt, v_gt, ga_params, sim_args)
    x_final, fitness_history = ga.run()
    
    # Generate comparison plots
    ga.benchmark(x_gt, x_final)
    
    # save final results
    save_as_pkl(data=x_final, pkl_path=os.path.join(save_path, 'x_final.pkl'))
    # Cleanup ground truth directory
    if os.path.exists(sim_args.data):
        clean_simulation_directory(sim_args.data)