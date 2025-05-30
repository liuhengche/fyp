# Genetic Algorithm with Parallel Simulation Support
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

PWD = 'simulation1'
ROADNETWORK_FILE = 'osm.net.xml'
SUMOCONFIG_FILE = 'sim1.sumocfg'
DETECTOR_FILE = 'sim1.add.xml'

def clean_simulation_directory(dir_path):
    """Remove temporary simulation directory"""
    try:
        shutil.rmtree(dir_path)
    except Exception as e:
        print(f"Warning: Failed to remove directory {dir_path}: {str(e)}")

class GeneticAlgorithm:
    def __init__(self, q_gt, k_gt, v_gt, ga_args, sim_args):
        # Ground truth data
        self.q_gt = q_gt
        self.k_gt = k_gt
        self.v_gt = v_gt
        
        # GA parameters 
        self.pop_size = ga_args['pop_size']
        self.max_gen = ga_args['max_gen']
        self.elite_rate = ga_args['elite_rate']
        self.mutation_rate = ga_args['mutation_rate']
        self.crossover_rate = ga_args['crossover_rate']
        
        # Simulation setup 
        self.base_sim_args = sim_args
        self.interval_n = (sim_args.duration // sim_args.period)
        self.x_shape = (15, 15, self.interval_n)
        
        # Threading setup
        self.thread_pool = ThreadPool(processes=min(ga_args['pop_size'], os.cpu_count()-1))
        
        # Initialize population 
        self.population = self._initialize_population()
        self.fitness_history = []

    def _initialize_population(self):
        """Generate initial population with random demand matrices"""
        return [np.random.randint(low=0, high=200, size=self.x_shape) 
                for _ in range(self.pop_size)]

    def _sumo_simulation_task(self, idx, individual, gen):
        """Parallel simulation task with directory management"""
        # Create temporary directory
        output_dir = os.path.join(PWD, f"gen_{gen}_{idx}")
        os.makedirs(output_dir, exist_ok=True)

        # Copy required files
        for file in [SUMOCONFIG_FILE, ROADNETWORK_FILE, DETECTOR_FILE]:
            src = os.path.join(PWD, file)
            dst = os.path.join(output_dir, file)
            if os.path.exists(src):
                shutil.copy(src, dst)

        # Parse and modify detector file
        add_path = os.path.join(output_dir, DETECTOR_FILE)
        tree = ET.parse(add_path)
        root = tree.getroot()
        for detector in root.iter('inductionLoop'):
            detector_id = detector.get('id')
            detector.set('file', os.path.join('simulation', f'{detector_id[:-2]}.xml'))
        tree.write(add_path)

        # Configure simulation arguments
        sim_args = argparse.Namespace()
        sim_args.config = os.path.join(output_dir, SUMOCONFIG_FILE)
        sim_args.data = os.path.join(output_dir, 'simulation')
        sim_args.flow = os.path.join(output_dir, 'sim.rou.xml')
        sim_args.duration = self.base_sim_args.duration
        sim_args.period = self.base_sim_args.period
        sim_args.seed = self.base_sim_args.seed
        sim_args.mute_step_logs = True
        sim_args.mute_warnings = True
        os.makedirs(sim_args.data, exist_ok=True)

        # Run simulation
        simulator = SumoSimulationParallel(sim_args)
        q, k, v = simulator.run_sumo(od_matrix=individual)
        
        # Calculate objective using NRMSE
        score = sumo_objective_function_nrmse([q, k, v], [self.q_gt, self.k_gt, self.v_gt])
        
        # Cleanup temporary directory
        clean_simulation_directory(output_dir)
        
        return score

    def _parallel_evaluation(self, gen):
        """Handle parallel evaluation using thread pool"""
        task_inputs = [(idx, ind, gen) for idx, ind in enumerate(self.population)]
        return self.thread_pool.starmap(self._sumo_simulation_task, task_inputs)

    def _select_parents(self, fitness_scores):
        """Tournament selection"""
        selected = []
        tournament_size = 3
        for _ in range(self.pop_size):
            candidates = np.random.choice(range(len(fitness_scores)), tournament_size)
            winner = candidates[np.argmin([fitness_scores[c] for c in candidates])]
            selected.append(self.population[winner])
        return selected

    def _crossover(self, parent1, parent2):
        """Uniform crossover"""
        mask = np.random.rand(*self.x_shape) < 0.5
        child = parent1 * mask + parent2 * (~mask)
        return np.clip(child, 20, 100)

    def _mutate(self, individual):
        """Gaussian mutation"""
        mutation_mask = np.random.rand(*self.x_shape) < self.mutation_rate
        mutation_values = np.random.normal(0, 10, size=self.x_shape)
        mutated = individual + mutation_values * mutation_mask
        return np.clip(mutated, 20, 100)

    def _create_new_generation(self, parents, gen):
        """Generate new population through crossover and mutation"""
        new_pop = []
        
        # Parallel elite evaluation
        elite_size = int(self.pop_size * self.elite_rate)
        elite_inputs = [(idx, p, gen) for idx, p in enumerate(parents)]
        elite_scores = self.thread_pool.starmap(self._sumo_simulation_task, elite_inputs)
        elites = sorted(zip(parents, elite_scores), key=lambda x: x[1])[:elite_size]
        new_pop.extend([e[0] for e in elites])
        
        # Generate offspring
        while len(new_pop) < self.pop_size:
            parent_indices = np.arange(len(parents))
            selected_indices = np.random.choice(parent_indices, 2, replace=False)
            parent1, parent2 = parents[selected_indices[0]], parents[selected_indices[1]]
            
            child = self._crossover(parent1, parent2) if np.random.rand() < self.crossover_rate else parent1.copy()
            new_pop.append(self._mutate(child))
            
        return new_pop

    def run(self):
        """Main GA loop with parallel evaluation"""
        add_log("Genetic Algorithm Testing {:%Y-%m-%d_%H}\n".format(datetime.now()), 'GA_testing.txt')
        try:
            for gen in range(self.max_gen):
                print(f"\nGeneration {gen} - Parallel Evaluation")
                
                # Parallel fitness evaluation
                fitness_scores = self._parallel_evaluation(gen)
                # assert len(fitness_scores) == len(self.population), "Incomplete evaluation results!"
                if (len(fitness_scores) != len(self.population)):
                    print("Incomplete evaluation results!")
                    continue

                # Record best fitness
                best_score = np.min(fitness_scores)
                self.fitness_history.append(best_score)
                print(f"Gen {gen}: Best Fitness = {best_score}")
                add_log(f"Gen {gen}: Best Fitness = {best_score}\n", 'GA_testing.txt')
                
                # Evolutionary operations
                print("Selecting parents...")
                parents = self._select_parents(fitness_scores)
                print("Creating new generation...")
                self.population = self._create_new_generation(parents, gen)
        
        finally:
            self.thread_pool.close()
            self.thread_pool.join()

        # Final evaluation
        best_individual = self.population[np.argmin(fitness_scores)]
        return best_individual, self.fitness_history

    def benchmark(self, x_gt, x_final):
        """Generate comparison plots"""
        # Create temporary directory for final evaluation
        output_dir = os.path.join(PWD, "final_eval")
        os.makedirs(output_dir, exist_ok=True)

        # Configure simulation

        sim_args = argparse.Namespace()
        sim_args.config = os.path.join(output_dir, SUMOCONFIG_FILE)
        sim_args.data = os.path.join(output_dir, 'simulation')
        sim_args.flow = os.path.join(output_dir, 'sim.rou.xml')
        sim_args.duration = self.base_sim_args.duration
        sim_args.period = self.base_sim_args.period
        sim_args.seed = self.base_sim_args.seed
        sim_args.mute_step_logs = True
        sim_args.mute_warnings = True
        os.makedirs(sim_args.data, exist_ok=True)

        # Run final simulation
        simulator = SumoSimulationParallel(sim_args)
        q_final, k_final, v_final = simulator.run_sumo(od_matrix=x_final)
        
        # Flatten arrays for correlation calculation
        q_gt_flat, k_gt_flat, v_gt_flat = self.q_gt.flatten(), self.k_gt.flatten(), self.v_gt.flatten()
        q_final_flat = q_final.flatten()
        k_final_flat = k_final.flatten()
        v_final_flat = v_final.flatten()
        x_gt_flat, x_final_flat = x_gt.flatten(), x_final.flatten()
        
        # Generate plots 
        plot_pearson_correlation_scatter(q_gt_flat, q_final_flat, 
                                        upper=max(np.max(self.q_gt), np.max(q_final)),
                                        var_name='q')
        plot_pearson_correlation_scatter(k_gt_flat, k_final_flat,
                                        upper=max(np.max(self.k_gt), np.max(k_final)),
                                        var_name='k')
        plot_pearson_correlation_scatter(v_gt_flat, v_final_flat,
                                        upper=max(np.max(self.v_gt), np.max(v_final)),
                                        var_name='v')
        plot_pearson_correlation_scatter(x_gt_flat, x_final_flat,
                                        upper=max(np.max(x_gt), np.max(x_final)),
                                        var_name='x')
        plot_historical_objective_function_value(
            x=range(len(self.fitness_history)),
            y_list=[self.fitness_history],
            label_list=['Fitness Evolution']
        )
        
        # Cleanup final directory
        clean_simulation_directory(output_dir)
