import os
import gc
import subprocess
import sys
import logging
import random
import shutil
import joblib
import numpy as np
import tensorflow as tf
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

# Install dependencies if not already installed
try:
    import mne
except ImportError:
    print("Installing mne...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "mne"])
    import mne

try:
    import esinet
    from esinet import Simulation, Net
    from esinet.forward import create_forward_model, get_info
except ImportError:
    print("Installing esinet...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "esinet"])
    import esinet
    from esinet import Simulation, Net
    from esinet.forward import create_forward_model, get_info

from scipy.spatial.distance import cdist
from sklearn.metrics import roc_auc_score

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

def to_array(x):
    if hasattr(x, "data"): return np.asarray(x.data)
    x = np.asarray(x)
    return x if x.ndim == 2 else x[:, None]

def multisource_metrics(gt, pr, pos_mm, neighbors):
    def get_local_maxima(vec):
        vec = vec.ravel()
        gmax = np.max(vec)
        if gmax <= 0: return np.array([np.argmax(vec)])
        cand = [i for i in range(len(vec)) if len(neighbors[i]) > 0 and np.all(vec[i] > vec[neighbors[i]])]
        if not cand: return np.array([np.argmax(vec)])
        cand = sorted(cand, key=lambda i: vec[i], reverse=True)
        selected = []
        for i in cand:
            if not selected: selected.append(i)
            else:
                d = np.linalg.norm(pos_mm[selected] - pos_mm[i], axis=1)
                if np.all(d > 30.0): selected.append(i)
        return np.array(selected)

    gt_max = get_local_maxima(gt)
    pr_max = get_local_maxima(pr)
    D = cdist(pos_mm[gt_max], pos_mm[pr_max])
    min_d = D.min(axis=1)
    return np.mean(min_d), np.mean(min_d <= 30.0) * 100

def median_mad(data):
    arr = np.array(data)
    arr = arr[np.isfinite(arr)]
    if len(arr) == 0: return np.nan, np.nan
    med = np.median(arr)
    mad = np.median(np.abs(arr - med))
    return med, mad

def initialize_pipeline():
    logging.info("Initializing Forward Models...")
    info = get_info(sfreq=100)
    fwd = create_forward_model(info=info, sampling='ico4')

    # Setup for testing (avoiding Inverse Crime)
    info_test = info.copy()
    for i in range(len(info_test['chs'])):
        info_test['chs'][i]['loc'][:3] += np.random.normal(0, 0.002, 3)
    fwd_test = create_forward_model(info=info_test, sampling='ico4')

    sim_settings = {
        'method': 'standard', 'number_of_sources': (1, 6), 'extents': (21, 58),
        'amplitudes': (5, 10), 'shapes': 'gaussian', 'duration_of_trial': 1.0,
        'target_snr': (4.5, 4.5), 'beta_noise': (0, 0), 'source_spread': 'region_growing',
    }

    fwd_gm = mne.convert_forward_solution(fwd, force_fixed=True, surf_ori=True, use_cps=True)
    pos_gm = np.vstack([s['rr'][s['vertno']] for s in fwd_gm['src']]) * 1000
    adj = mne.spatial_src_adjacency(fwd_gm['src']).tocsr()
    neighbors = [adj.indices[adj.indptr[i]:adj.indptr[i+1]] for i in range(adj.shape[0])]

    return info, fwd, info_test, fwd_test, sim_settings, pos_gm, neighbors

class GeneticOptimizer:
    def __init__(self, fwd, info, pos_gm, neighbors, sim_settings, population_size=6, generations=3):
        self.fwd = fwd
        self.info = info
        self.pos_gm = pos_gm
        self.neighbors = neighbors
        self.sim_settings = sim_settings
        self.pop_size = population_size
        self.generations = generations

    def fitness(self, epochs, train_sim, val_sim):
        net = Net(self.fwd, model_type='convdip')

        try:
            # We use a fixed learning rate for GA to focus on epochs as requested
            # but we could also optimize it if needed.
            net.fit(train_sim, epochs=epochs, validation_split=0.0)
            y_true = val_sim.source_data
            y_pred = net.predict(val_sim)

            mle_list = []
            for i in range(len(y_true)):
                jt = to_array(y_true[i])[:, 0]
                jp = to_array(y_pred[i])[:, 0]
                mle, _ = multisource_metrics(np.abs(jt), np.abs(jp), self.pos_gm, self.neighbors)
                mle_list.append(mle)

            avg_mle = np.mean(mle_list)
            # Fitness is inverse of MLE (minimize MLE)
            fitness_val = 1.0 / (avg_mle + 1e-6)
            return fitness_val, avg_mle
        except Exception as e:
            logging.error(f"Error during fitness evaluation: {e}")
            return 0, 1000
        finally:
            del net
            tf.keras.backend.clear_session()
            gc.collect()

    def optimize(self, n_samples):
        logging.info(f"Starting GA optimization for sample size: {n_samples}")

        # Simulate data once for all individuals in this generation
        train_sim = Simulation(self.fwd, self.info, settings=self.sim_settings)
        train_sim.simulate(n_samples=n_samples)

        val_sim = Simulation(self.fwd, self.info, settings=self.sim_settings)
        # Use a reasonable size for validation during GA to save time
        val_samples = min(200, max(50, n_samples // 10))
        val_sim.simulate(n_samples=val_samples)

        # Initialize population: [epochs]
        population = [[random.randint(1, 500)] for _ in range(self.pop_size)]

        best_ind = None
        best_fitness = -1
        best_mle = 1000

        for gen in range(self.generations):
            logging.info(f"Generation {gen+1}/{self.generations}")
            fitness_scores = []
            for ind in population:
                fit, mle = self.fitness(int(ind[0]), train_sim, val_sim)
                fitness_scores.append(fit)
                if fit > best_fitness:
                    best_fitness = fit
                    best_ind = ind
                    best_mle = mle
                logging.info(f"Individual: Epochs={int(ind[0])} -> MLE: {mle:.2f}")

            # Selection (Tournament)
            new_population = []
            for _ in range(self.pop_size // 2):
                # Winner 1
                candidates = random.sample(list(zip(population, fitness_scores)), 2)
                parent1 = max(candidates, key=lambda x: x[1])[0]
                # Winner 2
                candidates = random.sample(list(zip(population, fitness_scores)), 2)
                parent2 = max(candidates, key=lambda x: x[1])[0]

                # Crossover
                child1 = [(parent1[0] + parent2[0]) // 2]
                child2 = [random.randint(min(parent1[0], parent2[0]), max(parent1[0], parent2[0]))]

                # Mutation
                for child in [child1, child2]:
                    if random.random() < 0.3: # 30% mutation rate
                        child[0] = np.clip(child[0] + random.randint(-50, 50), 1, 500)
                    new_population.append(child)
            population = new_population

        logging.info(f"Best epochs found for {n_samples} samples: {int(best_ind[0])} (MLE: {best_mle:.2f})")
        return int(best_ind[0]), train_sim

def run_simulation():
    info, fwd, info_test, fwd_test, sim_settings, pos_gm, neighbors = initialize_pipeline()

    sample_sizes = [1000, 2000, 5000, 10000, 50000, 100000]

    # Updated results directory for SharePoint/OneDrive synchronization
    # The OneDrive root is at C:\Users\PouriaRK\OneDrive - University of Toledo
    results_dir = r'C:\Users\PouriaRK\OneDrive - University of Toledo\EEG'

    # Fallback check: sometimes synced folders are inside a 'Documents' subfolder
    if not os.path.exists(os.path.dirname(results_dir)):
        alt_path = r'C:\Users\PouriaRK\OneDrive - University of Toledo\Documents\EEG'
        if os.path.exists(os.path.dirname(alt_path)):
            results_dir = alt_path

    # Final fallback to local directory if neither OneDrive path is accessible
    if not os.path.exists(os.path.dirname(results_dir)):
        logging.warning(f"OneDrive path not found. Falling back to local results directory.")
        results_dir = 'esinet_project_results'

    os.makedirs(results_dir, exist_ok=True)
    logging.info(f"Results will be saved to: {results_dir}")

    summary_file = os.path.join(results_dir, 'performance_report.txt')
    with open(summary_file, 'w') as f:
        f.write("EEG Source Localization - GA Optimized Epochs\n")
        f.write("============================================\n")

    for n in sample_sizes:
        logging.info(f"\n{'='*30}\nPROCESSING SAMPLE SIZE: {n}\n{'='*30}")

        # Adjust population and generations based on sample size to keep it feasible
        if n <= 5000:
            pop_size, gens = 6, 3
        elif n <= 10000:
            pop_size, gens = 4, 2
        else:
            pop_size, gens = 2, 2 # Minimal GA for very large sets

        optimizer = GeneticOptimizer(fwd, info, pos_gm, neighbors, sim_settings,
                                     population_size=pop_size, generations=gens)
        best_epochs, train_sim = optimizer.optimize(n)

        # Save simulation data as requested
        joblib.dump(train_sim, os.path.join(results_dir, f'sim_chunk_{n}.pkl'))

        # Final training with best parameters
        logging.info(f"Training final model for size {n} with {best_epochs} epochs...")
        net = Net(fwd, model_type='convdip')
        history = net.fit(train_sim, epochs=best_epochs, validation_split=0.1)

        # Plot training curve for this run
        plt.figure(figsize=(10, 5))
        plt.plot(history.history['loss'], label='Train Loss')
        plt.plot(history.history['val_loss'], label='Val Loss')
        plt.title(f'Training History - {n} samples ({best_epochs} epochs)')
        plt.legend()
        plt.savefig(os.path.join(results_dir, f'training_report_{n}.png'))
        plt.close()

        # Evaluation on 1000 test samples
        sim_test = Simulation(fwd_test, info_test, settings=sim_settings)
        sim_test.simulate(n_samples=1000)
        y_true = sim_test.source_data
        y_pred = net.predict(sim_test)

        mle_l, auc_l, found_l = [], [], []
        for i in range(len(y_true)):
            jt = to_array(y_true[i])[:, 0]
            jp = to_array(y_pred[i])[:, 0]
            mle, found = multisource_metrics(np.abs(jt), np.abs(jp), pos_gm, neighbors)
            mle_l.append(mle)
            found_l.append(found)
            jt_binary = (np.abs(jt) > 0).astype(int)
            if len(np.unique(jt_binary)) > 1:
                auc_l.append(roc_auc_score(jt_binary, np.abs(jp)) * 100)

        mle_m, mle_mad = median_mad(mle_l)
        auc_m, auc_mad = median_mad(auc_l)
        found_avg = np.mean(found_l)

        with open(summary_file, 'a') as f:
            f.write(f"\nSample Size: {n}\n")
            f.write(f"Best Epochs: {best_epochs}\n")
            f.write(f"Median MLE: {mle_m:.2f} mm (MAD: {mle_mad:.2f})\n")
            f.write(f"Median AUC: {auc_m:.2f} % (MAD: {auc_mad:.2f})\n")
            f.write(f"Sources Found: {found_avg:.2f} %\n")
            f.write("-" * 20 + "\n")

        # Save final model
        net.model.save(os.path.join(results_dir, f'trained_net_{n}.keras'))

        # Cleanup
        del net, train_sim, sim_test
        gc.collect()
        tf.keras.backend.clear_session()

if __name__ == "__main__":
    run_simulation()
