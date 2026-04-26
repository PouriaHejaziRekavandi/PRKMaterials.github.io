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
    import esinet
    from esinet import Simulation, Net
    from esinet.forward import create_forward_model, get_info
except ImportError:
    print("Installing/Updating dependencies...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-U", "mne", "esinet", "scipy", "scikit-learn"])
    import mne
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
        # Ensure vec is not empty and contains at least one non-zero value
        if vec.size == 0:
            return np.array([], dtype=int)

        if np.all(vec <= 0):
            return np.array([np.argmax(vec)])

        gmax = np.max(vec)
        cand = [i for i in range(len(vec)) if i < len(neighbors) and len(neighbors[i]) > 0 and np.all(vec[i] > vec[neighbors[i]])]
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

    if gt_max.size == 0 or pr_max.size == 0:
        return 100.0, 0.0 # Return high error if no sources found

    try:
        D = cdist(pos_mm[gt_max], pos_mm[pr_max])
        if D.size == 0: return 100.0, 0.0
        min_d = D.min(axis=1)
        return np.mean(min_d), np.mean(min_d <= 30.0) * 100
    except Exception as e:
        logging.error(f"Distance calculation error: {e}")
        return 100.0, 0.0

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
            net.fit(train_sim, epochs=epochs, validation_split=0.0)
            y_true = val_sim.source_data
            y_pred = net.predict(val_sim)

            mle_list = []
            for i in range(len(y_true)):
                jt = to_array(y_true[i])[:, 0]
                jp = to_array(y_pred[i])[:, 0]

                if jt.size == 0 or jp.size == 0 or np.all(np.isnan(jp)):
                    mle_list.append(100.0)
                    continue

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
        # Optimization: Use a smaller subset for GA to find the epoch trend faster
        ga_samples = min(n_samples, 5000)
        logging.info(f"Starting GA optimization using {ga_samples} samples")

        train_sim = Simulation(self.fwd, self.info, settings=self.sim_settings)
        train_sim.simulate(n_samples=ga_samples)

        val_sim = Simulation(self.fwd, self.info, settings=self.sim_settings)
        val_samples = min(200, max(50, ga_samples // 10))
        val_sim.simulate(n_samples=val_samples)

        population = [[random.randint(1, 500)] for _ in range(self.pop_size)]
        best_ind = None
        best_fitness = -1
        best_mle = 1000

        for gen in range(self.generations):
            logging.info(f"GA Generation {gen+1}/{self.generations}")
            fitness_scores = []
            for ind in population:
                fit, mle = self.fitness(int(ind[0]), train_sim, val_sim)
                fitness_scores.append(fit)
                if fit > best_fitness:
                    best_fitness, best_ind, best_mle = fit, ind, mle
                logging.info(f"  Epochs={int(ind[0])} -> MLE: {mle:.2f}")

            # Tournament Selection
            new_population = []
            for _ in range(self.pop_size // 2):
                p1 = max(random.sample(list(zip(population, fitness_scores)), 2), key=lambda x: x[1])[0]
                p2 = max(random.sample(list(zip(population, fitness_scores)), 2), key=lambda x: x[1])[0]
                # Crossover & Mutation
                c1 = [(p1[0] + p2[0]) // 2]
                if random.random() < 0.3: c1[0] = np.clip(c1[0] + random.randint(-50, 50), 1, 500)
                new_population.append(c1)
                c2 = [random.randint(min(p1[0], p2[0]), max(p1[0], p2[0]))]
                if random.random() < 0.3: c2[0] = np.clip(c2[0] + random.randint(-50, 50), 1, 500)
                new_population.append(c2)
            population = new_population

        logging.info(f"GA Best: {int(best_ind[0])} epochs (MLE: {best_mle:.2f})")
        return int(best_ind[0])

def run_simulation():
    info, fwd, info_test, fwd_test, sim_settings, pos_gm, neighbors = initialize_pipeline()
    sample_sizes = [1000, 2000, 5000, 10000, 50000, 100000]
    results_dir = '/content/onedrive/Documents/EEG'
    if not os.path.exists('/content/onedrive'): results_dir = 'esinet_project_results'
    os.makedirs(results_dir, exist_ok=True)

    summary_file = os.path.join(results_dir, 'performance_report.txt')
    with open(summary_file, 'w') as f: f.write("EEG Source Localization - Robust GA\n" + "="*30 + "\n")

    for n in sample_sizes:
        logging.info(f"\nPROCESSING SAMPLE SIZE: {n}")
        pop_size, gens = (6, 3) if n <= 5000 else ((4, 2) if n <= 10000 else (2, 2))
        optimizer = GeneticOptimizer(fwd, info, pos_gm, neighbors, sim_settings, population_size=pop_size, generations=gens)
        best_epochs = optimizer.optimize(n)

        # Large-scale Simulation in Chunks to save memory
        logging.info(f"Simulating {n} samples...")
        train_sim = Simulation(fwd, info, settings=sim_settings)
        if n > 20000:
            # Chunked simulation for very large sets
            all_source_data = []
            all_eeg_data = []
            for _ in range(n // 10000):
                chunk = Simulation(fwd, info, settings=sim_settings)
                chunk.simulate(n_samples=10000)
                all_source_data.append(chunk.source_data)
                all_eeg_data.append(chunk.eeg_data)
            train_sim.source_data = np.concatenate(all_source_data, axis=0)
            train_sim.eeg_data = np.concatenate(all_eeg_data, axis=0)
        else:
            train_sim.simulate(n_samples=n)

        joblib.dump(train_sim, os.path.join(results_dir, f'sim_chunk_{n}.pkl'))

        logging.info(f"Final Training ({best_epochs} epochs)...")
        net = Net(fwd, model_type='convdip')
        history = net.fit(train_sim, epochs=best_epochs, validation_split=0.1)

        # Evaluation
        sim_test = Simulation(fwd_test, info_test, settings=sim_settings)
        sim_test.simulate(n_samples=1000)
        y_true = sim_test.source_data
        y_pred = net.predict(sim_test)

        mle_l, auc_l, found_l = [], [], []
        for i in range(len(y_true)):
            jt, jp = to_array(y_true[i])[:, 0], to_array(y_pred[i])[:, 0]
            if jt.size == 0 or jp.size == 0: continue
            mle, found = multisource_metrics(np.abs(jt), np.abs(jp), pos_gm, neighbors)
            mle_l.append(mle); found_l.append(found)
            jt_b = (np.abs(jt) > 0).astype(int)
            if len(np.unique(jt_b)) > 1: auc_l.append(roc_auc_score(jt_b, np.abs(jp)) * 100)

        mle_m, mle_mad = median_mad(mle_l)
        auc_m, auc_mad = median_mad(auc_l)
        with open(summary_file, 'a') as f:
            f.write(f"\nSize: {n} | Epochs: {best_epochs}\n")
            f.write(f"MLE: {mle_m:.2f} (±{mle_mad:.2f}) | AUC: {auc_m:.2f}% | Found: {np.mean(found_l):.2f}%\n")

        net.model.save(os.path.join(results_dir, f'trained_net_{n}.keras'))
        plt.figure(); plt.plot(history.history['loss'], label='Loss'); plt.savefig(os.path.join(results_dir, f'training_{n}.png')); plt.close()
        del net, train_sim, sim_test; gc.collect(); tf.keras.backend.clear_session()

if __name__ == "__main__":
    run_simulation()
