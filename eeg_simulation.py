import os
import gc
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
from tqdm.auto import tqdm

import mne
import esinet
from esinet import Simulation, Net
from esinet.forward import create_forward_model, get_info
from scipy.spatial.distance import cdist
from sklearn.metrics import roc_auc_score

# ============================================================
# MANUAL CONFIGURATION
# ============================================================
SAMPLE_SIZES = [500, 1000]
EPOCHS = 100
LEARNING_RATE = 0.0001
BATCH_SIZE = 32

SIM_SETTINGS = {
    'method': 'standard',
    'number_of_sources': (1, 6),
    'extents': (21, 58),
    'amplitudes': (5, 10),
    'shapes': 'gaussian',
    'duration_of_trial': 1.0,
    'target_snr': (4.5, 4.5),
    'beta_noise': (0, 0),
    'source_spread': 'region_growing',
}

RESULTS_DIR = '/content/onedrive/Documents/EEG'
if not os.path.exists('/content/onedrive'):
    RESULTS_DIR = 'esinet_project_results'
# ============================================================

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

def to_array(x):
    if hasattr(x, "data"): return np.asarray(x.data)
    x = np.asarray(x)
    return x if x.ndim == 2 else x[:, None]

def multisource_metrics(gt, pr, pos_mm, adj):
    def get_local_maxima(vec):
        vec = vec.ravel()
        if vec.size == 0:
            return np.array([], dtype=int)

        if np.all(vec <= 0):
            return np.array([np.argmax(vec)])

        # Vectorized local maxima detection
        neighbor_indices = adj.indices
        row_ptr = adj.indptr
        node_indices = np.repeat(np.arange(len(vec)), np.diff(row_ptr))

        # A node is NOT a local maximum if any neighbor is > to it
        not_max_mask = vec[neighbor_indices] > vec[node_indices]
        is_not_max = np.zeros(len(vec), dtype=bool)
        is_not_max[node_indices[not_max_mask]] = True

        has_neighbors = np.diff(row_ptr) > 0
        cand_mask = (~is_not_max) & (vec > 0) & has_neighbors
        cand = np.where(cand_mask)[0]

        if cand.size == 0:
            return np.array([np.argmax(vec)])

        # Sort candidates by value descending
        cand = cand[np.argsort(vec[cand])[::-1]]

        selected = []
        for i in cand:
            if not selected:
                selected.append(i)
            else:
                # Spatial pruning: keep only maxima at least 30mm apart
                d = np.linalg.norm(pos_mm[selected] - pos_mm[i], axis=1)
                if np.all(d > 30.0):
                    selected.append(i)
        return np.array(selected)

    gt_max = get_local_maxima(gt)
    pr_max = get_local_maxima(pr)

    if gt_max.size == 0 or pr_max.size == 0:
        return 100.0, 0.0

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

    fwd_gm = mne.convert_forward_solution(fwd, force_fixed=True, surf_ori=True, use_cps=True)
    pos_gm = np.vstack([s['rr'][s['vertno']] for s in fwd_gm['src']]) * 1000
    adj = mne.spatial_src_adjacency(fwd_gm['src']).tocsr()

    return info, fwd, info_test, fwd_test, pos_gm, adj

def get_persistent_simulation(fwd, info, settings, n_target, results_dir):
    sim_path = os.path.join(results_dir, 'simulation_data.joblib')
    if os.path.exists(sim_path):
        logging.info(f"Loading cached simulation from {sim_path}")
        sim_data = joblib.load(sim_path)

        n_current = len(sim_data.source_data)
        if n_current < n_target:
            n_needed = n_target - n_current
            logging.info(f"Simulating additional {n_needed} samples...")
            sim_new = Simulation(fwd, info, settings=settings)
            sim_new.simulate(n_samples=n_needed)

            sim_data.source_data = np.concatenate([sim_data.source_data, sim_new.source_data], axis=0)
            sim_data.eeg_data = np.concatenate([sim_data.eeg_data, sim_new.eeg_data], axis=0)
            joblib.dump(sim_data, sim_path)
        return sim_data
    else:
        logging.info(f"Creating new simulation with {n_target} samples")
        sim_all = Simulation(fwd, info, settings=settings)
        sim_all.simulate(n_samples=n_target)
        joblib.dump(sim_all, sim_path)
        return sim_all

def run_simulation():
    info, fwd, info_test, fwd_test, pos_gm, adj = initialize_pipeline()

    os.makedirs(RESULTS_DIR, exist_ok=True)

    summary_file = os.path.join(RESULTS_DIR, 'performance_report.txt')
    if not os.path.exists(summary_file):
        with open(summary_file, 'w') as f: f.write("EEG Source Localization - Manual Configuration\n" + "="*40 + "\n")

    for n in tqdm(SAMPLE_SIZES, desc="Overall Progress"):
        logging.info(f"\n{'='*20}\nPROCESSING SAMPLE SIZE: {n}\n{'='*20}")

        # Check if already done
        model_path = os.path.join(RESULTS_DIR, f'trained_net_{n}.keras')
        if os.path.exists(model_path):
            logging.info(f"Sample size {n} already processed. Skipping.")
            continue

        # 1. Get Simulation Data
        full_sim = get_persistent_simulation(fwd, info, SIM_SETTINGS, n, RESULTS_DIR)

        # Create a light copy for the current size
        current_sim = Simulation(fwd, info, settings=SIM_SETTINGS)
        current_sim.source_data = full_sim.source_data[:n]
        current_sim.eeg_data = full_sim.eeg_data[:n]

        # 2. Training with manual parameters
        logging.info(f"Training (Ep={EPOCHS}, LR={LEARNING_RATE}, BS={BATCH_SIZE})...")
        net = Net(fwd, model_type='convdip')

        # We manually set parameters if fit() doesn't support them.
        try:
            # We use a custom callback to capture the history because net.fit() returns the Net object
            class HistoryCallback(tf.keras.callbacks.Callback):
                def on_train_begin(self, logs=None):
                    self.history = {'loss': []}
                def on_epoch_end(self, epoch, logs=None):
                    self.history['loss'].append(logs.get('loss'))

            history_cb = HistoryCallback()
            net.fit(current_sim, epochs=EPOCHS, validation_split=0.1, learning_rate=LEARNING_RATE, batch_size=BATCH_SIZE, callbacks=[history_cb])
            loss_data = history_cb.history['loss']
        except Exception as e:
            logging.warning(f"Advanced training failed, falling back to default: {e}")
            history = net.fit(current_sim, epochs=EPOCHS, validation_split=0.1)
            # Try to extract history from Net object if available
            if hasattr(net, 'history') and hasattr(net.history, 'history'):
                loss_data = net.history.history['loss']
            else:
                loss_data = []

        # 3. Evaluation on independent test set
        logging.info("Evaluating on test set...")
        sim_test = Simulation(fwd_test, info_test, settings=SIM_SETTINGS)
        sim_test.simulate(n_samples=500)
        y_true = sim_test.source_data
        y_pred = net.predict(sim_test)

        mle_l, auc_l, found_l = [], [], []
        for i in range(len(y_true)):
            jt, jp = to_array(y_true[i])[:, 0], to_array(y_pred[i])[:, 0]
            if jt.size == 0 or jp.size == 0: continue
            mle, found = multisource_metrics(np.abs(jt), np.abs(jp), pos_gm, adj)
            mle_l.append(mle); found_l.append(found)
            jt_b = (np.abs(jt) > 0).astype(int)
            if len(np.unique(jt_b)) > 1: auc_l.append(roc_auc_score(jt_b, np.abs(jp)) * 100)

        mle_m, mle_mad = median_mad(mle_l)
        auc_m, auc_mad = median_mad(auc_l)
        with open(summary_file, 'a') as f:
            f.write(f"\nSize: {n} | Ep: {EPOCHS} | LR: {LEARNING_RATE} | BS: {BATCH_SIZE}\n")
            f.write(f"MLE: {mle_m:.2f} (±{mle_mad:.2f}) | AUC: {auc_m:.2f}% | Found: {np.mean(found_l):.2f}%\n")

        # Save results
        net.model.save(model_path)
        if loss_data:
            plt.figure(); plt.plot(loss_data, label='Loss'); plt.title(f'Training Loss (n={n})'); plt.savefig(os.path.join(RESULTS_DIR, f'training_{n}.png')); plt.close()

        # Memory Cleanup
        del net, current_sim, sim_test; gc.collect(); tf.keras.backend.clear_session()

if __name__ == "__main__":
    run_simulation()
