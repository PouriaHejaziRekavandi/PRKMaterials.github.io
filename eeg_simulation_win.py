import os
import gc
import joblib
import numpy as np
import mne
import tensorflow as tf
from tensorflow.keras import backend as K
from scipy.spatial.distance import cdist
from sklearn.metrics import roc_auc_score
import matplotlib
matplotlib.use('Agg') # Headless plotting for Colab/Servers
import matplotlib.pyplot as plt
import seaborn as sns
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

# Try to import esinet
try:
    from esinet import Simulation, Net
    from esinet.forward import create_forward_model, get_info
except ImportError:
    logging.error("❌ esinet not found. Please run: pip install esinet mne tensorflow scipy scikit-learn matplotlib seaborn pyvistaqt")

def setup_environment(save_dir='esinet_project_results'):
    """
    Prepares the save directory for a Windows environment.
    """
    save_dir = os.path.abspath(save_dir)
    os.makedirs(save_dir, exist_ok=True)
    logging.info(f"✅ Results and models will be saved to: {save_dir}")
    return save_dir

def get_neighbors(fwd):
    """Precompute neighbors for the source space to calculate Localization Error."""
    fwd_fixed = mne.convert_forward_solution(fwd, force_fixed=True, surf_ori=True, use_cps=True)
    adj = mne.spatial_src_adjacency(fwd_fixed['src']).tocsr()
    neighbors = []
    for i in range(adj.shape[0]):
        neighbors.append(adj.indices[adj.indptr[i]:adj.indptr[i+1]])
    return neighbors

def multisource_metrics(gt, pr, pos_mm, neighbors, threshold_mm=30.0):
    """
    Calculates Maximum Localization Error (MLE) and % Sources Found.
    """
    def get_local_maxima(vec):
        vec = vec.ravel()
        if np.max(vec) <= 0: return np.array([np.argmax(vec)])
        cand = [i for i in range(len(vec)) if len(neighbors[i]) > 0 and np.all(vec[i] > vec[neighbors[i]])]
        if not cand: return np.array([np.argmax(vec)])
        cand = sorted(cand, key=lambda i: vec[i], reverse=True)
        selected = []
        for i in cand:
            if not selected: selected.append(i)
            else:
                d = np.linalg.norm(pos_mm[selected] - pos_mm[i], axis=1)
                if np.all(d > threshold_mm): selected.append(i)
        return np.array(selected)

    gt_max = get_local_maxima(gt)
    pr_max = get_local_maxima(pr)
    distances = cdist(pos_mm[gt_max], pos_mm[pr_max])
    min_distances = distances.min(axis=1)
    return np.mean(min_distances), np.mean(min_distances <= threshold_mm) * 100

def train_incremental(fwd, info, settings, save_dir, total_samples=10000, chunk_size=1000):
    """
    Trains the model in chunks with progress saving and resuming capability.
    """
    net_path = os.path.join(save_dir, 'trained_net.keras')

    # Check for GPU
    if tf.config.list_physical_devices('GPU'):
        logging.info("✅ GPU detected and will be used for training.")
    else:
        logging.info("ℹ️ No GPU detected. Training will proceed on CPU.")

    net = Net(fwd, model_type='convdip')

    # Load existing weights if available
    if os.path.exists(net_path):
        logging.info(f"🔄 Resuming from existing model: {net_path}")
        try:
            net.model = tf.keras.models.load_model(net_path)
        except Exception as e:
            logging.warning(f"Could not load existing model, starting fresh: {e}")

    num_chunks = total_samples // chunk_size
    history_accumulator = {'loss': [], 'val_loss': []}

    for i in range(num_chunks):
        logging.info(f"\n--- Chunk {i+1}/{num_chunks} ---")

        sim_chunk_path = os.path.join(save_dir, f'sim_chunk_{i+1}.pkl')

        # Load or Simulate
        if os.path.exists(sim_chunk_path):
            logging.info(f"✅ Loading chunk {i+1} from disk...")
            sim = joblib.load(sim_chunk_path)
        else:
            logging.info(f"⏳ Simulating chunk {i+1} (this might take a while)...")
            sim = Simulation(fwd, info, settings=settings)
            sim.simulate(n_samples=chunk_size)
            joblib.dump(sim, sim_chunk_path)
            logging.info(f"💾 Chunk {i+1} saved to disk.")

        # Fit
        fit_result = net.fit(sim, epochs=10, validation_split=0.1)

        # Safely extract loss history
        if hasattr(fit_result, 'history'):
            history_accumulator['loss'].extend(fit_result.history.get('loss', []))
            history_accumulator['val_loss'].extend(fit_result.history.get('val_loss', []))
        elif hasattr(net, 'model') and hasattr(net.model, 'history') and hasattr(net.model.history, 'history'):
            history_accumulator['loss'].extend(net.model.history.history.get('loss', []))
            history_accumulator['val_loss'].extend(net.model.history.history.get('val_loss', []))

        # Save Model
        net.model.save(net_path)

        # Cleanup
        del sim
        gc.collect()

    return net, history_accumulator

def main():
    save_dir = setup_environment()

    # 1. Forward Model (ico4 for higher accuracy)
    logging.info("Setting up forward models (ico4)...")
    info = get_info(sfreq=100)
    fwd_train = create_forward_model(info=info, sampling='ico4')

    # Inverse Crime Prevention: Perturb electrode positions for testing
    info_test = info.copy()
    for i in range(len(info_test['chs'])):
        info_test['chs'][i]['loc'][:3] += np.random.normal(0, 0.002, 3)
    fwd_test = create_forward_model(info=info_test, sampling='ico4')

    # 2. Simulation Settings
    custom_settings = {
        'method': 'standard',
        'number_of_sources': (1, 6),
        'extents': (21, 58),
        'amplitudes': (5, 10),
        'shapes': 'gaussian',
        'duration_of_trial': 0.5,
        'sample_frequency': 100,
        'target_snr': (4.5, 4.5),
        'beta_noise': (0, 0),
        'source_spread': 'region_growing',
        'source_number_weighting': False,
        'source_time_course': 'pulse',
    }

    # 3. Incremental Training (Updated to 10000 samples)
    net, history = train_incremental(fwd_train, info, custom_settings, save_dir, total_samples=10000, chunk_size=1000)

    # 4. Evaluation
    logging.info("\n📊 Final Evaluation on Perturbed Test Set...")
    sim_test = Simulation(fwd_test, info_test, settings=custom_settings)
    sim_test.simulate(n_samples=1000)

    y_true = sim_test.source_data
    y_pred = net.predict(sim_test)

    fwd_fixed = mne.convert_forward_solution(fwd_train, force_fixed=True, surf_ori=True, use_cps=True)
    pos_mm = np.vstack([s['rr'][s['vertno']] for s in fwd_fixed['src']]) * 1000
    neighbors = get_neighbors(fwd_train)

    mle_list, found_list, auc_list, mse_list = [], [], [], []

    for i in range(len(y_true)):
        jt = np.asarray(y_true[i].data)[:, 0] if hasattr(y_true[i], "data") else np.asarray(y_true[i])[:, 0]
        jp = np.asarray(y_pred[i].data)[:, 0] if hasattr(y_pred[i], "data") else np.asarray(y_pred[i])[:, 0]

        mle, found = multisource_metrics(np.abs(jt), np.abs(jp), pos_mm, neighbors)
        mle_list.append(mle)
        found_list.append(found)
        mse_list.append(np.mean((jt - jp)**2))

        jt_binary = (np.abs(jt) > 0).astype(int)
        if len(np.unique(jt_binary)) > 1:
            auc_list.append(roc_auc_score(jt_binary, np.abs(jp)))

    # 5. Output Report
    report = f"""
==================================================
✨ FINAL PERFORMANCE REPORT ✨
==================================================
Spatial AUC:     {np.mean(auc_list)*100:.2f}%
MSE:             {np.mean(mse_list):.3e}
Avg Error (mm):  {np.mean(mle_list):.2f} mm
Sources Found:   {np.mean(found_list):.2f}%
==================================================
"""
    print(report)
    with open(os.path.join(save_dir, 'performance_report.txt'), 'w') as f:
        f.write(report)

    # 6. Plotting
    plt.figure(figsize=(15, 6))
    plt.subplot(1, 2, 1)
    if history['loss']:
        plt.plot(history['loss'], label='Train')
        plt.plot(history['val_loss'], label='Val')
        plt.title('Learning Curve')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
    else:
        plt.title('Learning Curve (History Not Available)')

    plt.subplot(1, 2, 2)
    sns.histplot(mle_list, kde=True, color='purple')
    plt.title('Localization Error (MLE) Distribution')
    plt.xlabel('Error [mm]')

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'training_report.png'))
    logging.info(f"🎉 All done! Results saved in {save_dir}")

if __name__ == "__main__":
    main()
