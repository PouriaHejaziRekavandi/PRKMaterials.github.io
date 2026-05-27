import os
import gc
import shutil
import numpy as np
import mne
import tensorflow as tf
from tensorflow.keras import backend as K
from scipy.spatial import cKDTree
from scipy.spatial.distance import cdist
from sklearn.metrics import roc_auc_score
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import logging

from esinet import Simulation, Net
from esinet.forward import create_forward_model, get_info
from pyvirtualdisplay import Display

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

# ============================================================
# Global Configuration
# ============================================================
TOTAL_SAMPLES = 2000
CHUNK_SIZE = 1000
EPOCHS = 10
SIM_SETTINGS = {
    'method': 'standard', 'number_of_sources': (1, 6), 'extents': (21, 58),
    'amplitudes': (5, 10), 'shapes': 'gaussian', 'duration_of_trial': 1.0,
    'target_snr': (4.5, 4.5), 'beta_noise': (0, 0), 'source_spread': 'region_growing',
}

def setup_environment():
    # Initialize virtual display
    display = Display(visible=0, size=(1366, 768))
    display.start()
    os.environ["PYVIRTUALDISPLAY_DISPLAYFD"] = "0"

    # Try to mount Google Drive to save checkpoints persistently
    checkpoints_dir = 'checkpoints'
    try:
        from google.colab import drive
        drive.mount('/content/drive')
        checkpoints_dir = '/content/drive/MyDrive/eeg_checkpoints'
        logging.info("Google Drive mounted successfully.")
    except Exception as e:
        logging.info(f"Could not mount Drive, falling back to local storage: {e}")
        checkpoints_dir = os.path.abspath(checkpoints_dir)

    return checkpoints_dir

# ============================================================
# Optimized Batch Training Functions
# ============================================================
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
    arr = arr[np.isfinite(arr)] # حذف مقادیر نامعتبر
    if len(arr) == 0:
        return np.nan, np.nan
    med = np.median(arr)
    mad = np.median(np.abs(arr - med))
    return med, mad

def initialize_pipeline():
    logging.info("Initializing Forward Models...")
    info = get_info(sfreq=100)
    fwd = create_forward_model(info=info, sampling='ico4')

    # Setup AGM for testing (avoiding Inverse Crime)
    info_test = info.copy()
    for i in range(len(info_test['chs'])):
        info_test['chs'][i]['loc'][:3] += np.random.normal(0, 0.002, 3)

    # FIXED: Changed 'oct3' to 'ico4' so the source space dimensions match the training data for MSE calculation
    fwd_test = create_forward_model(info=info_test, sampling='ico4')

    # Precompute Leadfields and Neighbors
    fwd_gm = mne.convert_forward_solution(fwd, force_fixed=True, surf_ori=True, use_cps=True)
    pos_gm = np.vstack([s['rr'][s['vertno']] for s in fwd_gm['src']]) * 1000
    adj = mne.spatial_src_adjacency(fwd_gm['src']).tocsr()
    neighbors = np.split(adj.indices, adj.indptr[1:-1])

    # Create the Network once
    net = Net(fwd, model_type='convdip')

    return info, fwd, info_test, fwd_test, pos_gm, neighbors, net


def train_model(net, fwd, info, sim_settings, checkpoints_dir, total_samples=TOTAL_SAMPLES, chunk_size=CHUNK_SIZE):
    num_chunks = total_samples // chunk_size

    os.makedirs(checkpoints_dir, exist_ok=True)

    logging.info(f"Starting Incremental Training for {total_samples} samples in {num_chunks} chunks.")

    all_loss = []
    all_val_loss = []

    for chunk_idx in range(num_chunks):
        logging.info(f"\n--- Processing Chunk {chunk_idx + 1}/{num_chunks} ---")

        # 1. Simulate a batch
        sim_train = Simulation(fwd, info, settings=sim_settings)
        sim_train.simulate(n_samples=chunk_size)

        # 2. Fit the model (weights are preserved across calls)
        # Monkey-patch net.model.fit to inject a custom callback to capture metrics
        original_fit = net.model.fit
        def patched_fit(*args, **kwargs):
            metrics_callback = tf.keras.callbacks.LambdaCallback(
                on_epoch_end=lambda epoch, logs: (
                    all_loss.append(logs.get('loss')),
                    all_val_loss.append(logs.get('val_loss'))
                )
            )
            callbacks = list(kwargs.get('callbacks') or [])
            callbacks.append(metrics_callback)
            kwargs['callbacks'] = callbacks
            return original_fit(*args, **kwargs)

        net.model.fit = patched_fit
        try:
            net.fit(sim_train, epochs=EPOCHS, validation_split=0.1)
        finally:
            net.model.fit = original_fit

        # 3. Save progress locally to Drive
        chunk_dir = os.path.join(checkpoints_dir, f'convdip_checkpoint_chunk_{chunk_idx+1}')
        if os.path.exists(chunk_dir):
            shutil.rmtree(chunk_dir)
        os.makedirs(chunk_dir, exist_ok=True)
        net.model.save(f'{chunk_dir}.keras')

        # 4. Memory Cleanup
        del sim_train
        # FIXED: Removed K.clear_session() here. Clearing the session destroys the `net` graph.
        gc.collect()
        logging.info(f"Chunk {chunk_idx + 1} completed and memory cleared.")

    return all_loss, all_val_loss


def evaluate_model(net, fwd_test, info_test, sim_settings, pos_gm, neighbors):
    logging.info("\nPerforming Final Evaluation on Fixed Test Set...")
    sim_test = Simulation(fwd_test, info_test, settings=sim_settings)
    sim_test.simulate(n_samples=1000)
    y_true = sim_test.source_data
    y_pred = net.predict(sim_test)

    mle_l, found_l, auc_l, mse_l, nmse_l = [], [], [], [], []

    for i in range(len(y_true)):
        jt = to_array(y_true[i])[:, 0] # Peak time point (True)
        jp = to_array(y_pred[i])[:, 0] # Peak time point (Predicted)

        # محاسبه MLE و Sources Found
        mle, found = multisource_metrics(np.abs(jt), np.abs(jp), pos_gm, neighbors)
        mle_l.append(mle)
        found_l.append(found)

        # محاسبه MSE
        mse = np.mean((jt - jp) ** 2)
        mse_l.append(mse)

        # محاسبه nMSE (نرمال شده با انرژی سیگنال واقعی)
        nmse = mse / (np.mean(jt ** 2) + 1e-10)
        nmse_l.append(nmse)

        # محاسبه AUC
        jt_binary = (np.abs(jt) > 0).astype(int)
        if jt_binary.any() and not jt_binary.all():
            # FIXED: Multiplied by 100 to display properly as percentage in the final report
            auc = roc_auc_score(jt_binary, np.abs(jp)) * 100
            auc_l.append(auc)
        else:
            auc_l.append(np.nan)

    # 2. استخراج مقادیر آماری با استفاده از تابع
    auc_m, auc_mad = median_mad(auc_l)
    mse_m, mse_mad = median_mad(mse_l)
    nmse_m, nmse_mad = median_mad(nmse_l)
    mle_m, mle_mad = median_mad(mle_l)

    # 3. چاپ خروجی
    print("\n" + "="*70)
    print("FINAL PERFORMANCE REPORT (Paper-style)")
    print("="*70)

    print(f"AUC [%]        : {auc_m:.2f} (MAD {auc_mad:.2f})")
    print(f"MSE            : {mse_m:.3e} (MAD {mse_mad:.3e})")
    print(f"nMSE           : {nmse_m:.4f} (MAD {nmse_mad:.4f})")
    print(f"MLE [mm]       : {mle_m:.2f} (MAD {mle_mad:.2f})")
    print(f"% Sources Found: {np.mean(found_l):.2f} (SD {np.std(found_l):.2f})")
    print("="*70)

    return y_true, y_pred, mle_l, found_l, auc_l, mse_l, nmse_l

def plot_results(all_loss, all_val_loss, mle_l):
    # Plot the training loss curve
    plt.figure(figsize=(10, 5))
    if all_loss:
        plt.plot(all_loss, label='Training Loss')
    if all_val_loss:
        plt.plot(all_val_loss, label='Validation Loss')
    plt.title('Training and Validation Loss Curve Over Chunks')
    plt.xlabel('Epochs (cumulative)')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.savefig('training_loss_curve.png')
    logging.info("Saved training loss curve to 'training_loss_curve.png'.")
    plt.close()

    # Plot MLE distribution
    if mle_l:
        plt.figure(figsize=(10, 6))
        sns.histplot(mle_l, bins=30, kde=True, color='skyblue', edgecolor='black')
        plt.title('Distribution of Maximum Localization Error (MLE)')
        plt.xlabel('MLE [mm]')
        plt.ylabel('Frequency')
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.savefig('mle_distribution.png')
        logging.info("Saved MLE distribution plot to 'mle_distribution.png'.")
        plt.close()

def save_and_load_data(y_true, y_pred, mle_l, mse_l, nmse_l, auc_l, found_l):
    save_path = 'neural_analysis_data.npz'

    logging.info("Extracting and saving data...")

    # Extract the peak time points for true and predicted sources
    y_true_extracted = [to_array(y)[:, 0] for y in y_true]
    y_pred_extracted = [to_array(y)[:, 0] for y in y_pred]

    # Save the important arrays and metrics into a compressed numpy archive
    np.savez_compressed(
        save_path,
        y_true_peak=np.array(y_true_extracted),
        y_pred_peak=np.array(y_pred_extracted),
        mle=np.array(mle_l),
        mse=np.array(mse_l),
        nmse=np.array(nmse_l),
        auc=np.array(auc_l),
        sources_found=np.array(found_l)
    )
    logging.info(f"\nSuccessfully saved important neural analysis data to: {save_path}")

    # Test loading
    if os.path.exists(save_path):
        logging.info(f"Loading data from {save_path}...\n")
        with np.load(save_path, allow_pickle=False) as data:
            for key in data.files:
                arr = data[key]
                print(f"--- {key} ---")
                print(f"Shape: {arr.shape}")
                print(f"Data type: {arr.dtype}")
                if arr.size > 0:
                    print(f"Sample values: {arr.flatten()[:5]}")
                print("="*40)
    else:
        logging.error(f"File not found: {save_path}")

def main():
    checkpoints_dir = setup_environment()

    info, fwd, info_test, fwd_test, pos_gm, neighbors, net = initialize_pipeline()

    all_loss, all_val_loss = train_model(
        net, fwd, info, SIM_SETTINGS, checkpoints_dir,
        total_samples=TOTAL_SAMPLES, chunk_size=CHUNK_SIZE
    )

    y_true, y_pred, mle_l, found_l, auc_l, mse_l, nmse_l = evaluate_model(
        net, fwd_test, info_test, SIM_SETTINGS, pos_gm, neighbors
    )

    plot_results(all_loss, all_val_loss, mle_l)

    save_and_load_data(y_true, y_pred, mle_l, mse_l, nmse_l, auc_l, found_l)


if __name__ == "__main__":
    main()
