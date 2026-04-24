import os
import joblib
import tensorflow as tf
from google.colab import drive
from esinet.forward import create_forward_model, get_info
from esinet import Simulation, Net

# 1. Mount Google Drive to save progress safely
drive.mount('/content/drive')
save_dir = '/content/drive/MyDrive/esinet_project'
os.makedirs(save_dir, exist_ok=True)

# Define file paths for saving
sim_path = os.path.join(save_dir, 'simulation.pkl')
net_path = os.path.join(save_dir, 'trained_net.pkl')

# 2. Setup Forward Model (this is fast, so we do it every time)
print("Setting up forward model...")
info = get_info()
fwd = create_forward_model(info=info, sampling='ico3')

# 3. Simulation (This takes ~20 minutes, so we load it if we already did it)
if os.path.exists(sim_path):
    print("✅ Found saved simulation on Google Drive! Loading it now (this saves ~20 minutes)...")
    sim = joblib.load(sim_path)
else:
    print("⏳ No saved simulation found. Starting 20-minute simulation...")
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
    sim = Simulation(fwd, info, settings=custom_settings)
    sim.simulate(n_samples=5000)

    print("💾 Saving simulation to Google Drive...")
    joblib.dump(sim, sim_path)
    print("✅ Simulation saved!")

# 4. Neural Network Training
if os.path.exists(net_path):
    print("✅ Found trained network on Google Drive! Loading it now...")
    net = joblib.load(net_path)
else:
    print("⏳ No saved network found. Building and training a new one...")
    net = Net(fwd, model_type='convdip')
    net.fit(sim, epochs=20)

    print("💾 Saving trained network to Google Drive...")
    joblib.dump(net, net_path)
    print("✅ Network saved successfully!")

print("🎉 All done! Your environment is fully ready and backed up.")
