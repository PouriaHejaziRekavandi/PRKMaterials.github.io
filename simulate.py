import sys
sys.path.insert(0, '../')
from esinet import Simulation, Net
from esinet.forward import create_forward_model, get_info

# Create generic Forward Model
info = get_info()
fwd = create_forward_model(info=info, sampling='ico2')

# Simulate M/EEG data
settings = dict(duration_of_trial=0.1)
sim = Simulation(fwd, info, settings=settings)
sim.simulate(n_samples=1000)

# Train neural network (LSTM) on the simulated data
net = Net(fwd)
net.fit(sim, epochs=30)

# Plot
stc = net.predict(sim.eeg_data[0])[0]
sim.source_data[0].plot(surface="white", hemi="both")
stc.plot(surface="white", hemi="both")
