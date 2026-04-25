# EEG Simulation Setup and Execution Guide (Windows PowerShell)

This guide provides the steps to set up your environment and run the EEG source localization simulation on Windows.

## 1. Prerequisites
- Python 3.9 or higher installed (Check with `python --version`).
- NVIDIA GPU with CUDA installed (Optional, but recommended for faster training).

## 2. Setup Environment
Open Windows PowerShell and run the following commands:

```powershell
# Create a virtual environment
python -m venv venv

# Activate the virtual environment
.\venv\Scripts\Activate.ps1

# Upgrade pip
python -m pip install --upgrade pip

# Install dependencies
pip install -r requirements.txt
```

## 3. Run Simulation
Once the environment is set up, run the simulation script:

```powershell
python eeg_simulation_win.py
```

## 4. Results
The simulation will create a folder named `esinet_project_results` containing:
- `trained_net.keras`: The trained neural network model.
- `performance_report.txt`: A text summary of the model's accuracy.
- `training_report.png`: Plots showing the learning curve and localization error distribution.
- `sim_chunk_*.pkl`: Saved simulation data (to allow resuming).
