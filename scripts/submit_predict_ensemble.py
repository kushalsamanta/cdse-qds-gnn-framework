#!/usr/bin/env python
"""
submit_predict_ensemble.py

This script automates the submission of prediction jobs for an ensemble
of ALIGNN models. For each ensemble run folder (e.g., run_0, run_1, ..., run_{N-1})
located in BASE_OUTPUT_DIR, it performs these steps:
  1. Creates a "predict" subfolder inside the run folder.
  2. Copies the run's config file (optional) into the predict folder.
  3. Writes a predict.py file in the predict folder that explicitly sets the prediction 
     root directory to:
         /scratch/gilbreth/samantak/ALIGNN_AIMD_DFT_ML/OH/train_up_to_10k/train_10k/rad_15/root_dir_oh_predict_1
     and then loads the best model checkpoint from "../temp/best_model.pt" relative to the run folder.
  4. Writes a run-specific SBATCH job script (predict.sh) in the predict folder.
     This job script explicitly sets the working directory (-D directive) so that all outputs
     are written into the predict folder.
  5. Submits the job via sbatch.

Please adjust the paths as needed.
"""

import os
import subprocess
import shutil
import time

# ----- Base output directory for ensemble training runs -----
BASE_OUTPUT_DIR = "/scratch/gilbreth/samantak/ALIGNN_AIMD_DFT_ML/Cl/train_to_10k"
ENSEMBLE_SIZE = 20
ENSEMBLE_RUNS = [os.path.join(BASE_OUTPUT_DIR, f"run_{i}") for i in range(ENSEMBLE_SIZE)]

# Fixed prediction root directory for inputs (POSCAR files, id_prop.csv, etc.).
PREDICT_ROOT = "/scratch/gilbreth/samantak/ALIGNN_AIMD_DFT_ML/Cl/train_to_10k/cl_predict_1"

# --- Predict.py content ---
# This predict.py explicitly uses PREDICT_ROOT for input files.
PREDICT_PY_CONTENT = f'''#!/usr/bin/env python
import os
import torch
import csv
from alignn.models.alignn_atomwise import ALIGNNAtomWise, ALIGNNAtomWiseConfig
from jarvis.core.atoms import Atoms
from alignn.graphs import Graph

# Prediction configuration
output_features = 1
# Load the best model checkpoint from the parent "temp" folder.
checkpoint_path = os.path.join("..", "temp", "best_model.pt")
device = "cpu"
if torch.cuda.is_available():
    device = torch.device("cuda")

# Build the model using the same configuration as training.
model = ALIGNNAtomWise(
    ALIGNNAtomWiseConfig(
        name="alignn_atomwise",
        alignn_layers=4,
        gcn_layers=4,
        atom_input_features=92,
        edge_input_features=80,
        triplet_input_features=50,
        embedding_features=64,
        hidden_features=256,
        output_features=1,
        link="identity",
        zero_inflated=False,
        classification=False,
        calculate_gradient=False
    )
)
model.load_state_dict(torch.load(checkpoint_path, map_location=device))
model.eval()
model = model.to(device)

# Graph generation settings.
cutoff = 5.0
max_neighbors = 12

# Use the fixed prediction root directory.
poscar_dir = r"{PREDICT_ROOT}"

output_csv = 'prediction.csv'
with open(output_csv, mode='w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['POSCAR File', 'Predicted Value'])
    for file in os.listdir(poscar_dir):
        if file.lower().endswith('.vasp'):
            poscar_file = os.path.join(poscar_dir, file)
            try:
                atoms = Atoms.from_poscar(poscar_file)
            except Exception as e:
                print(f"Error loading {{file}}: {{e}}")
                continue
            g, lg = Graph.atom_dgl_multigraph(atoms, cutoff=float(cutoff), max_neighbors=max_neighbors)
            prediction = model([g.to(device), lg.to(device)])['out'].detach().cpu().numpy().flatten().tolist()
            writer.writerow([file, prediction[0]])
print("Predictions saved to", output_csv)
'''

# --- Predict.sh content ---
def get_predict_sh_content(predict_folder, run_index):
    # Here we set the working directory explicitly using -D.
    return f'''#!/bin/bash
#SBATCH -A standby
#SBATCH -N 1
#SBATCH --gpus=1
#SBATCH -t 00:30:00
#SBATCH --job-name predict_ran_{run_index}
##SBATCH --constraint='a100'
#SBATCH -o {predict_folder}/slurm-%j.out
#SBATCH -D {predict_folder}

# Load conda environment.
source ~/.bashrc
conda_setup
conda activate alignn_original

python predict.py
'''

def write_predict_files(predict_folder, run_index):
    """Write predict.py and predict.sh into the given predict folder."""
    predict_py_path = os.path.join(predict_folder, "predict.py")
    predict_sh_path = os.path.join(predict_folder, "predict.sh")
    
    with open(predict_py_path, "w") as f:
        f.write(PREDICT_PY_CONTENT)
    with open(predict_sh_path, "w") as f:
        f.write(get_predict_sh_content(predict_folder, run_index))
    
    os.chmod(predict_py_path, 0o755)
    os.chmod(predict_sh_path, 0o755)
    print(f"Predict files written in {predict_folder}")

def submit_predict_job(predict_folder):
    """Submit the prediction job using sbatch."""
    job_script = os.path.join(predict_folder, "predict.sh")
    subprocess.run(["sbatch", job_script], cwd=predict_folder, check=True)

def main():
    for idx, run_folder in enumerate(ENSEMBLE_RUNS):
        if not os.path.isdir(run_folder):
            print(f"Run folder {run_folder} does not exist; skipping.")
            continue

        # Create a predict subfolder inside the run folder.
        predict_folder = os.path.join(run_folder, "predict")
        os.makedirs(predict_folder, exist_ok=True)
        print(f"Created predict folder: {predict_folder}")

        # Optionally copy the config file from the run folder to the predict folder.
        config_src = os.path.join(run_folder, "config.json")
        config_dst = os.path.join(predict_folder, "config.json")
        if os.path.exists(config_src):
            shutil.copy(config_src, config_dst)
            print(f"Copied config file to {config_dst}")
        else:
            print(f"No config file found in {run_folder}.")

        # Write predict.py and predict.sh into the predict folder.
        write_predict_files(predict_folder, idx)

        # Submit the prediction job.
        try:
            submit_predict_job(predict_folder)
            print(f"Submitted prediction job from {predict_folder}")
        except subprocess.CalledProcessError as e:
            print(f"Error submitting prediction job from {predict_folder}: {e}")
        time.sleep(1)
    
    print("All prediction ensemble jobs submitted.")

if __name__ == "__main__":
    import subprocess, shutil
    main()

