#!/usr/bin/env python
"""
submit_predict_all_atoms_ensemble.py

This script automates the submission of prediction jobs over an ensemble of ALIGNN runs,
for all atomic labels detected in a reference POSCAR file.
For each ensemble run folder (e.g., run_0, run_1, …) in BASE_ENSEMBLE_DIR:
  - It reads a reference POSCAR file from COMMON_POSCAR_DIR to extract atomic labels.
  - For each detected atom label (e.g., "Cd1", "Cd2", "Se1", etc.), it creates a subfolder
    (named "atom_imp_<label>") inside the run folder.
  - It copies the common prediction script (predict_atom_imp.py) into that subfolder.
  - It writes a run‐specific SBATCH job script (predict.sh) in that subfolder that changes 
    the directory to the subfolder and calls predict_atom_imp.py with the appropriate arguments.
  - The job is then submitted via sbatch.
  
All outputs (SLURM log files, predictions CSV, etc.) are contained within each "atom_imp_<label>" folder.

Usage:
    python submit_predict_all_atoms_ensemble.py
"""

import os
import subprocess
import time
import shutil
import json
import re

# ===== CONFIGURATION: adjust these paths as needed =====

# Base ensemble directory where ensemble run folders (e.g., run_0, run_1, …) reside.
BASE_ENSEMBLE_DIR = "/scratch/gilbreth/samantak/ALIGNN_AIMD_DFT_ML/Cl/train_to_10k"

# Common POSCAR prediction directory (contains all POSCAR files for prediction).
COMMON_POSCAR_DIR = "/scratch/gilbreth/samantak/ALIGNN_AIMD_DFT_ML/Cl/train_to_10k/root_dir_atom_imp"

# Path to the common prediction script (predict_atom_imp.py)
PREDICT_SCRIPT_SOURCE = os.path.abspath("predict_atom_imp.py")

# Graph generation parameters (adjust as needed)
CUTOFF = 5.0
MAX_NEIGHBORS = 12

# =================================================================

def get_atom_labels_from_poscar(poscar_filepath):
    """
    Reads a POSCAR file and extracts atomic labels from the Cartesian coordinate block.
    Assumes the file contains a line with "Cartesian" after which each nonempty line
    has atomic coordinates and the last token is the atom label.
    Returns a list of atomic labels (e.g. ["Cd1", "Cd2", "Se1", "O1", "H1", ...]).
    """
    with open(poscar_filepath, "r") as f:
        lines = f.readlines()
    start = None
    for i, line in enumerate(lines):
        if "Cartesian" in line:
            start = i + 1
            break
    if start is None:
        raise ValueError(f"No 'Cartesian' line found in {poscar_filepath}")
    labels = []
    for line in lines[start:]:
        if line.strip() == "":
            continue
        parts = line.strip().split()
        # Last token is assumed to be the atomic label.
        labels.append(parts[-1])
    return labels

def write_job_script(script_path, work_dir, target_atom, run_folder):
    """
    Writes an SBATCH submission script (predict.sh) into work_dir.
    The script changes directory to work_dir, ensuring that all outputs 
    (SLURM logs, predictions CSV, etc.) end up in that folder.
    The job name is set to include the run folder name and target atom label.
    """
    script_content = f"""#!/bin/bash
#SBATCH -A standby
#SBATCH -N 1
#SBATCH --gpus=1
#SBATCH -t 00:30:00
#SBATCH --job-name {os.path.basename(run_folder)}_{target_atom}
##SBATCH --constraint='a100'
#SBATCH -o {work_dir}/slurm-%j.out

# Change to the working folder so that all outputs are in {work_dir}
cd "{work_dir}"

# Set PATH for CUDA.
export PATH=/usr/local/cuda/bin:$PATH

# Load the conda environment.
source ~/.bashrc
conda_setup
conda activate alignn_original

# Run the prediction script with appropriate arguments.
python predict_atom_imp.py --run_folder "$(pwd)" --target_atom "{target_atom}" --poscar_dir "{COMMON_POSCAR_DIR}" --cutoff {CUTOFF} --max_neighbors {MAX_NEIGHBORS}
"""
    with open(script_path, "w") as f:
        f.write(script_content)
    os.chmod(script_path, 0o755)

def main():
    # Use a reference POSCAR file from the common prediction directory.
    poscar_files = sorted([f for f in os.listdir(COMMON_POSCAR_DIR) if f.lower().endswith(".vasp")])
    if not poscar_files:
        raise ValueError(f"No POSCAR files found in {COMMON_POSCAR_DIR}")
    reference_file = os.path.join(COMMON_POSCAR_DIR, poscar_files[0])
    print("Using reference POSCAR file for extracting atomic labels:", reference_file)
    
    try:
        atom_labels = get_atom_labels_from_poscar(reference_file)
    except Exception as e:
        raise ValueError(f"Error extracting atomic labels: {e}")
    
    # Remove duplicates while preserving order (if a structure has repeated labels, keep all occurrences)
    # Here we return the labels as they appear (e.g., ["Cd1", "Cd2", "Cd3", ...]).
    print(f"Extracted {len(atom_labels)} atomic labels: {atom_labels}")

    # Get ensemble run folders (directories starting with "run_") inside the base ensemble directory.
    ensemble_folders = [os.path.join(BASE_ENSEMBLE_DIR, d) for d in os.listdir(BASE_ENSEMBLE_DIR) if d.startswith("run_")]
    if not ensemble_folders:
        raise ValueError(f"No ensemble run folders found in {BASE_ENSEMBLE_DIR}")
    print(f"Found {len(ensemble_folders)} ensemble run folders.")

    # For each ensemble run folder...
    for run_folder in ensemble_folders:
        print(f"\nProcessing ensemble run folder: {run_folder}")
        # For each atomic label extracted from the reference POSCAR...
        for target_atom in atom_labels:
            # Create the target folder within the run folder.
            target_folder = os.path.join(run_folder, f"atom_imp_{target_atom}")
            os.makedirs(target_folder, exist_ok=True)
            print(f"  Created target folder: {target_folder}")
            
            # Copy predict_atom_imp.py into the target folder.
            dest_predict_script = os.path.join(target_folder, "predict_atom_imp.py")
            shutil.copy(PREDICT_SCRIPT_SOURCE, dest_predict_script)
            print(f"  Copied predict_atom_imp.py to: {dest_predict_script}")

            # (Optionally, copy a config file as needed; if the prediction script requires one, do so here.)

            # Write the SBATCH submission script (predict.sh) in the target folder.
            job_script_path = os.path.join(target_folder, "predict.sh")
            write_job_script(job_script_path, target_folder, target_atom, run_folder)
            print(f"  Written SBATCH script at: {job_script_path}")

            # Submit the job.
            try:
                subprocess.run(["sbatch", job_script_path], check=True)
                print(f"  Submitted prediction job for target atom '{target_atom}' in folder: {target_folder}")
            except subprocess.CalledProcessError as e:
                print(f"  Job submission failed for folder {target_folder}: {e}")
            # Pause briefly between job submissions.
            time.sleep(1)

    print("\nAll ensemble prediction jobs submitted.")

if __name__ == "__main__":
    main()

