#!/usr/bin/env python
"""
submit_ensemble.py

This script automatically submits jobs for an ensemble of 20 ALIGNN training runs.
For each run, it:
  - Creates a unique run folder (e.g., run_0, run_1, …, run_19) under a base output directory.
  - Copies the base config file (config_example.json) into that run folder and updates the "random_seed" field
    with a unique random seed.
  - Copies the entire "root_dir" folder (from the current working directory) into the run folder.
  - Copies and shuffles the master id_prop.csv file (from the current directory) into the copied root_dir folder.
  - Writes a run-specific SBATCH job script that calls the training script with --root_dir set to the run-specific "root_dir".
  - Submits the job using sbatch so that outputs (e.g., slurm logs, checkpoints, and temp directories)
    remain in the run folder.

Please adjust the BASE_SOURCE_ROOT_DIR, BASE_CONFIG, BASE_OUTPUT_DIR, and other paths as needed.
"""

import os
import json
import random
import subprocess
import time
import shutil
import pandas as pd

# ----- Base directories and ensemble settings -----
# The BASE_SOURCE_ROOT_DIR should be the full path to the folder named "root_dir" in your current working directory.
BASE_SOURCE_ROOT_DIR = os.path.join(os.getcwd(), "root_dir")
# The master id_prop.csv file is expected to be in the current working directory.
BASE_ID_PROP = os.path.join(os.getcwd(), "id_prop.csv")
# Base configuration file.
BASE_CONFIG = "/scratch/gilbreth/samantak/ALIGNN_AIMD_DFT_ML/Cl/train_to_10k/config_example.json"
# Output directory where run folders will be created.
BASE_OUTPUT_DIR = "/scratch/gilbreth/samantak/ALIGNN_AIMD_DFT_ML/Cl/train_to_10k"
# Set ensemble size.
ENSEMBLE_SIZE = 20

# Pre-generate a list of unique seeds.
# This produces a list of 20 unique integers between 10000 and 99999.
unique_seeds = random.sample(range(10000, 100000), ENSEMBLE_SIZE)

def update_config(base_config_path, new_config_path, new_seed):
    """Read the base config file, update 'random_seed' with new_seed, and save to new_config_path."""
    with open(base_config_path, "r") as f:
        cfg = json.load(f)
    cfg["random_seed"] = new_seed
    with open(new_config_path, "w") as f:
        json.dump(cfg, f, indent=4)

def copy_and_shuffle_idprop(source_csv, dest_csv, seed):
    """
    Read the id_prop.csv file from source_csv, shuffle its rows using the provided seed,
    and save the shuffled DataFrame to dest_csv.
    """
    try:
        df = pd.read_csv(source_csv)
    except Exception as e:
        print(f"Error reading {source_csv}: {e}")
        return False
    df_shuffled = df.sample(frac=1, random_state=seed).reset_index(drop=True)
    df_shuffled.to_csv(dest_csv, index=False)
    return True

def write_job_script(job_script_path, run_dir, new_run_root_dir, config_path, run_index):
    """
    Write a run-specific SBATCH submission script.
    The script changes directory to the run folder and calls the training script using the run-local root_dir.
    """
    job_script = f"""#!/bin/bash
#SBATCH -A standby
#SBATCH -N 1
#SBATCH --gpus=1
#SBATCH -t 02:00:00
#SBATCH --job-name cl10k_run_{run_index}
#SBATCH -o {run_dir}/slurm-%j.out

# Change to the run folder so that outputs are stored there
cd "{run_dir}"

# Set PATH for CUDA
export PATH=/usr/local/cuda/bin:$PATH

# Load conda environment
source ~/.bashrc
conda_setup
conda activate alignn_original

# Run the training script.
# It uses the new config file and reads id_prop.csv from the run-specific root_dir.
train_alignn.py_bak --root_dir "{new_run_root_dir}" --config "{config_path}" --output_dir "./temp"
"""
    with open(job_script_path, "w") as f:
        f.write(job_script)
    os.chmod(job_script_path, 0o755)

def submit_job(job_script_path):
    """Submit the job using sbatch."""
    subprocess.run(["sbatch", job_script_path], check=True)

def main():
    for i in range(ENSEMBLE_SIZE):
        # Create a unique run folder inside BASE_OUTPUT_DIR.
        run_dir = os.path.join(BASE_OUTPUT_DIR, f"run_{i}")
        os.makedirs(run_dir, exist_ok=True)
        print(f"Created ensemble run folder: {run_dir}")

        # Get the pre-generated unique seed for this run.
        new_seed = unique_seeds[i]
        print(f"Run {i}: Generated unique seed: {new_seed}")

        # Create a new config file in the run folder.
        new_config_path = os.path.join(run_dir, "config.json")
        update_config(BASE_CONFIG, new_config_path, new_seed)
        print(f"Run {i}: New config file written to: {new_config_path}")

        # Copy the entire "root_dir" folder from BASE_SOURCE_ROOT_DIR into the run folder.
        dest_root_dir = os.path.join(run_dir, "root_dir")
        if os.path.exists(BASE_SOURCE_ROOT_DIR):
            if os.path.exists(dest_root_dir):
                shutil.rmtree(dest_root_dir)
            shutil.copytree(BASE_SOURCE_ROOT_DIR, dest_root_dir)
            print(f"Run {i}: Copied 'root_dir' from {BASE_SOURCE_ROOT_DIR} to {dest_root_dir}")
        else:
            print(f"Run {i}: BASE_SOURCE_ROOT_DIR not found at {BASE_SOURCE_ROOT_DIR}")

        # Copy and shuffle the id_prop.csv file.
        dest_idprop_path = os.path.join(dest_root_dir, "id_prop.csv")
        if os.path.exists(BASE_ID_PROP):
            success = copy_and_shuffle_idprop(BASE_ID_PROP, dest_idprop_path, new_seed)
            if success:
                print(f"Run {i}: Shuffled id_prop.csv copied to {dest_idprop_path}")
            else:
                print(f"Run {i}: Failed to copy and shuffle id_prop.csv")
        else:
            print(f"Run {i}: BASE id_prop.csv not found at {BASE_ID_PROP}")

        # Write the run-specific SBATCH submission script.
        job_script_path = os.path.join(run_dir, "submit_job.sh")
        write_job_script(job_script_path, run_dir, dest_root_dir, new_config_path, i)
        print(f"Run {i}: Job script created at: {job_script_path}")

        # Submit the job.
        try:
            submit_job(job_script_path)
            print(f"Run {i}: Job submitted successfully.")
        except subprocess.CalledProcessError as e:
            print(f"Run {i}: Error submitting job: {e}")

        # Pause briefly between submissions.
        time.sleep(1)
    print("All ensemble jobs submitted.")

if __name__ == "__main__":
    main()

