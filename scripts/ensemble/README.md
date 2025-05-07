# Ensemble Training Scripts 📂

This folder contains everything needed to launch the **20‑member ALIGNN
ensemble** that predicts the band‑gap of the Cl‑passivated
Cd<sub>28</sub>Se<sub>17</sub>Cl<sub>22</sub> quantum‑dot trajectory.

---

## Contents

| File / Dir | Purpose |
|------------|---------|
| **`submit_ensemble.py`** | Creates `run_0 … run_19` folders, writes seed‑specific configs, shuffles labels, generates `submit_job.sh`, then submits each job via `sbatch`. |
| **`submit_predict_ensemble.py`** | After all runs finish, loads every checkpoint and merges their predictions into a single CSV. |
| **`submit_predict_all_atoms_ensemble.py`** | Batch‑predicts per‑atom importance scores (Feature Nullification Analysis). |
| **`config_example.json`** | Base ALIGNN hyper‑parameter file; `submit_ensemble.py` clones and inserts a unique `random_seed`. |

---
