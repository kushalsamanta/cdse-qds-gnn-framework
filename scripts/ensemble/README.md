# Ensemble Automation  
**ALIGNN workflow for Cd₂₈Se₁₇Cl₂₂ QD band‑gap training, prediction & atom‑level interpretation**

This repository gathers every script and instruction required to:

* train **20 independent ALIGNN models** on Cl‑passivated CdSe quantum dots (Cd₂₈Se₁₇Cl₂₂)  
* predict **full‑trajectory band‑gaps** (0 – 10 ps)  
* compute **atom‑wise importance scores** via *Feature Nullification Analysis* (FNA)  
* summarise per‑atom error statistics in a single CSV

A frozen copy of all trained checkpoints, prediction outputs and analysis artefacts is archived on Zenodo: **10.5281/zenodo.15359153**.

---

## 0 · Prerequisites  

This pipeline builds directly on the official **ALIGNN** code‐base by *Kamal Choudhary et al.* (GitHub).  
Please cite their original work if you use this repository:

> Choudhary, K.; et al. *Atomistic Line Graph Neural Network for Improved Materials Property Predictions.* **npj Comput. Mater.** 2021, 7, 185.

Key modifications introduced here:

* automated **ensemble** job submission  
* per‐atom masking for **interpretability**  
* **transfer‑learning** hooks for extended trajectories  

---

## 1 · Script Catalogue  

| File | Role |
|------|------|
| **`submit_ensemble.py`** | Launch 20 ALIGNN training jobs on a 10 % subsample of the trajectory (80 / 10 / 10 train / val / test). |
| **`submit_predict_ensemble.py`** | Use each checkpoint to predict band‑gaps for the remaining 90 % frames. |
| **`submit_predict_all_atoms_ensemble.py`** | For every **model × atom label**, run FNA through `predict_atom_imp.py`. |
| **`average_rmse_per_atom_ensemble.py`** | Aggregate RMSEs to `average_rmse_results.csv` (meV). |
| **`predict_atom_imp.py`** | Mask a chosen atom (zero its graph features) and re‑predict. |

---

## 2 · Theory of Operation  

### 2.1 Training subset logic  

* AIMD: **10 ps @ 1 fs → 10 001 frames**  
* Subsample **every 10 fs** → **1 000 geometries** (10 %)  
* For each run *<i>* (0–19):  
  1. shuffle the subset  
  2. split **80 %/10 %/10 %** → train/val/test  
  3. train with `train_alignn.py`  

Ensembling reduces variance and captures dynamics more robustly.

### 2.2 Ensemble prediction  

* **9 001 unseen frames** (the remaining 90 %) are fed to every model.  
* Per‑frame predictions are **averaged** across the 20 runs.

### 2.3 Atom importance (FNA)  

* During inference, **zero** the node & edge features of a target atom.  
* The drop in predicted band‑gap ≡ that atom’s **importance**.  
* Results are saved as `run_<i>/atom_imp_<Species><Index>.csv`.

### 2.4 Transfer learning (10 – 15 ps)  

1. Collect 500 new structures (10 – 15 ps, 10 fs spacing).  
2. Load each pretrained model.  
3. **Freeze** elemental embeddings; fine‑tune the remaining layers with `fine_tune_modified.py`.  
4. Predict extended‑trajectory properties with minimal extra cost.

---

## 3 · Summary Outputs  


