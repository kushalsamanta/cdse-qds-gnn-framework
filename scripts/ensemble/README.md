# Ensemble Automation  
**ALIGNN workflow for Cd<sub>28</sub>Se<sub>17</sub>Cl<sub>22</sub> quantum‑dot band‑gap training, prediction, atom‑level interpretation & transfer‑learning to extended trajectories**

This repository bundles every script and guideline you need to

* train **20 independent ALIGNN models** on Cl‑passivated CdSe QDs (Cd<sub>28</sub>Se<sub>17</sub>Cl<sub>22</sub>)  
* predict **full‑trajectory band‑gaps** for the first 10 ps  
* compute **atom‑wise importance scores** with *Feature Nullification Analysis* (FNA)  
* **transfer‑learn** those models to a 10 – 15 ps continuation

A frozen snapshot of all checkpoints, predictions and analysis artefacts lives on Zenodo:  
<https://doi.org/10.5281/zenodo.15359153>

---

## 0 · Prerequisites

This pipeline is built on the official **ALIGNN** code‑base by *Kamal Choudhary et al.* –  
<https://github.com/usnistgov/alignn>  

If you use this repository, please cite:

> Choudhary, K.; et al. *Atomistic Line Graph Neural Network for Improved Materials Property Predictions.* **npj Comput. Mater.** 2021, 7, 185.

Our main additions are

* automated **ensemble** job submission  
* per‑atom masking for **interpretability**  
* **transfer‑learning** hooks for longer trajectories  

---

## 1 · Script Catalogue

| File | Role |
|------|------|
| **`submit_ensemble.py`** | Launch 20 ALIGNN training jobs on a 10 % subsample (80 / 10 / 10 train / val / test), shuffling labels for every run. |
| **`submit_predict_ensemble.py`** | Use each checkpoint to predict band‑gaps for the remaining 90 % frames. |
| **`submit_predict_all_atoms_ensemble.py`** | For every **model × each atom label**, run FNA through `predict_atom_imp.py`. |
| **`average_rmse_per_atom_ensemble.py`** | Aggregate ensemble RMSEs into `average_rmse_results.csv` (meV). |
| **`predict_atom_imp.py`** | Mask one atom (zero its graph features) and re‑predict the band‑gap. |
| **`fine_tune_modified.py`** | Freeze embeddings and fine‑tune the model head for 10 – 15 ps data. |

---

## 2 · Theory of Operation

### 2.1 Training subset logic
* **AIMD trajectory**: 10 ps @ 1 fs → 10 001 frames  
* **Subsample** every 10 fs → 1 000 geometries (10 %)  
* For each run *i* (0 – 19)  
  1. shuffle the 1 000 frames  
  2. split **80 %/10 %/10 %** → train/val/test  
  3. train with `train_alignn.py` from the ALIGNN repo (see citation above)  

The 20‑model ensemble reduces variance and captures dynamical fluctuations robustly.

### 2.2 Ensemble prediction
* **9 001 unseen frames** (the other 90 %) are sent to every model.  
* Per‑frame outputs are **averaged** across the 20 runs.

### 2.3 Atom importance (FNA)
* During inference we **zero** the node & edge features of one atom.  
* The resulting drop in predicted band‑gap is interpreted as that atom’s **importance**.  
* Outputs reside in `run_<i>/atom_imp_<Species><Index>/`.

### 2.4 Transfer learning (10 – 15 ps)
1. Collect **500 new structures + DFT labels** (10 – 15 ps, 10 fs spacing).  
2. Load all 20 pretrained models.  
3. **Freeze** elemental embeddings & first three message‑passing blocks.  
4. Fine‑tune remaining layers with `fine_tune_modified.py` (≈ 50 epochs, early stop).  
5. Predict extended‑trajectory band‑gaps at a fraction of the original cost.

---

