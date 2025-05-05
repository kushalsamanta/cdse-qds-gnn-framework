# CdSe-QDs-GNN-Framework
<p align="justify">
Machine learning framework for predicting time-resolved electronic properties in ligand-passivated CdSe quantum dots (Cd<sub>28</sub>Se<sub>17</sub>X<sub>22</sub>, X = Cl, OH). This repository includes AIMD trajectories, DFT-calculated electronic properties, and graph-based neural network models (ALIGNN and CGCNN), along with atom-specific importance analyses via Feature Nullification Analysis (FNA). Developed for studying bandgap and subgap fluctuations over extended trajectories using transfer learning and minimal DFT sampling.

</p>
<img src="assets/kushal_gp.png" alt="Framework overview" width="800">

---

## What’s inside 📂
| Folder | Contents |
|--------|----------|
| `src/` | Training & inference scripts for **ALIGNN** and **CGCNN** |
| `assets/` | Figures (framework overview, key results) |
| `data/` | Small demo trajectory + DFT labels (full datasets on Zenodo) |
| `notebooks/` | Reproduce the paper’s parity plots and time‑series |
| `tests/` | Minimal pytest suite for CI |

---
