#!/usr/bin/env python
"""
predict_atom_imp.py

This script loads a pretrained ALIGNNAtomWise model (from the parent "temp" folder)
and—for each POSCAR file in a given common prediction directory—computes a modified prediction
by zeroing out (i.e. setting to zero) the contributions from a specified target atom. In particular,
it zeroes:
  (a) The node features for the target atom.
  (b) The edge features (bond lengths, stored in g.edata['r']) for any edge that involves the target atom.
  (c) The angle features (stored in lg.edata['h']) in the line graph for any edge whose corresponding original edge
      involves the target atom.
The output (modified prediction) is saved to a CSV file in the current run folder.
 
Usage:
    python predict_atom_imp.py --run_folder <full_run_folder> --target_atom <TARGET> --poscar_dir <common_poscar_dir> [--cutoff <cutoff>] [--max_neighbors <n>]
Example:
    python predict_atom_imp.py --run_folder "/path/to/run_2/atom_imp_Cd7" --target_atom Cd7 --poscar_dir "/path/to/poscars" --cutoff 5.0 --max_neighbors 15
"""

import os
import re
import torch
import csv
import argparse
import numpy as np
from alignn.models.alignn_atomwise import ALIGNNAtomWise, ALIGNNAtomWiseConfig
from jarvis.core.atoms import Atoms
from alignn.graphs import Graph

def parse_args():
    parser = argparse.ArgumentParser(
        description="Predict with modified graph (zeroing target atom contributions)."
    )
    parser.add_argument("--run_folder", required=True,
                        help="Full path to the run folder (e.g., /.../run_2/atom_imp_Cd7).")
    parser.add_argument("--target_atom", required=True,
                        help="Target atom label (e.g., 'Cd7').")
    parser.add_argument("--poscar_dir", required=True,
                        help="Common directory containing POSCAR (.vasp) files.")
    parser.add_argument("--cutoff", type=float, default=5.0,
                        help="Graph construction cutoff (default: 5.0).")
    parser.add_argument("--max_neighbors", type=int, default=12,
                        help="Maximum number of neighbors (default: 15).")
    return parser.parse_args()

def main():
    args = parse_args()

    # Print and interpret the target atom.
    target_atom = args.target_atom.strip()  # e.g., "Cd7"
    print("Using target atom label:", target_atom)
    m = re.match(r"([A-Za-z]+)(\d+)", target_atom)
    if not m:
        raise ValueError(f"Invalid target atom format: {target_atom}. Expected format like 'Cd7'.")
    element_type, occ_str = m.groups()
    target_occurrence = int(occ_str)
    print(f"Interpreted as element '{element_type}' with occurrence {target_occurrence}")

    # Set device.
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print("Using device:", device)

    # Set the output directory to the provided run folder (assumed to be the proper target folder).
    run_folder = os.path.normpath(args.run_folder)
    print("Output directory for predictions:", run_folder)

    # --- Load pretrained model from parent "temp" folder ---
    # Assume the pretrained checkpoint is in the parent of the run_folder in subfolder "temp".
    parent_run_folder = os.path.join(os.path.dirname(run_folder), "temp")
    checkpoint_path = os.path.join(parent_run_folder, "best_model.pt")
    print("Loading pretrained model from:", checkpoint_path)

    # Model configuration: make sure these match your training settings.
    model_config = ALIGNNAtomWiseConfig(
        name="alignn_atomwise",
        alignn_layers=4,
        gcn_layers=4,
        atom_input_features=92,
        edge_input_features=80,         # Must match training configuration.
        triplet_input_features=50,        # Must match training configuration.
        embedding_features=64,
        hidden_features=256,
        output_features=1,
        grad_multiplier=-1,
        calculate_gradient=True,
        graphwise_weight=1.0,
        zero_inflated=False,
        classification=False
    )
    model = ALIGNNAtomWise(model_config)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()
    model.to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print("Pretrained model loaded. Total model parameters:", total_params)

    # --- Prepare output CSV file in the run folder ---
    output_csv = os.path.join(run_folder, f"predictions_{target_atom}.csv")
    with open(output_csv, mode="w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["POSCAR File", "Modified Prediction"])

        # Process each POSCAR file in the provided prediction directory.
        poscar_files = sorted([f for f in os.listdir(args.poscar_dir) if f.lower().endswith(".vasp")])
        print(f"Found {len(poscar_files)} POSCAR files in {args.poscar_dir}.")

        for poscar_file in poscar_files:
            poscar_path = os.path.join(args.poscar_dir, poscar_file)
            print(f"\nProcessing file: {poscar_file}")
            try:
                atoms = Atoms.from_poscar(poscar_path)
            except Exception as e:
                print(f"  Error loading {poscar_file}: {e}")
                continue

            # Identify the target atom occurrence in the file.
            atom_counter = 0
            target_index = None
            for idx, label in enumerate(atoms.elements):
                if label.strip().lower() == element_type.lower():
                    atom_counter += 1
                    if atom_counter == target_occurrence:
                        target_index = idx
                        break
            if target_index is None:
                print(f"  Target atom {target_atom} not found in {poscar_file}. Skipping file.")
                continue
            print(f"  Found target atom '{target_atom}' at index {target_index}.")

            # --- Generate the original graph and line graph.
            g, lg = Graph.atom_dgl_multigraph(atoms, cutoff=args.cutoff, max_neighbors=args.max_neighbors)
            
            # --- Debug: Print original features for target edges.
            if "atom_features" in g.ndata:
                print("  Original node feature for target atom:", g.ndata["atom_features"][target_index])
            src, dst = g.edges()
            edge_mask = (src == target_index) | (dst == target_index)
            if "r" in g.edata:
                orig_edge_r = g.edata["r"][edge_mask].clone()
                print("  Original edge 'r' features for edges involving target:", orig_edge_r)
            # For the line graph, try to zero out the angles for edges mapping to the flagged edges.
            # If the line graph stores the mapping in "edge_ids", we use that.
            if "edge_ids" in lg.ndata and "h" in lg.edata:
                edge_ids = lg.ndata["edge_ids"]  # Shape: (num_line_nodes,)
                flagged_edge_indices = torch.nonzero(edge_mask).squeeze()
                lg_mask = torch.zeros(lg.edata["h"].shape[0], dtype=torch.bool)
                for idx in flagged_edge_indices:
                    lg_mask |= (edge_ids == idx)
                if lg_mask.sum() > 0:
                    orig_lg_h = lg.edata["h"][lg_mask].clone()
                    print("  Original line graph 'h' features for flagged edges:", orig_lg_h)
                else:
                    print("  No flagged line graph entries found via 'edge_ids'.")
            else:
                # If no mapping is found but the number of line graph nodes matches the number of edges in g,
                # assume ordering is the same.
                if "h" in lg.edata and lg.edata["h"].shape[0] == g.edata["r"].shape[0]:
                    lg_mask = edge_mask.clone()
                    orig_lg_h = lg.edata["h"][lg_mask].clone()
                    print("  Original line graph 'h' features (by ordering):", orig_lg_h)
                else:
                    lg_mask = None
                    print("  No 'edge_ids' mapping found and inconsistent ordering; skipping zeroing of line graph angles.")

            # --- Zero out contributions from the target atom.
            # (A) Zero the target node feature.
            if "atom_features" in g.ndata:
                orig_node_feat = g.ndata["atom_features"][target_index].clone()
                g.ndata["atom_features"][target_index] = torch.zeros_like(orig_node_feat)
                print("  Node feature before zeroing:", orig_node_feat)
                print("  Node feature after zeroing:", g.ndata["atom_features"][target_index])
            # (B) Zero out edge features ('r') for edges involving the target atom.
            if "r" in g.edata:
                g.edata["r"][edge_mask] = 0.0
                print("  Edge 'r' features after zeroing:", g.edata["r"][edge_mask])
            # (C) Zero out line graph angle features ('h') for the corresponding edges.
            if (lg_mask is not None) and ("h" in lg.edata):
                lg.edata["h"][lg_mask] = 0.0
                print("  Line graph 'h' features after zeroing:", lg.edata["h"][lg_mask])

            # --- Make a prediction using the modified graph.
            try:
                output = model([g.to(device), lg.to(device)])["out"]
                pred = output.detach().cpu().numpy().flatten()[0]
                print(f"  Modified prediction for {poscar_file}: {pred}")
            except Exception as e:
                print(f"  Error during prediction for {poscar_file}: {e}")
                continue

            writer.writerow([poscar_file, pred])
    
    print(f"\nPredictions saved to {output_csv}")

if __name__ == "__main__":
    main()

