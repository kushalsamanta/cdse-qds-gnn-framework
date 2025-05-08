#!/usr/bin/env python
"""
transfer_improved_no_fc.py

This script implements transfer learning for ALIGNN in a feature extraction style—
with only the elemental embedding layer frozen. In this version, the final fully connected (FC)
head of the pretrained model is NOT replaced. Instead, the pretrained model is used as-is, 
so no key mismatches occur when loading the checkpoint.

Usage:
    python transfer_improved_no_fc.py --root_dir <new_data_dir> --config_name <config.json> --restart_model_path <path_to_checkpoint> [other options]
"""

import os
import sys
import json
import csv
import zipfile
import time
import random
import argparse
import torch
import torch.distributed as dist
import torch.optim as optim
import torch.nn as nn

from alignn.data import get_train_val_loaders
from alignn.train import train_dgl
from alignn.config import TrainingConfig
from jarvis.db.jsonutils import loadjson
from alignn.models.alignn_atomwise import ALIGNNAtomWise, ALIGNNAtomWiseConfig
from jarvis.core.atoms import Atoms
from ase.stress import voigt_6_to_full_3x3_stress

# Set device
device = "cpu"
if torch.cuda.is_available():
    device = torch.device("cuda")

def setup(rank=0, world_size=0, port="12356"):
    """Set up distributed training if using multiple GPUs."""
    if port == "":
        port = str(random.randint(10000, 99999))
    if world_size > 1:
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = port
        dist.init_process_group("nccl", rank=rank, world_size=world_size)
        torch.cuda.set_device(rank)

def cleanup(world_size):
    if world_size > 1:
        dist.destroy_process_group()

# ------------------------------------------------------------------------------
# Argument Parsing
# ------------------------------------------------------------------------------
parser = argparse.ArgumentParser(
    description="ALIGNN Transfer Learning via Feature Extraction (Freeze Only Embeddings)"
)
parser.add_argument("--root_dir", default="./",
                    help="Directory with new dataset files (id_prop.json, id_prop.csv, etc.)")
parser.add_argument("--config_name", default="config.json",
                    help="Path to training configuration JSON file")
parser.add_argument("--file_format", default="poscar",
                    help="Format for new data files: poscar/cif/xyz/pdb")
parser.add_argument("--classification_threshold", default=None,
                    help="Threshold for classification (if applicable)")
parser.add_argument("--batch_size", default=None,
                    help="Batch size (e.g., 16)")
parser.add_argument("--epochs", default=None,
                    help="Number of epochs (e.g., 300)")
parser.add_argument("--target_key", default="target",
                    help="Key corresponding to the target property in the dataset")
parser.add_argument("--id_key", default="jid",
                    help="Key for structure ID")
parser.add_argument("--force_key", default="forces",
                    help="Key for gradient-level data (if used)")
parser.add_argument("--atomwise_key", default="forces",
                    help="Key for atomwise-level data")
parser.add_argument("--stresswise_key", default="stresses",
                    help="Key for stress-level data")
parser.add_argument("--output_dir", default="temp",
                    help="Directory for saving outputs")
parser.add_argument("--restart_model_path", default=None,
                    help="Path to the pretrained model checkpoint (required)")
parser.add_argument("--device", default=None,
                    help="Device for training (e.g., cpu, cuda)")

# ------------------------------------------------------------------------------
# Training Function (Feature Extraction Transfer Learning without FC head replacement)
# ------------------------------------------------------------------------------
def train_for_folder(
    rank=0,
    world_size=0,
    root_dir=".",
    config_name="config.json",
    classification_threshold=None,
    batch_size=None,
    epochs=None,
    id_key="jid",
    target_key="target",
    atomwise_key="forces",
    gradwise_key="forces",
    stresswise_key="stresses",
    file_format="poscar",
    restart_model_path=None,
    output_dir=None,
):
    # Step 1: Setup distributed training.
    setup(rank=rank, world_size=world_size)
    print(f"Step 1: Distributed setup complete (Rank: {rank}, World Size: {world_size})")

    # Step 2: Load new data.
    print("Step 2: Loading new data from:", root_dir)
    id_prop_json = os.path.join(root_dir, "id_prop.json")
    id_prop_json_zip = os.path.join(root_dir, "id_prop.json.zip")
    id_prop_csv = os.path.join(root_dir, "id_prop.csv")
    if os.path.exists(id_prop_json):
        dat = loadjson(id_prop_json)
        print("Loaded new data from id_prop.json")
    elif os.path.exists(id_prop_json_zip):
        dat = json.loads(zipfile.ZipFile(id_prop_json_zip).read("id_prop.json"))
        print("Loaded new data from id_prop.json.zip")
    elif os.path.exists(id_prop_csv):
        with open(id_prop_csv, "r") as f:
            reader = csv.reader(f)
            dat = [row for row in reader]
        print("Loaded new data from id_prop.csv")
    else:
        print("No valid new data file found in", root_dir)
        sys.exit(1)

    dataset = []
    for entry in dat:
        info = {}
        if isinstance(entry, list):
            file_name = entry[0].strip()
            info["jid"] = file_name
            values = [float(val) for val in entry[1:]]
            info["target"] = values[0] if len(values) == 1 else values
            file_path = os.path.join(root_dir, file_name)
            if file_format.lower() == "poscar":
                atoms = Atoms.from_poscar(file_path)
            elif file_format.lower() == "cif":
                atoms = Atoms.from_cif(file_path)
            elif file_format.lower() == "xyz":
                atoms = Atoms.from_xyz(file_path, box_size=500)
            elif file_format.lower() == "pdb":
                atoms = Atoms.from_pdb(file_path, max_lat=500)
            else:
                raise NotImplementedError(f"File format '{file_format}' not implemented.")
            info["atoms"] = atoms.to_dict()
        else:
            info["jid"] = entry[id_key]
            info["target"] = entry[target_key]
            info["atoms"] = entry["atoms"]
        dataset.append(info)
    print("Step 2: Number of new data points loaded:", len(dataset))

    # Step 3: Load training configuration.
    print("Step 3: Loading configuration from:", config_name)
    config_dict = loadjson(config_name)
    config = TrainingConfig(**config_dict)
    if classification_threshold is not None:
        config.classification_threshold = float(classification_threshold)
    if output_dir is not None:
        config.output_dir = output_dir
    if batch_size is not None:
        config.batch_size = int(batch_size)
    if epochs is not None:
        config.epochs = int(epochs)
    print(f"Step 3: Config loaded. Epochs: {config.epochs}, Batch Size: {config.batch_size}, Learning Rate: {config.learning_rate}")

    # Step 4: Load the pretrained model.
    if restart_model_path is None:
        raise ValueError("Pretrained model checkpoint (--restart_model_path) is required.")
    print("Step 4: Loading pretrained model from:", restart_model_path)
    rest_config = loadjson(restart_model_path.replace("current_model.pt", "config.json"))
    pretrained_config = ALIGNNAtomWiseConfig(**rest_config["model"])
    model = ALIGNNAtomWise(pretrained_config)
    model.load_state_dict(torch.load(restart_model_path, map_location=device))
    model = model.to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Step 4: Pretrained model loaded. Total model parameters: {total_params}")

    # Step 5: Freeze only the pretrained elemental embeddings.
    for param in model.atom_embedding.parameters():
        param.requires_grad = False
    frozen_embedding_params = sum(p.numel() for p in model.atom_embedding.parameters() if not p.requires_grad)
    print(f"Step 5: Elemental embeddings frozen. Frozen parameters in embedding: {frozen_embedding_params}")

    # Step 6: (Removed) Retain the original final FC head.
    print("Step 6: Retaining original final FC head.")

    # Step 7: Set up the AdamW optimizer for all trainable parameters.
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.AdamW(trainable_params, lr=config.learning_rate, weight_decay=config.weight_decay)
    object.__setattr__(config, "custom_optimizer", optimizer)
    total_trainable = sum(p.numel() for p in trainable_params)
    print(f"Step 7: Optimizer set. Total trainable parameters: {total_trainable}")

    # Step 8: Prepare data loaders.
    print("Step 8: Preparing data loaders...")
    line_graph = config.model.alignn_layers > 0
    train_loader, val_loader, test_loader, prepare_batch = get_train_val_loaders(
        dataset_array=dataset,
        target="target",
        target_atomwise="atomwise_target" if "atomwise_target" in dataset[0] else "",
        target_grad="atomwise_grad" if "atomwise_grad" in dataset[0] else "",
        target_stress="stresses" if "stresses" in dataset[0] else "",
        n_train=config.n_train,
        n_val=config.n_val,
        n_test=config.n_test,
        train_ratio=config.train_ratio,
        val_ratio=config.val_ratio,
        test_ratio=config.test_ratio,
        line_graph=line_graph,
        batch_size=config.batch_size,
        atom_features=config.atom_features,
        neighbor_strategy=config.neighbor_strategy,
        standardize=config.atom_features != "cgcnn",
        id_tag=config.id_tag,
        pin_memory=config.pin_memory,
        workers=config.num_workers,
        save_dataloader=config.save_dataloader,
        use_canonize=config.use_canonize,
        filename=config.filename,
        cutoff=config.cutoff,
        cutoff_extra=config.cutoff_extra,
        max_neighbors=config.max_neighbors,
        output_features=config.model.output_features,
        classification_threshold=config.classification_threshold,
        target_multiplication_factor=config.target_multiplication_factor,
        standard_scalar_and_pca=config.standard_scalar_and_pca,
        keep_data_order=config.keep_data_order,
        output_dir=config.output_dir,
        use_lmdb=config.use_lmdb
    )
    print("Step 8: Data loaders ready.")

    # Step 9: Start training.
    t_start = time.time()
    print(f"Step 9: Starting training (Rank: {rank}, World Size: {world_size})")
    train_dgl(
        config,
        model=model,
        train_val_test_loaders=[train_loader, val_loader, test_loader, prepare_batch],
        rank=rank,
        world_size=world_size,
    )
    t_end = time.time()
    print(f"Step 9: Training completed in {t_end - t_start:.2f} seconds.")
    cleanup(world_size)

# ------------------------------------------------------------------------------
# Main Entry Point
# ------------------------------------------------------------------------------
if __name__ == "__main__":
    args = parser.parse_args(sys.argv[1:])
    world_size = int(torch.cuda.device_count())
    print("Main: Detected World Size:", world_size)
    if world_size > 1:
        torch.multiprocessing.spawn(
            train_for_folder,
            args=(
                world_size,
                args.root_dir,
                args.config_name,
                args.classification_threshold,
                args.batch_size,
                args.epochs,
                args.id_key,
                args.target_key,
                args.atomwise_key,
                args.force_key,
                args.stresswise_key,
                args.file_format,
                args.restart_model_path,
                args.output_dir,
            ),
            nprocs=world_size,
        )
    else:
        train_for_folder(
            0,
            world_size,
            args.root_dir,
            args.config_name,
            args.classification_threshold,
            args.batch_size,
            args.epochs,
            args.id_key,
            args.target_key,
            args.atomwise_key,
            args.force_key,
            args.stresswise_key,
            args.file_format,
            args.restart_model_path,
            args.output_dir,
        )
    try:
        cleanup(world_size)
    except Exception:
        pass

