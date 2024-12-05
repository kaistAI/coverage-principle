#!/usr/bin/env python3
"""
build_trajectories.py

Build trajectory matrices from collected vectors across checkpoints.

Enhanced to utilize multiprocessing for both groups and checkpoint files, leveraging available CPUs.

Usage:
    python build_trajectories.py --save_dir /path/to/checkpoints/ \
        --output_dir /path/to/trajectories/ \
        --groups train_inferred test_inferred_iid test_inferred_ood \
        --layer 5 --position 1 --id_field id --num_workers 16

Author: Your Name
Date: YYYY-MM-DD
"""

import os
import json
import argparse
from collections import defaultdict
import numpy as np
from tqdm import tqdm
import multiprocessing as mp
import h5py
import re

def collect_datapoint_ids_and_vector_dim(save_dir, first_checkpoint_file, groups, layer, position, id_field):
    """
    Collects datapoint IDs and determines vector dimensions from the first checkpoint file.

    Args:
        save_dir (str): Directory containing checkpoint JSON files.
        first_checkpoint_file (str): Filename of the first checkpoint.
        groups (list of str): List of group names to process.
        layer (int): Layer number to extract vectors from.
        position (int): Position index to extract vectors from.
        id_field (str): Field name used as unique identifier for datapoints.

    Returns:
        dict: For each group, returns a tuple (datapoint_ids, vector_dim)
    """
    file_path = os.path.join(save_dir, first_checkpoint_file)
    with open(file_path, 'r') as fp:
        data = json.load(fp)

    group_info = {}
    for group in groups:
        if group not in data:
            print(f"Group '{group}' not found in checkpoint '{first_checkpoint_file}'. Skipping.")
            continue

        datapoint_ids = []
        vector_dim = None

        for instance in data[group]:
            datapoint_id = instance.get(id_field) or instance.get('input_text')
            if datapoint_id:
                datapoint_ids.append(datapoint_id)
                # Determine vector dimension from the first instance
                if vector_dim is None:
                    hidden_states = instance.get('hidden_states', [])
                    for hidden_state in hidden_states:
                        if hidden_state.get('layer') == layer and hidden_state.get('position') == position:
                            vector = None
                            if layer == 0:
                                vector = hidden_state.get('embedding')
                            else:
                                vector = hidden_state.get('mlp_output') or hidden_state.get('residual')
                            if vector is not None:
                                vector_dim = len(vector)
                            break
        if vector_dim is None:
            print(f"Could not determine vector dimension for group '{group}'. Skipping.")
            continue

        group_info[group] = (datapoint_ids, vector_dim)

    return group_info

def process_checkpoint_file(args):
    """
    Process a single checkpoint file and update trajectories.

    Args:
        args (tuple): Contains (checkpoint_file, group, datapoint_ids, layer, position, id_field)
    """
    (checkpoint_file, group, datapoint_ids_set, layer, position, id_field) = args

    vector_types = ['residual']
    if layer != 0:
        vector_types = ['post_mlp', 'residual']  # Skipping 'post_attention' vectors

    # Initialize a dictionary to store vectors for this checkpoint
    checkpoint_data = {vt: {} for vt in vector_types}

    file_path = os.path.join(save_dir, checkpoint_file)
    with open(file_path, 'r') as fp:
        data = json.load(fp)
        if group in data:
            for instance in data[group]:
                datapoint_id = instance.get(id_field) or instance.get('input_text')
                if datapoint_id in datapoint_ids_set:
                    hidden_states = instance.get('hidden_states', [])
                    for hidden_state in hidden_states:
                        if hidden_state.get('layer') == layer and hidden_state.get('position') == position:
                            if layer == 0:
                                residual = hidden_state.get('embedding')
                                if residual:
                                    vector = np.array(residual, dtype='float32')
                                    checkpoint_data['residual'][datapoint_id] = vector
                            else:
                                # Process 'post_mlp' and 'residual' vectors only
                                post_mlp = hidden_state.get('mlp_output')
                                residual = hidden_state.get('residual')

                                if post_mlp:
                                    vector = np.array(post_mlp, dtype='float32')
                                    checkpoint_data['post_mlp'][datapoint_id] = vector

                                if residual:
                                    vector = np.array(residual, dtype='float32')
                                    checkpoint_data['residual'][datapoint_id] = vector
                            break  # Found the relevant hidden state
        else:
            print(f"Group '{group}' not found in checkpoint '{checkpoint_file}'. Skipping.")

    return (checkpoint_file, checkpoint_data)

def process_group(group_info_tuple):
    """
    Process all checkpoint files for a single group.

    Args:
        group_info_tuple (tuple): Contains (group, checkpoint_files, datapoint_ids, layer, position, id_field, num_steps)
    """
    (group, checkpoint_files, datapoint_ids, layer, position, id_field, num_steps, num_workers) = group_info_tuple

    print(f"Processing group '{group}' with {num_workers} worker processes...")

    datapoint_ids_set = set(datapoint_ids)
    vector_types = ['residual']
    if layer != 0:
        vector_types = ['post_mlp', 'residual']  # Skipping 'post_attention' vectors

    # Initialize in-memory trajectories
    trajectories = {vt: {} for vt in vector_types}
    for vt in vector_types:
        for datapoint_id in datapoint_ids:
            trajectories[vt][datapoint_id] = None  # We'll initialize arrays when we know vector_dim

    # Prepare arguments for processing checkpoint files
    checkpoint_args = [
        (checkpoint_file, group, datapoint_ids_set, layer, position, id_field)
        for checkpoint_file in checkpoint_files
    ]

    # Process checkpoint files in parallel
    with mp.Pool(processes=num_workers) as pool:
        results = list(tqdm(pool.imap_unordered(process_checkpoint_file, checkpoint_args),
                            total=len(checkpoint_args),
                            desc=f"Processing checkpoints for group '{group}'"))

    # Sort results by checkpoint order
    checkpoint_order = {checkpoint_file: idx for idx, checkpoint_file in enumerate(checkpoint_files)}
    results.sort(key=lambda x: checkpoint_order[x[0]])

    # Build trajectories
    for step_idx, (checkpoint_file, checkpoint_data) in enumerate(results):
        for vt in vector_types:
            for datapoint_id, vector in checkpoint_data[vt].items():
                if trajectories[vt][datapoint_id] is None:
                    vector_dim = vector.shape[0]
                    trajectories[vt][datapoint_id] = np.empty((num_steps, vector_dim), dtype='float32')
                trajectories[vt][datapoint_id][step_idx, :] = vector

    # Save trajectories to disk
    save_trajectories(output_dir, group, trajectories)

def save_trajectories(output_dir, group, trajectories):
    """
    Save trajectories to disk using HDF5 format.

    Args:
        output_dir (str): Directory to save the trajectory files.
        group (str): Group name.
        trajectories (dict): Trajectories data to save.
    """
    group_dir = os.path.join(output_dir, group)
    os.makedirs(group_dir, exist_ok=True)

    for vt, data in trajectories.items():
        vt_file = os.path.join(group_dir, f"{vt}.h5")
        with h5py.File(vt_file, 'w') as hf:
            for datapoint_id, trajectory in data.items():
                if trajectory is not None:
                    hf.create_dataset(datapoint_id, data=trajectory)
        print(f"Saved trajectories for vector type '{vt}' in group '{group}' to '{vt_file}'")

def main():
    parser = argparse.ArgumentParser(description="Build trajectory matrices from collected vectors across checkpoints.")
    parser.add_argument("--save_dir", required=True, help="Directory where checkpoint JSON files are stored.")
    parser.add_argument("--output_dir", required=True, help="Directory to save the trajectory matrices.")
    parser.add_argument("--groups", nargs='+', required=True, help="List of group names to process (e.g., train_inferred test_inferred_iid test_inferred_ood).")
    parser.add_argument("--layer", type=int, required=True, help="Layer number to extract vectors from.")
    parser.add_argument("--position", type=int, required=True, help="Position index to extract vectors from.")
    parser.add_argument("--id_field", default="id", help="Field name used as unique identifier for datapoints. Defaults to 'id'. Use 'input_text' if 'id' is not available.")
    parser.add_argument("--num_workers", type=int, default=32, help="Number of parallel workers to use.")
    args = parser.parse_args()

    global save_dir, output_dir  # Needed to access within process_checkpoint_file
    save_dir = args.save_dir
    output_dir = args.output_dir
    groups = args.groups
    layer = args.layer
    position = args.position
    id_field = args.id_field
    num_workers = args.num_workers

    os.makedirs(output_dir, exist_ok=True)

    # Function to extract step number from filename
    def extract_step_number(filename):
        match = re.search(r'\d+', filename)
        return int(match.group()) if match else -1

    # Get list of checkpoint files and sort numerically
    checkpoint_files = [
        f for f in os.listdir(save_dir)
        if f.startswith("composition_") and f.endswith('.json')
    ]
    checkpoint_files.sort(key=lambda f: extract_step_number(f))

    if not checkpoint_files:
        print("No checkpoint data found. Exiting.")
        return

    num_steps = len(checkpoint_files)
    first_checkpoint_file = checkpoint_files[0]

    # Collect datapoint IDs and vector dimensions
    group_info = collect_datapoint_ids_and_vector_dim(save_dir, first_checkpoint_file, groups, layer, position, id_field)

    # Prepare arguments for group processing
    group_tasks = []
    for group in groups:
        if group in group_info:
            datapoint_ids, vector_dim = group_info[group]
            group_tasks.append((group, checkpoint_files, datapoint_ids, layer, position, id_field, num_steps, num_workers))
        else:
            print(f"Group '{group}' information is missing. Skipping.")

    # Process groups sequentially or in parallel
    # Since we are using multiprocessing within groups, we can process groups sequentially to avoid overloading the system
    for group_task in group_tasks:
        process_group(group_task)

if __name__ == "__main__":
    main()