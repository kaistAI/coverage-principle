import os
import json
import argparse
from collections import defaultdict
import numpy as np
from tqdm import tqdm

def extract_step(checkpoint_name):
    """
    Extracts the step number from a checkpoint name.
    Assumes checkpoint naming like 'checkpoint-1000'.
    
    Args:
        checkpoint_name (str): Name of the checkpoint directory.
    
    Returns:
        int: Extracted step number.
    """
    try:
        return int(checkpoint_name.split('-')[-1])
    except ValueError:
        raise ValueError(f"Invalid checkpoint name format: {checkpoint_name}")

def load_checkpoint_data(save_dir, fname_pattern):
    """
    Loads all checkpoint JSON files from the save directory.
    
    Args:
        save_dir (str): Directory containing checkpoint JSON files.
        fname_pattern (str): Filename pattern with a placeholder for step, e.g., 'composition_dedup_{}.json'.
    
    Returns:
        list of tuples: Each tuple contains (step, data_dict).
    """
    checkpoint_files = [
        f for f in os.listdir(save_dir) 
        if f.startswith(fname_pattern.split('{}')[0]) and f.endswith('.json')
    ]
    print(checkpoint_files)
    checkpoints = []
    for f in checkpoint_files:
        # Extract step from filename
        step_part = f.replace(fname_pattern.split('{}')[0], '').replace('.json', '')
        try:
            step = int(step_part)
        except ValueError:
            print(f"Skipping file with invalid step number: {f}")
            continue
        
        file_path = os.path.join(save_dir, f)
        with open(file_path, 'r') as fp:
            data = json.load(fp)
            checkpoints.append((step, data))
    
    # Sort checkpoints by step
    checkpoints.sort(key=lambda x: x[0])
    return checkpoints

def build_trajectories(checkpoints, groups, output_dir, layer, position, id_field):
    """
    Builds trajectory matrices for each datapoint across all checkpoints.
    
    Args:
        checkpoints (list of tuples): Each tuple contains (step, data_dict).
        groups (list of str): List of group names to process.
        output_dir (str): Directory to save the trajectory matrices.
        layer (int): Layer number to extract vectors from.
        position (int): Position index to extract vectors from.
        id_field (str): Field name used as unique identifier for datapoints.
    """
    # Initialize a nested dictionary to hold trajectories
    # trajectories[group][datapoint_id]['post_attention'] = list of vectors
    # trajectories[group][datapoint_id]['post_mlp'] = list of vectors
    # trajectories[group][datapoint_id]['residual'] = list of vectors
    trajectories = {group: defaultdict(lambda: {'post_attention': [], 'post_mlp': [], 'residual': []}) for group in groups}
    
    total_checkpoints = len(checkpoints)
    
    for step, data in tqdm(checkpoints, desc="Processing Checkpoints"):
        for group in groups:
            if group not in data:
                print(f"Group '{group}' not found in checkpoint step {step}. Skipping.")
                continue
            for instance in data[group]:
                # Assuming each instance has a unique identifier. Replace 'id' with the actual key if different.
                datapoint_id = instance.get(id_field) or instance.get('input_text')  # Replace as needed
                if not datapoint_id:
                    print(f"Datapoint in group '{group}' missing '{id_field}' or 'input_text'. Skipping.")
                    continue
                
                hidden_states = instance.get('hidden_states', [])
                post_attention = None
                post_mlp = None
                residual = None
                for hidden_state in hidden_states:
                    if hidden_state.get('layer') == layer and hidden_state.get('position') == position:
                        if layer == 0:
                            # For layer 0, only 'embedding' is relevant (treated as 'residual')
                            post_attention = None
                            post_mlp = None
                            residual = hidden_state.get('embedding')
                        else:
                            post_attention = hidden_state.get('attn_output')
                            post_mlp = hidden_state.get('mlp_output')
                            residual = hidden_state.get('residual')
                        break  # Found the relevant hidden state

                # Append vectors to respective trajectory lists if they exist
                if layer == 0:
                    if residual is not None:
                        trajectories[group][datapoint_id]['residual'].append(residual)
                    else:
                        print(f"No residual vector found for datapoint '{datapoint_id}' in group '{group}' at layer {layer}, position {position} in step {step}.")
                else:
                    if post_attention is not None:
                        trajectories[group][datapoint_id]['post_attention'].append(post_attention)
                    else:
                        print(f"No post_attention vector found for datapoint '{datapoint_id}' in group '{group}' at layer {layer}, position {position} in step {step}.")
                    
                    if post_mlp is not None:
                        trajectories[group][datapoint_id]['post_mlp'].append(post_mlp)
                    else:
                        print(f"No post_mlp vector found for datapoint '{datapoint_id}' in group '{group}' at layer {layer}, position {position} in step {step}.")
                    
                    if residual is not None:
                        trajectories[group][datapoint_id]['residual'].append(residual)
                    else:
                        print(f"No residual vector found for datapoint '{datapoint_id}' in group '{group}' at layer {layer}, position {position} in step {step}.")

    # Convert lists to NumPy arrays and save separately for each vector type
    for group in groups:
        group_dir = os.path.join(output_dir, group)
        # Create sub-directories for each vector type
        post_attention_dir = os.path.join(group_dir, 'post_attention')
        post_mlp_dir = os.path.join(group_dir, 'post_mlp')
        residual_dir = os.path.join(group_dir, 'residual')
        
        os.makedirs(post_attention_dir, exist_ok=True)
        os.makedirs(post_mlp_dir, exist_ok=True)
        os.makedirs(residual_dir, exist_ok=True)
        
        print(f"Saving trajectories for group '{group}'...")
        for datapoint_id, vectors_dict in tqdm(trajectories[group].items(), desc=f"Saving group '{group}'"):
            # Check if all vectors have the expected length
            expected_length = total_checkpoints
            incomplete = False
            for vec_type in ['post_attention', 'post_mlp', 'residual']:
                if layer == 0 and vec_type != 'residual':
                    # For layer 0, only 'residual' is relevant
                    continue
                if len(vectors_dict[vec_type]) != expected_length:
                    print(f"Datapoint '{datapoint_id}' in group '{group}' has {len(vectors_dict[vec_type])} '{vec_type}' vectors instead of {expected_length}. Skipping.")
                    incomplete = True
                    break
            if incomplete:
                continue
            
            # Save each type of trajectory separately
            if layer == 0:
                # For layer 0, only 'residual' trajectories exist
                trajectory_residual = np.array(vectors_dict['residual'])  # Shape: (num_checkpoints, vector_dim)
                save_path_residual = os.path.join(residual_dir, f"{datapoint_id}.npy")
                np.save(save_path_residual, trajectory_residual)
            else:
                # For other layers, save 'post_attention', 'post_mlp', and 'residual' trajectories
                trajectory_post_attention = np.array(vectors_dict['post_attention'])  # Shape: (num_checkpoints, vector_dim)
                trajectory_post_mlp = np.array(vectors_dict['post_mlp'])              # Shape: (num_checkpoints, vector_dim)
                trajectory_residual = np.array(vectors_dict['residual'])              # Shape: (num_checkpoints, vector_dim)
                
                save_path_post_attention = os.path.join(post_attention_dir, f"{datapoint_id}.npy")
                save_path_post_mlp = os.path.join(post_mlp_dir, f"{datapoint_id}.npy")
                save_path_residual = os.path.join(residual_dir, f"{datapoint_id}.npy")
                
                np.save(save_path_post_attention, trajectory_post_attention)
                np.save(save_path_post_mlp, trajectory_post_mlp)
                np.save(save_path_residual, trajectory_residual)
    
def main():
    parser = argparse.ArgumentParser(description="Build trajectory matrices from collected vectors across checkpoints.")
    parser.add_argument("--save_dir", required=True, help="Directory where 'composition_dedup_<step>.json' files are stored.")
    parser.add_argument("--output_dir", required=True, help="Directory to save the trajectory matrices.")
    parser.add_argument("--groups", nargs='+', required=True, help="List of group names to process (e.g., train_inferred test_inferred_iid test_inferred_ood).")
    parser.add_argument("--layer", type=int, required=True, help="Layer number to extract vectors from.")
    parser.add_argument("--position", type=int, required=True, help="Position index to extract vectors from.")
    parser.add_argument("--id_field", default="id", help="Field name used as unique identifier for datapoints. Defaults to 'id'. Use 'input_text' if 'id' is not available.")
    args = parser.parse_args()
    
    save_dir = args.save_dir
    output_dir = args.output_dir
    groups = args.groups
    layer = args.layer
    position = args.position
    id_field = args.id_field
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Load all checkpoint data
    checkpoints = load_checkpoint_data(save_dir, fname_pattern="composition_{}.json")
    
    if not checkpoints:
        print("No checkpoint data found. Exiting.")
        return
    
    # Build trajectories
    build_trajectories(checkpoints, groups, output_dir, layer, position, id_field)
    
if __name__ == "__main__":
    main()