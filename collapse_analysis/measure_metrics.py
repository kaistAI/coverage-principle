import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
import numpy as np
from scipy.spatial.distance import cosine
from collections import defaultdict
import argparse
from tqdm import tqdm
import re

# -------------- NEW IMPORTS FOR GPU ACCELERATION --------------
import torch
import torch.nn.functional as F

def plot_similarity_distribution(similarities, title, save_path, color='blue', kde=True):
    """Create and save a distribution plot for similarities."""
    plt.figure(figsize=(10, 6))
    sns.histplot(similarities, kde=kde, color=color)
    plt.title(title)
    plt.xlabel('Cosine Similarity')
    plt.ylabel('Count')
    plt.savefig(save_path)
    plt.close()

def plot_comparison_distributions(dist1, dist2, labels, title, save_path):
    """Create and save an overlapping distribution plot for two sets of similarities."""
    plt.figure(figsize=(10, 6))
    sns.histplot(dist1, kde=True, color='blue', alpha=0.5, label=labels[0])
    sns.histplot(dist2, kde=True, color='red', alpha=0.5, label=labels[1])
    plt.title(title)
    plt.xlabel('Cosine Similarity')
    plt.ylabel('Count')
    plt.legend()
    plt.savefig(save_path)
    plt.close()

def load_data(file_path):
    print(f"\nLoading data from: {file_path}")
    with open(file_path, 'r') as f:
        data = json.load(f)
    print(f"Successfully loaded. Number of top-level keys in JSON: {len(data)}")
    return data

# Original CPU-based cosine_similarity using scipy:
def cosine_similarity(v1, v2):
    """
    Matches: 1 - distance.cosine(...)
    which is dot(v1,v2)/(||v1||*||v2||).
    """
    return 1 - cosine(v1, v2)

def process_vectors(data, layer):
    """
    Convert the JSON data into grouped_vectors and all_vectors lists.
    Using tqdm for progress logging as we iterate over data.
    """
    print(f"\nProcessing vectors for layer {layer}.")
    grouped_vectors = defaultdict(list)
    all_vectors = []

    # If data is large, you might want to nest the tqdm for 'instances' too,
    # but here we apply it to the top-level iteration.
    for target, instances in tqdm(data.items(), desc="Grouping vectors by target"):
        for instance in instances:
            # Example path: instance['hidden_states'][0]['post_mlp']
            vector = instance['hidden_states'][0]['post_mlp']
            if vector is not None:
                grouped_vectors[target].append(vector)
                all_vectors.append(vector)
    
    print(f"Finished processing. Number of groups: {len(grouped_vectors)}. "
          f"Total vectors collected: {len(all_vectors)}")
    return grouped_vectors, all_vectors

def calculate_within_group_similarity_gpu(group):
    """
    Compute the mean and distribution of within-group cosine similarities
    using GPU-based matrix multiplication (with no Python nested loop).
    """
    k = len(group)
    if k <= 1:
        return None, []

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    X = torch.tensor(group, dtype=torch.float32, device=device)

    # Normalize (for cosine = dot product of normalized vectors)
    X = F.normalize(X, p=2, dim=1)  # shape: (k, D)

    # Compute the pairwise similarity matrix
    sim_matrix = torch.matmul(X, X.t())  # shape: (k, k)

    # Extract the upper-triangular part (i < j), excluding diagonal
    indices = torch.triu_indices(k, k, offset=1)
    similarities_upper = sim_matrix[indices[0], indices[1]]

    similarities_list = similarities_upper.cpu().numpy().tolist()

    return float(np.mean(similarities_list)), similarities_list

def calculate_between_group_similarity_gpu(group_means):
    """
    Compute mean pairwise cosine similarity among multiple group means,
    using GPU-based matrix multiplication (no Python nested loop).
    """
    M = len(group_means)
    if M < 2:
        return None, []

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    X = torch.tensor(group_means, dtype=torch.float32, device=device)

    # Normalize each mean so dot(X[i], X[j]) = cosine similarity
    X = F.normalize(X, p=2, dim=1)

    # Compute the pairwise similarity matrix
    sim_matrix = torch.matmul(X, X.t())  # shape: (M, M)

    # Extract the upper-triangular part (i < j), excluding diagonal
    indices = torch.triu_indices(M, M, offset=1)
    similarities_upper = sim_matrix[indices[0], indices[1]]

    similarities_list = similarities_upper.cpu().numpy().tolist()

    return float(np.mean(similarities_list)), similarities_list

def compute_all_similarities_gpu(all_vectors):
    """
    Computes all pairwise cosine similarities on GPU using
    a single matrix multiplication of normalized vectors.

    Returns a Python list of all pairwise similarities (i<j).
    """
    if len(all_vectors) < 2:
        return []
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    X = torch.tensor(all_vectors, dtype=torch.float32, device=device)
    X = F.normalize(X, p=2, dim=1)  # shape: (N, D)

    sim_matrix = torch.matmul(X, X.t())  # shape: (N, N)

    N = sim_matrix.shape[0]
    # Get indices for upper triangular portion (above the diagonal)
    indices = torch.triu_indices(N, N, offset=1)  # offset=1 to skip diagonal

    similarities_upper = sim_matrix[indices[0], indices[1]]

    return similarities_upper.cpu().numpy().tolist()

def calculate_metrics(grouped_vectors, all_vectors, ood=False):
    """
    If ood=True (or nonsense=True), we calculate the all-vector similarity 
    distribution using compute_all_similarities_gpu(...).
    Otherwise, we skip that step.
    """
    # Collect within-group similarities
    within_group_similarities_all = []
    between_group_similarities = []
    group_mean_vectors = []

    # We can track progress with tqdm if we expect many groups.
    # But typically, grouping was done in process_vectors.
    # If you have many groups, you can wrap the loop below in tqdm as well.
    for bridge_entity, group in tqdm(grouped_vectors.items()):
        if len(group) > 1:
            _, similarities = calculate_within_group_similarity_gpu(group)
            if similarities:
                within_group_similarities_all.extend(similarities)
        # Always compute and store the mean
        group_mean_vectors.append(np.mean(group, axis=0))

    # Between-group similarities
    _, between_similarities = calculate_between_group_similarity_gpu(group_mean_vectors)
    between_group_similarities.extend(between_similarities)

    # compute the all-vector similarities
    all_similarities = compute_all_similarities_gpu(all_vectors)
    # If OOD (or nonsense), compute the all-vector similarities
    # all_similarities = None
    # if ood and len(all_vectors) > 1:
    #     print("Computing all-vector similarities (OOD or Nonsense)...")
    #     all_similarities = compute_all_similarities_gpu(all_vectors)

    return (np.mean(within_group_similarities_all) if within_group_similarities_all else 0,
            within_group_similarities_all,
            np.mean(between_group_similarities) if between_group_similarities else 0,
            between_group_similarities,
            group_mean_vectors,
            all_similarities)

def main():
    parser = argparse.ArgumentParser(description='Calculate and visualize similarity metrics for ID, OOD, and Nonsense vectors')
    parser.add_argument('--id_train_file', required=True, help='Path to the ID Train vector file')
    parser.add_argument('--id_test_file', required=True, help='Path to the ID Test vector file')
    parser.add_argument('--ood_file', required=True, help='Path to the OOD vector file')
    parser.add_argument('--nonsense_file', required=True, help='Path to the Nonsense vector file')
    # parser.add_argument('--layer', type=int, default=7, help='Layer to analyze (default: 7)')
    parser.add_argument('--output_dir', required=True, help='Directory to save the plots and results')
    args = parser.parse_args()
    
    # Argument Error Check
    assert re.findall(r"\(\d+,\s*\d+\)", args.id_train_file)[0] == re.findall(r"\(\d+,\s*\d+\)", args.output_dir)[0]

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    matches = re.findall(r"\(\d+,\s*\d+\)", args.output_dir)
    assert len(matches) == 1
    target_layer = matches[0].strip(")(").split(",")[0]

    # Prepare a file to save numerical results
    results_file = output_dir / f"similarity_metrics_layer{target_layer}.txt"

    # Load data
    id_train_data = load_data(args.id_train_file)
    id_test_data = load_data(args.id_test_file)
    ood_data = load_data(args.ood_file)
    nonsense_data = load_data(args.nonsense_file)

    # Process into grouped vectors
    id_train_vectors, id_train_all_vectors = process_vectors(id_train_data, target_layer)
    id_test_vectors, id_test_all_vectors = process_vectors(id_test_data, target_layer)
    ood_vectors, ood_all_vectors = process_vectors(ood_data, target_layer)
    nonsense_vectors, nonsense_all_vectors = process_vectors(nonsense_data, target_layer)

    print(f"\nAnalyzing layer {target_layer}...")

    # =============== ID Train Metrics ===============
    print("\nComputing ID Train metrics...")
    (id_train_within_similarity, id_train_within_similarities,
     id_train_between_similarity, id_train_between_similarities,
     id_train_group_means, id_train_all_similarities) = calculate_metrics(id_train_vectors, id_train_all_vectors, ood=False)
    
    print(f"  ID within-group similarity (mean): {id_train_within_similarity:.4f}")
    print(f"  ID between-group similarity (mean): {id_train_between_similarity:.4f}")
    
    # =============== ID Test Metrics ===============
    print("\nComputing ID Test metrics...")
    (id_test_within_similarity, id_test_within_similarities,
     id_test_between_similarity, id_test_between_similarities,
     id_test_group_means, id_test_all_similarities) = calculate_metrics(id_test_vectors, id_test_all_vectors, ood=False)
    
    print(f"  ID within-group similarity (mean): {id_test_within_similarity:.4f}")
    print(f"  ID between-group similarity (mean): {id_test_between_similarity:.4f}")

    # =============== OOD Metrics ===============
    print("\nComputing OOD metrics...")
    (ood_within_similarity, ood_within_similarities,
     ood_between_similarity, ood_between_similarities,
     ood_group_means, ood_all_similarities) = calculate_metrics(ood_vectors, ood_all_vectors, ood=True)
    
    print(f"  OOD within-group similarity (mean): {ood_within_similarity:.4f}")
    print(f"  OOD between-group similarity (mean): {ood_between_similarity:.4f}")
    if ood_all_similarities:
        print(f"  OOD all-vector similarity (mean): {np.mean(ood_all_similarities):.4f}")
    else:
        print(f"  OOD all-vector similarity: None (only one OOD vector?)")

    # =============== Nonsense Metrics ===============
    print("\nComputing Nonsense metrics...")
    (nonsense_within_similarity, nonsense_within_similarities,
     nonsense_between_similarity, nonsense_between_similarities,
     nonsense_group_means, nonsense_all_similarities) = calculate_metrics(nonsense_vectors, nonsense_all_vectors, ood=True)
    
    print(f"  Nonsense within-group similarity (mean): {nonsense_within_similarity:.4f}")
    print(f"  Nonsense between-group similarity (mean): {nonsense_between_similarity:.4f}")
    if nonsense_all_similarities:
        print(f"  Nonsense all-vector similarity (mean): {np.mean(nonsense_all_similarities):.4f}")
    else:
        print(f"  Nonsense all-vector similarity: None (only one nonsense vector?)")

    # =============== Cross-Category Comparisons ===============
    print("\nComputing cross-category similarities...")
    
    # ID Train group mean vs ID Test group mean for same bridge entity
    id_train_group_mean_dict = {}
    for bridge_entity, vector_list in id_train_vectors.items():
        id_train_group_mean_dict[bridge_entity] = np.mean(vector_list, axis=0)
    id_test_group_mean_dict = {}
    for bridge_entity, vector_list in id_test_vectors.items():
        id_test_group_mean_dict[bridge_entity] = np.mean(vector_list, axis=0)
    intersection_bridge_entity_set = set(id_train_group_mean_dict.keys()) & set(id_test_group_mean_dict.keys())
    id_train_test_similarity_each_group = [cosine_similarity(id_train_group_mean_dict[bridge_entity], id_test_group_mean_dict[bridge_entity]) for bridge_entity in intersection_bridge_entity_set]
    avg_id_train_test_similarity_each_group = np.mean(id_train_test_similarity_each_group)
    print(f"  Average similarity (ID Train group mean vs ID Test group mean for same group): {avg_id_train_test_similarity_each_group:.4f}")
        
    # Global OOD mean vs ID Train group means
    ood_global_mean = np.mean(np.vstack(ood_group_means), axis=0)
    id_train_ood_similarities = [cosine_similarity(ood_global_mean, id_mean) for id_mean in id_train_group_means]
    avg_id_train_ood_similarity = np.mean(id_train_ood_similarities)
    print(f"  Average similarity (OOD global mean vs ID Train group means): {avg_id_train_ood_similarity:.4f}")
    
    # Global OOD mean vs ID Test group means
    id_test_ood_similarities = [cosine_similarity(ood_global_mean, id_mean) for id_mean in id_test_group_means]
    avg_id_test_ood_similarity = np.mean(id_test_ood_similarities)
    print(f"  Average similarity (OOD global mean vs ID Test group means): {avg_id_test_ood_similarity:.4f}")

    # Global Nonsense mean vs ID Train group means
    nonsense_global_mean = np.mean(np.vstack(nonsense_group_means), axis=0)
    id_train_nonsense_similarities = [cosine_similarity(nonsense_global_mean, id_mean) for id_mean in id_train_group_means]
    avg_id_train_nonsense_similarity = np.mean(id_train_nonsense_similarities)
    print(f"  Average similarity (Nonsense global mean vs ID Train group means): {avg_id_train_nonsense_similarity:.4f}")
    
    # Global Nonsense mean vs ID Test group means
    id_test_nonsense_similarities = [cosine_similarity(nonsense_global_mean, id_mean) for id_mean in id_test_group_means]
    avg_id_test_nonsense_similarity = np.mean(id_test_nonsense_similarities)
    print(f"  Average similarity (Nonsense global mean vs ID Test group means): {avg_id_test_nonsense_similarity:.4f}")

    # Global Nonsense mean vs OOD group means
    ood_nonsense_similarities = [cosine_similarity(nonsense_global_mean, ood_mean) for ood_mean in ood_group_means]
    avg_ood_nonsense_similarity = np.mean(ood_nonsense_similarities)
    print(f"  Average similarity (Nonsense global mean vs OOD group means): {avg_ood_nonsense_similarity:.4f}")

    # =============== Write Numerical Results to File ===============
    print(f"\nWriting results to {results_file}...")
    with open(results_file, 'w') as f:
        f.write(f"Analyzing layer {target_layer}\n\n")

        f.write("=== ID Train Metrics ===\n")
        f.write(f"ID-Train within-group similarity (mean): {id_train_within_similarity:.4f}\n")
        f.write(f"ID-Train between-group similarity (mean): {id_train_between_similarity:.4f}\n")
        if id_train_all_similarities:
            f.write(f"ID-Train all-vector similarity (mean): {np.mean(id_train_all_similarities):.4f}\n\n")
        else:
            f.write(f"ID-Train all-vector similarity (mean): None\n\n")
        
        f.write("=== ID Test Metrics ===\n")
        f.write(f"ID-Test within-group similarity (mean): {id_test_within_similarity:.4f}\n")
        f.write(f"ID-Test between-group similarity (mean): {id_test_between_similarity:.4f}\n")
        if id_test_all_similarities:
            f.write(f"ID-Test all-vector similarity (mean): {np.mean(id_test_all_similarities):.4f}\n\n")
        else:
            f.write(f"ID-Test all-vector similarity (mean): None\n\n")

        f.write("=== OOD Metrics ===\n")
        f.write(f"OOD within-group similarity (mean): {ood_within_similarity:.4f}\n")
        f.write(f"OOD between-group similarity (mean): {ood_between_similarity:.4f}\n")
        if ood_all_similarities:
            f.write(f"OOD all-vector similarity (mean): {np.mean(ood_all_similarities):.4f}\n\n")
        else:
            f.write("OOD all-vector similarity (mean): None\n\n")

        f.write("=== Nonsense Metrics ===\n")
        f.write(f"Nonsense within-group similarity (mean): {nonsense_within_similarity:.4f}\n")
        f.write(f"Nonsense between-group similarity (mean): {nonsense_between_similarity:.4f}\n")
        if nonsense_all_similarities:
            f.write(f"Nonsense all-vector similarity (mean): {np.mean(nonsense_all_similarities):.4f}\n\n")
        else:
            f.write("Nonsense all-vector similarity (mean): None\n\n")
            
        f.write("=== ID_Train-ID_Test Similarities ===\n")
        f.write(f"Average similarity (ID Train group mean vs ID Test group mean for same group): {avg_id_train_test_similarity_each_group:.4f}\n\n")

        f.write("=== ID_Train-OOD Similarities ===\n")
        f.write(f"Average similarity (OOD global mean vs ID Train group means): {avg_id_train_ood_similarity:.4f}\n\n")
        
        f.write("=== ID_Test-OOD Similarities ===\n")
        f.write(f"Average similarity (OOD global mean vs ID Test group means): {avg_id_test_ood_similarity:.4f}\n\n")

        f.write("=== ID_Train-Nonsense Similarities ===\n")
        f.write(f"Average similarity (Nonsense global mean vs ID Train group means): {avg_id_train_nonsense_similarity:.4f}\n\n")
        
        f.write("=== ID_Test-Nonsense Similarities ===\n")
        f.write(f"Average similarity (Nonsense global mean vs ID Test group means): {avg_id_test_nonsense_similarity:.4f}\n\n")

        f.write("=== OOD-Nonsense Similarities ===\n")
        f.write(f"Average similarity (Nonsense global mean vs OOD group means): {avg_ood_nonsense_similarity:.4f}\n\n")

    # =============== Create Plots ===============
    print("\nCreating plots...")

    # ---------------- ID Train PLOTS ----------------
    print("  - Plotting ID Train distributions...")
    plot_comparison_distributions(
        id_train_within_similarities,
        id_train_between_similarities,
        ['Within-group', 'Between-group'],
        f'ID Train Similarity Distributions (Layer {target_layer})',
        output_dir / f'id_train_similarities_layer{target_layer}.png'
    )
    
    if id_train_all_similarities and len(id_train_all_similarities) > 0:
        plot_similarity_distribution(
            id_train_all_similarities,
            f'ID Train All-Vector Similarity Distribution (Layer {target_layer})',
            output_dir / f'id_train_all_similarities_layer{target_layer}.png',
            color='green'
        )
    
    # ---------------- ID Test PLOTS ----------------
    print("  - Plotting ID Test distributions...")
    plot_comparison_distributions(
        id_test_within_similarities,
        id_test_between_similarities,
        ['Within-group', 'Between-group'],
        f'ID Train Similarity Distributions (Layer {target_layer})',
        output_dir / f'id_test_similarities_layer{target_layer}.png'
    )
    
    if id_test_all_similarities and len(id_test_all_similarities) > 0:
        plot_similarity_distribution(
            id_test_all_similarities,
            f'ID Test All-Vector Similarity Distribution (Layer {target_layer})',
            output_dir / f'id_test_all_similarities_layer{target_layer}.png',
            color='green'
        )

    # ---------------- OOD PLOTS ----------------
    print("  - Plotting OOD distributions...")
    plot_comparison_distributions(
        ood_within_similarities,
        ood_between_similarities,
        ['Within-group', 'Between-group'],
        f'OOD Similarity Distributions (Layer {target_layer})',
        output_dir / f'ood_similarities_layer{target_layer}.png'
    )

    if ood_all_similarities and len(ood_all_similarities) > 0:
        plot_similarity_distribution(
            ood_all_similarities,
            f'OOD All-Vector Similarity Distribution (Layer {target_layer})',
            output_dir / f'ood_all_similarities_layer{target_layer}.png',
            color='green'
        )

    
    
    plot_similarity_distribution(
        id_train_ood_similarities,
        f'ID_Train-OOD Mean Similarity Distribution (Layer {target_layer})',
        output_dir / f'id_train_ood_mean_similarities_layer{target_layer}.png',
        color='purple'
    )
    
    plot_similarity_distribution(
        id_test_ood_similarities,
        f'ID_Test-OOD Mean Similarity Distribution (Layer {target_layer})',
        output_dir / f'id_test_ood_mean_similarities_layer{target_layer}.png',
        color='purple'
    )

    plot_comparison_distributions(
        id_train_within_similarities,
        ood_within_similarities,
        ['ID_Train Within-group', 'OOD Within-group'],
        f'ID_Train vs OOD Within-Group Similarities (Layer {target_layer})',
        output_dir / f'id_train_vs_ood_within_group_layer{target_layer}.png'
    )
    
    plot_comparison_distributions(
        id_test_within_similarities,
        ood_within_similarities,
        ['ID Test Within-group', 'OOD Within-group'],
        f'ID_Test vs OOD Within-Group Similarities (Layer {target_layer})',
        output_dir / f'id_test_vs_ood_within_group_layer{target_layer}.png'
    )

    # ---------------- NONSENSE PLOTS ----------------
    print("  - Plotting Nonsense distributions...")
    plot_comparison_distributions(
        nonsense_within_similarities,
        nonsense_between_similarities,
        ['Within-group', 'Between-group'],
        f'Nonsense Similarity Distributions (Layer {target_layer})',
        output_dir / f'nonsense_similarities_layer{target_layer}.png'
    )

    if nonsense_all_similarities and len(nonsense_all_similarities) > 0:
        plot_similarity_distribution(
            nonsense_all_similarities,
            f'Nonsense All-Vector Similarity Distribution (Layer {target_layer})',
            output_dir / f'nonsense_all_similarities_layer{target_layer}.png',
            color='brown'
        )

    plot_similarity_distribution(
        id_train_nonsense_similarities,
        f'ID_Train-Nonsense Mean Similarity Distribution (Layer {target_layer})',
        output_dir / f'id_train_nonsense_mean_similarities_layer{target_layer}.png',
        color='orange'
    )
    
    plot_similarity_distribution(
        id_test_nonsense_similarities,
        f'ID_Test-Nonsense Mean Similarity Distribution (Layer {target_layer})',
        output_dir / f'id_test_nonsense_mean_similarities_layer{target_layer}.png',
        color='orange'
    )

    plot_similarity_distribution(
        ood_nonsense_similarities,
        f'OOD-Nonsense Mean Similarity Distribution (Layer {target_layer})',
        output_dir / f'ood_nonsense_mean_similarities_layer{target_layer}.png',
        color='teal'
    )

    plot_comparison_distributions(
        id_train_within_similarities,
        nonsense_within_similarities,
        ['ID_Train Within-group', 'Nonsense Within-group'],
        f'ID_Train vs Nonsense Within-Group Similarities (Layer {target_layer})',
        output_dir / f'id_train_vs_nonsense_within_group_layer{target_layer}.png'
    )
    
    plot_comparison_distributions(
        id_test_within_similarities,
        nonsense_within_similarities,
        ['ID_Test Within-group', 'Nonsense Within-group'],
        f'ID_Test vs Nonsense Within-Group Similarities (Layer {target_layer})',
        output_dir / f'id_test_vs_nonsense_within_group_layer{target_layer}.png'
    )

    plot_comparison_distributions(
        ood_within_similarities,
        nonsense_within_similarities,
        ['OOD Within-group', 'Nonsense Within-group'],
        f'OOD vs Nonsense Within-Group Similarities (Layer {target_layer})',
        output_dir / f'ood_vs_nonsense_within_group_layer{target_layer}.png'
    )

    print("Finished all computations and plots!")

if __name__ == "__main__":
    main()