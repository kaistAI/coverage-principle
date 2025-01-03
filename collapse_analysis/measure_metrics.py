import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
import numpy as np
from scipy.spatial.distance import cosine
from collections import defaultdict
import argparse
from tqdm import tqdm

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
    group_means = []

    # We can track progress with tqdm if we expect many groups.
    # But typically, grouping was done in process_vectors.
    # If you have many groups, you can wrap the loop below in tqdm as well.
    for group_key, group in tqdm(grouped_vectors.items()):
        if len(group) > 1:
            _, similarities = calculate_within_group_similarity_gpu(group)
            if similarities:
                within_group_similarities_all.extend(similarities)
        # Always compute and store the mean
        group_means.append(np.mean(group, axis=0))

    # Between-group similarities
    _, between_similarities = calculate_between_group_similarity_gpu(group_means)
    between_group_similarities.extend(between_similarities)

    # If OOD (or nonsense), compute the all-vector similarities
    all_similarities = None
    if ood and len(all_vectors) > 1:
        print("Computing all-vector similarities (OOD or Nonsense)...")
        all_similarities = compute_all_similarities_gpu(all_vectors)

    return (np.mean(within_group_similarities_all) if within_group_similarities_all else 0,
            within_group_similarities_all,
            np.mean(between_group_similarities) if between_group_similarities else 0,
            between_group_similarities,
            group_means,
            all_similarities)

def main():
    parser = argparse.ArgumentParser(description='Calculate and visualize similarity metrics for ID, OOD, and Nonsense vectors')
    parser.add_argument('--id_file', required=True, help='Path to the ID vector file')
    parser.add_argument('--ood_file', required=True, help='Path to the OOD vector file')
    parser.add_argument('--nonsense_file', required=True, help='Path to the Nonsense vector file')
    parser.add_argument('--layer', type=int, default=7, help='Layer to analyze (default: 7)')
    parser.add_argument('--output_dir', required=True, help='Directory to save the plots and results')
    args = parser.parse_args()

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Prepare a file to save numerical results
    results_file = output_dir / f"similarity_metrics_layer{args.layer}.txt"

    # Load data
    id_data = load_data(args.id_file)
    ood_data = load_data(args.ood_file)
    nonsense_data = load_data(args.nonsense_file)

    # Process into grouped vectors
    id_vectors, id_all_vectors = process_vectors(id_data, args.layer)
    ood_vectors, ood_all_vectors = process_vectors(ood_data, args.layer)
    nonsense_vectors, nonsense_all_vectors = process_vectors(nonsense_data, args.layer)

    print(f"\nAnalyzing layer {args.layer}...")

    # =============== ID Metrics ===============
    print("\nComputing ID metrics...")
    (id_within_similarity, id_within_similarities,
     id_between_similarity, id_between_similarities,
     id_group_means, _) = calculate_metrics(id_vectors, id_all_vectors, ood=False)
    
    print(f"  ID within-group similarity (mean): {id_within_similarity:.4f}")
    print(f"  ID between-group similarity (mean): {id_between_similarity:.4f}")

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

    # Global OOD mean vs ID group means
    ood_global_mean = np.mean(np.vstack(ood_group_means), axis=0)
    id_ood_similarities = [cosine_similarity(ood_global_mean, id_mean) for id_mean in id_group_means]
    avg_id_ood_similarity = np.mean(id_ood_similarities)
    print(f"  Average similarity (OOD global mean vs ID group means): {avg_id_ood_similarity:.4f}")

    # Global Nonsense mean vs ID group means
    nonsense_global_mean = np.mean(np.vstack(nonsense_group_means), axis=0)
    id_nonsense_similarities = [cosine_similarity(nonsense_global_mean, id_mean) for id_mean in id_group_means]
    avg_id_nonsense_similarity = np.mean(id_nonsense_similarities)
    print(f"  Average similarity (Nonsense global mean vs ID group means): {avg_id_nonsense_similarity:.4f}")

    # Global Nonsense mean vs OOD group means
    ood_nonsense_similarities = [cosine_similarity(nonsense_global_mean, ood_mean) for ood_mean in ood_group_means]
    avg_ood_nonsense_similarity = np.mean(ood_nonsense_similarities)
    print(f"  Average similarity (Nonsense global mean vs OOD group means): {avg_ood_nonsense_similarity:.4f}")

    # =============== Write Numerical Results to File ===============
    print(f"\nWriting results to {results_file}...")
    with open(results_file, 'w') as f:
        f.write(f"Analyzing layer {args.layer}\n\n")

        f.write("=== ID Metrics ===\n")
        f.write(f"ID within-group similarity (mean): {id_within_similarity:.4f}\n")
        f.write(f"ID between-group similarity (mean): {id_between_similarity:.4f}\n\n")

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

        f.write("=== ID-OOD Similarities ===\n")
        f.write(f"Average similarity (OOD global mean vs ID group means): {avg_id_ood_similarity:.4f}\n\n")

        f.write("=== ID-Nonsense Similarities ===\n")
        f.write(f"Average similarity (Nonsense global mean vs ID group means): {avg_id_nonsense_similarity:.4f}\n\n")

        f.write("=== OOD-Nonsense Similarities ===\n")
        f.write(f"Average similarity (Nonsense global mean vs OOD group means): {avg_ood_nonsense_similarity:.4f}\n\n")

    # =============== Create Plots ===============
    print("\nCreating plots...")

    # ---------------- ID PLOTS ----------------
    print("  - Plotting ID distributions...")
    plot_comparison_distributions(
        id_within_similarities,
        id_between_similarities,
        ['Within-group', 'Between-group'],
        f'ID Similarity Distributions (Layer {args.layer})',
        output_dir / f'id_similarities_layer{args.layer}.png'
    )

    # ---------------- OOD PLOTS ----------------
    print("  - Plotting OOD distributions...")
    plot_comparison_distributions(
        ood_within_similarities,
        ood_between_similarities,
        ['Within-group', 'Between-group'],
        f'OOD Similarity Distributions (Layer {args.layer})',
        output_dir / f'ood_similarities_layer{args.layer}.png'
    )

    if ood_all_similarities and len(ood_all_similarities) > 0:
        plot_similarity_distribution(
            ood_all_similarities,
            f'OOD All-Vector Similarity Distribution (Layer {args.layer})',
            output_dir / f'ood_all_similarities_layer{args.layer}.png',
            color='green'
        )

    plot_similarity_distribution(
        id_ood_similarities,
        f'ID-OOD Mean Similarity Distribution (Layer {args.layer})',
        output_dir / f'id_ood_mean_similarities_layer{args.layer}.png',
        color='purple'
    )

    plot_comparison_distributions(
        id_within_similarities,
        ood_within_similarities,
        ['ID Within-group', 'OOD Within-group'],
        f'ID vs OOD Within-Group Similarities (Layer {args.layer})',
        output_dir / f'id_vs_ood_within_group_layer{args.layer}.png'
    )

    # ---------------- NONSENSE PLOTS ----------------
    print("  - Plotting Nonsense distributions...")
    plot_comparison_distributions(
        nonsense_within_similarities,
        nonsense_between_similarities,
        ['Within-group', 'Between-group'],
        f'Nonsense Similarity Distributions (Layer {args.layer})',
        output_dir / f'nonsense_similarities_layer{args.layer}.png'
    )

    if nonsense_all_similarities and len(nonsense_all_similarities) > 0:
        plot_similarity_distribution(
            nonsense_all_similarities,
            f'Nonsense All-Vector Similarity Distribution (Layer {args.layer})',
            output_dir / f'nonsense_all_similarities_layer{args.layer}.png',
            color='brown'
        )

    plot_similarity_distribution(
        id_nonsense_similarities,
        f'ID-Nonsense Mean Similarity Distribution (Layer {args.layer})',
        output_dir / f'id_nonsense_mean_similarities_layer{args.layer}.png',
        color='orange'
    )

    plot_similarity_distribution(
        ood_nonsense_similarities,
        f'OOD-Nonsense Mean Similarity Distribution (Layer {args.layer})',
        output_dir / f'ood_nonsense_mean_similarities_layer{args.layer}.png',
        color='teal'
    )

    plot_comparison_distributions(
        id_within_similarities,
        nonsense_within_similarities,
        ['ID Within-group', 'Nonsense Within-group'],
        f'ID vs Nonsense Within-Group Similarities (Layer {args.layer})',
        output_dir / f'id_vs_nonsense_within_group_layer{args.layer}.png'
    )

    plot_comparison_distributions(
        ood_within_similarities,
        nonsense_within_similarities,
        ['OOD Within-group', 'Nonsense Within-group'],
        f'OOD vs Nonsense Within-Group Similarities (Layer {args.layer})',
        output_dir / f'ood_vs_nonsense_within_group_layer{args.layer}.png'
    )

    print("Finished all computations and plots!")

if __name__ == "__main__":
    main()