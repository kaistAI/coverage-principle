import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
import numpy as np
from scipy.spatial.distance import cosine
from collections import defaultdict
import argparse
from tqdm import tqdm


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
    with open(file_path, 'r') as f:
        return json.load(f)

def cosine_similarity(v1, v2):
    return 1 - cosine(v1, v2)

def process_vectors(data, layer):
    grouped_vectors = defaultdict(list)
    all_vectors = []
    
    for target, instances in data.items():
        for instance in instances:
            vector = instance['hidden_states'][0]['post_mlp']
            if vector is not None:
                grouped_vectors[target].append(vector)
                all_vectors.append(vector)
    
    return grouped_vectors, all_vectors

def calculate_within_group_similarity(group):
    if len(group) <= 1:
        return None
    
    similarities = []
    for i in range(len(group)):
        for j in range(i+1, len(group)):
            similarities.append(cosine_similarity(group[i], group[j]))
    
    return np.mean(similarities), similarities  # Return both mean and all similarities

def calculate_between_group_similarity(group_means):
    similarities = []
    for i in range(len(group_means)):
        for j in range(i+1, len(group_means)):
            similarities.append(cosine_similarity(group_means[i], group_means[j]))
    
    return np.mean(similarities), similarities  # Return both mean and all similarities

def calculate_metrics(grouped_vectors, all_vectors, ood=False):
    within_group_similarities_all = []
    between_group_similarities = []
    group_means = []
    
    # Collect all within-group similarities
    for group in grouped_vectors.values():
        if len(group) > 1:
            _, similarities = calculate_within_group_similarity(group)
            if similarities:
                within_group_similarities_all.extend(similarities)
        group_means.append(np.mean(group, axis=0))
    
    # Calculate between-group similarities
    _, between_similarities = calculate_between_group_similarity(group_means)
    between_group_similarities.extend(between_similarities)
    
    # Calculate all-vector similarities if OOD
    if ood:
        all_similarities = []
        for i in tqdm(range(len(all_vectors))):
            for j in range(i+1, len(all_vectors)):
                all_similarities.append(cosine_similarity(all_vectors[i], all_vectors[j]))
    else:
        all_similarities = None
    
    return (np.mean(within_group_similarities_all), within_group_similarities_all,
            np.mean(between_group_similarities), between_group_similarities,
            group_means, all_similarities)

def main():
    parser = argparse.ArgumentParser(description='Calculate and visualize similarity metrics for ID and OOD vectors')
    parser.add_argument('--id_file', required=True, help='Path to the ID vector file')
    parser.add_argument('--ood_file', required=True, help='Path to the OOD vector file')
    parser.add_argument('--layer', type=int, default=7, help='Layer to analyze (default: 7)')
    parser.add_argument('--output_dir', required=True, help='Directory to save the plots')
    args = parser.parse_args()

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    id_data = load_data(args.id_file)
    ood_data = load_data(args.ood_file)

    id_vectors, id_all_vectors = process_vectors(id_data, args.layer)
    ood_vectors, ood_all_vectors = process_vectors(ood_data, args.layer)

    print(f"Analyzing layer {args.layer}")

    # ID metrics with all similarity values
    (id_within_similarity, id_within_similarities,
     id_between_similarity, id_between_similarities,
     id_group_means, _) = calculate_metrics(id_vectors, id_all_vectors)

    print(f"ID within-group similarity: {id_within_similarity:.4f}")
    print(f"ID between-group similarity: {id_between_similarity:.4f}")

    # OOD metrics with all similarity values
    (ood_within_similarity, ood_within_similarities,
     ood_between_similarity, ood_between_similarities,
     ood_group_means, ood_all_similarities) = calculate_metrics(ood_vectors, ood_all_vectors, ood=True)

    print(f"OOD within-group similarity: {ood_within_similarity:.4f}")
    print(f"OOD between-group similarity: {ood_between_similarity:.4f}")
    print(f"OOD all-vector similarity: {np.mean(ood_all_similarities):.4f}")

    # Global OOD mean vs ID group means similarities
    ood_global_mean = np.mean(np.vstack(ood_group_means), axis=0)
    id_ood_similarities = [cosine_similarity(ood_global_mean, id_mean) for id_mean in id_group_means]
    avg_id_ood_similarity = np.mean(id_ood_similarities)
    print(f"Average similarity between OOD global mean and ID group means: {avg_id_ood_similarity:.4f}")

    # Create plots
    # 1. ID within-group vs between-group similarities
    plot_comparison_distributions(
        id_within_similarities,
        id_between_similarities,
        ['Within-group', 'Between-group'],
        f'ID Similarity Distributions (Layer {args.layer})',
        output_dir / f'id_similarities_layer{args.layer}.png'
    )

    # 2. OOD within-group vs between-group similarities
    plot_comparison_distributions(
        ood_within_similarities,
        ood_between_similarities,
        ['Within-group', 'Between-group'],
        f'OOD Similarity Distributions (Layer {args.layer})',
        output_dir / f'ood_similarities_layer{args.layer}.png'
    )

    # 3. All OOD similarities distribution
    plot_similarity_distribution(
        ood_all_similarities,
        f'OOD All-Vector Similarity Distribution (Layer {args.layer})',
        output_dir / f'ood_all_similarities_layer{args.layer}.png',
        color='green'
    )

    # 4. ID-OOD mean similarities distribution
    plot_similarity_distribution(
        id_ood_similarities,
        f'ID-OOD Mean Similarity Distribution (Layer {args.layer})',
        output_dir / f'id_ood_mean_similarities_layer{args.layer}.png',
        color='purple'
    )

    # 5. Combined ID and OOD within-group similarities
    plot_comparison_distributions(
        id_within_similarities,
        ood_within_similarities,
        ['ID Within-group', 'OOD Within-group'],
        f'ID vs OOD Within-Group Similarities (Layer {args.layer})',
        output_dir / f'id_vs_ood_within_group_layer{args.layer}.png'
    )

if __name__ == "__main__":
    main()