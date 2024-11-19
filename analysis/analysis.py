import argparse
import json
import numpy as np
from tqdm import tqdm
from collections import defaultdict
import logging
import os
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from scipy.spatial.distance import cosine
from scipy.linalg import svd
from transformers import AutoModelForCausalLM
import torch

def load_model_and_get_unembedding(model_path):
    model = AutoModelForCausalLM.from_pretrained(model_path)
    unembedding = model.lm_head.weight.detach().numpy()
    return unembedding

def perform_svd_on_unembedding(unembedding):
    U, S, Vt = np.linalg.svd(unembedding, full_matrices=False)
    return U, S, Vt

def compute_similarities(vectors, singular_vectors):
    similarities = []
    for vector in vectors:
        sim = np.abs(np.dot(vector, singular_vectors.T))
        similarities.append(sim)
    return np.array(similarities)

def plot_unembedding_analysis(S, similarities, save_path):
    plt.figure(figsize=(12, 8))
    plt.plot(range(1, len(S) + 1), S, 'b-', linewidth=2, label='Singular Values')
    
    for i, sim in enumerate(similarities):
        plt.plot(range(1, len(sim) + 1), sim, alpha=0.5, label=f'Vector {i+1}')
    
    plt.xlabel('Singular Vector Index')
    plt.ylabel('Value / Similarity')
    plt.title('Unembedding Matrix Analysis')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.yscale('log')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def perform_ssa(vector_group, n_components=20):
    # Ensure we have a 2D matrix
    matrix = np.array(vector_group)
    if matrix.ndim == 1:
        matrix = matrix.reshape(1, -1)
    
    try:
        # Perform SVD
        U, s, Vt = svd(matrix, full_matrices=False)
        
        # Return the singular values (spectrum)
        return s[:n_components]
    except np.linalg.LinAlgError:
        logging.warning(f"SVD did not converge for matrix of shape {matrix.shape}")
        return np.zeros(n_components)

def normalize_vectors(vectors):
    return vectors / np.linalg.norm(vectors, axis=1, keepdims=True)

# Modify the sample_and_perform_ssa function to use normalized vectors
def sample_and_perform_ssa(vector_groups, sample_size=1000, n_components=20):
    np.random.seed(42)
    ssa_results = {}
    sample_size = min([sample_size] + [len(v) for v in vector_groups.values()])
    print(f"sample_size: {sample_size}")
    for key, vectors in vector_groups.items():
        if len(vectors) >= sample_size:
            sampled_vectors = vectors[np.random.choice(len(vectors), sample_size, replace=False)]
        else:
            sampled_vectors = vectors[np.random.choice(len(vectors), sample_size, replace=True)]
        
        normalized_vectors = normalize_vectors(sampled_vectors)
        spectrum = perform_ssa(normalized_vectors, n_components)
        ssa_results[str(key)] = spectrum.tolist()
    
    return ssa_results

def plot_ssa_comparison(ssa_results, save_dir):
    vector_types = ['embedding', 'post_attention', 'post_mlp']
    for vector_type in vector_types:
        plt.figure(figsize=(12, 8))
        for key, spectrum in ssa_results.items():
            if vector_type in key:
                label = '_'.join(key.split(', ')[:3])  # Create label from layer, position, and data type
                plt.plot(range(1, len(spectrum) + 1), spectrum, '-', label=label)
        plt.xlabel('Component')
        plt.ylabel('Singular Value')
        plt.title(f'Singular Spectrum Comparison for {vector_type}')
        plt.yscale('log')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        save_path = os.path.join(save_dir, f'ssa_comparison_{vector_type}.png')
        plt.savefig(save_path)
        plt.close()
        logging.info(f"SSA comparison plot for {vector_type} saved to {save_path}")

def perform_pca_and_tsne_visualize(vector_groups, save_dir):
    for (layer, position) in set((l, p) for l, p, _, _ in vector_groups.keys()):
        for key in ['embedding', 'post_attention', 'post_mlp']:
            data = []
            labels = []
            for data_type in ['train_inferred', 'test_inferred_iid', 'test_inferred_ood']:
                vectors = vector_groups.get((layer, position, data_type, key), [])
                data.extend(vectors)
                labels.extend([data_type] * len(vectors))
            
            if not data:
                continue
            
            # Convert data to numpy array
            data = np.array(data)
            
            # Perform PCA
            pca = PCA(n_components=2)
            pca_result = pca.fit_transform(data)
            
            # Perform t-SNE
            tsne = TSNE(n_components=2, random_state=42)
            tsne_result = tsne.fit_transform(data)
            
            # Create and save PCA plot
            create_and_save_plot(pca_result, labels, layer, position, key, 'PCA', pca.explained_variance_ratio_, save_dir)
            
            # Create and save t-SNE plot
            create_and_save_plot(tsne_result, labels, layer, position, key, 't-SNE', None, save_dir)

def create_and_save_plot(result, labels, layer, position, key, method, var_explained, save_dir):
    plt.figure(figsize=(10, 8))
    
    colors = {'train_inferred': 'r', 'test_inferred_iid': 'g', 'test_inferred_ood': 'b'}
    for data_type in ['train_inferred', 'test_inferred_iid', 'test_inferred_ood']:
        mask = np.array(labels) == data_type
        plt.scatter(result[mask, 0], result[mask, 1],
                    c=colors[data_type], label=data_type, alpha=0.6)
    
    plt.xlabel('First Component')
    plt.ylabel('Second Component')
    plt.legend()
    plt.title(f'{method} of {key} vectors (Layer {layer}, Position {position})')
    
    if var_explained is not None:
        plt.text(0.05, 0.95, f'Variance Explained: {var_explained[0]:.2f}, {var_explained[1]:.2f}',
                 transform=plt.gca().transAxes, verticalalignment='top')
    
    save_path = os.path.join(save_dir, f'{method.lower()}_layer{layer}_pos{position}_{key}.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    logging.info(f"{method} visualization saved to {save_path}")

def compute_average_cosine_similarity(vectors):
    n = len(vectors)
    if n < 2:
        return 1.0  # Perfect similarity for 0 or 1 vector
    
    total_similarity = 0
    count = 0
    for i in range(n):
        for j in range(i+1, n):
            total_similarity += 1 - cosine(vectors[i], vectors[j])
            count += 1
    
    return total_similarity / count if count > 0 else 1.0

def compute_average_magnitude(vectors):
    return np.mean([np.linalg.norm(v) for v in vectors])

def load_dataset(dataset_path):
    with open(dataset_path, 'r') as f:
        dataset = json.load(f)
    logging.debug(f"Loaded {len(dataset)} instances from {dataset_path}")
    return dataset

def setup_logging(debug_mode):
    level = logging.DEBUG if debug_mode else logging.INFO
    logging.basicConfig(level=level, format='%(asctime)s - %(levelname)s - %(message)s')

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, help="Path to the evaluation dataset")
    parser.add_argument("--layer_pos_pairs", required=True, help="List of (layer, position) tuples to evaluate")
    parser.add_argument("--save_dir", required=True, help="Directory to save the analysis results")
    parser.add_argument("--save_fname", required=True, help="Filename to save the analysis results")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode for verbose output")
    parser.add_argument("--debug_samples", type=int, default=100, help="Number of samples to process in debug mode")
    parser.add_argument("--model_path", required=True, help="Path to the GPT-2 model checkpoint")
    parser.add_argument("--sample_size", type=int, default=5, help="Number of vectors to sample for unembedding analysis")
    
    args = parser.parse_args()
    
    setup_logging(args.debug)
    
    logging.info("Starting vector analysis...")
    
    dataset = load_dataset(args.dataset)
    layer_pos_pairs = eval(args.layer_pos_pairs)
    logging.debug(f"Layer position pairs: {layer_pos_pairs}")
    
    vector_groups = defaultdict(list)
    
    for data_type, instances in dataset.items():
        logging.info(f"Processing {data_type}...")
        for idx, instance in enumerate(tqdm(instances)):
            if args.debug and idx >= args.debug_samples:
                logging.debug(f"Reached debug sample limit ({args.debug_samples}) for {data_type}")
                break
            for hidden_state in instance['hidden_states']:
                layer = hidden_state['layer']
                position = hidden_state['position']
                if (layer, position) in layer_pos_pairs:
                    for key in ['embedding', 'post_attention', 'post_mlp']:
                        if key in hidden_state:
                            vector = hidden_state[key]
                            if vector is not None:
                                vector_groups[(layer, position, data_type, key)].append(np.array(vector))
    
    logging.info(f"Number of vector groups: {len(vector_groups)}")
    
    # Convert vector_groups values to numpy arrays
    for key in vector_groups:
        vector_groups[key] = np.array(vector_groups[key])
    
    # Load model and get unembedding matrix
    # unembedding = load_model_and_get_unembedding(args.model_path)
    
    # Perform SVD on unembedding matrix
    # U, S, Vt = perform_svd_on_unembedding(unembedding)
    
    # Sample vectors and compute similarities
    # unembedding_analysis = {}
    # for key, vectors in vector_groups.items():
    #     if len(vectors) >= args.sample_size:
    #         sampled_vectors = vectors[np.random.choice(len(vectors), args.sample_size, replace=False)]
    #     else:
    #         sampled_vectors = vectors[np.random.choice(len(vectors), args.sample_size, replace=True)]
        
    #     similarities = compute_similarities(sampled_vectors, Vt.T)
    #     unembedding_analysis[str(key)] = similarities.tolist()
    
    # Plot unembedding analysis
    # unembedding_plot_dir = os.path.join(args.save_dir, 'unembedding_plots')
    # os.makedirs(unembedding_plot_dir, exist_ok=True)
    
    # for key, similarities in unembedding_analysis.items():
    #     save_path = os.path.join(unembedding_plot_dir, f'unembedding_analysis_{key.replace(", ", "_")}.png')
    #     plot_unembedding_analysis(S, similarities, save_path)
    #     logging.info(f"Unembedding analysis plot saved to {save_path}")
    
    # Perform SSA
    ssa_results = sample_and_perform_ssa(vector_groups, sample_size=1000, n_components=20)
    logging.info(f"Number of SSA results: {len(ssa_results)}")

    # Optionally, create and save plots of the singular value spectra
    ssa_plot_dir = os.path.join(args.save_dir, 'ssa_plots')
    os.makedirs(ssa_plot_dir, exist_ok=True)
    for key, spectrum in ssa_results.items():
        plt.figure(figsize=(10, 6))
        plt.plot(range(1, len(spectrum) + 1), spectrum, 'bo-')
        plt.xlabel('Component')
        plt.ylabel('Singular Value')
        plt.title(f'Singular Spectrum for {key}')
        plt.yscale('log')
        save_path = os.path.join(ssa_plot_dir, f'ssa_spectrum_{key.replace(", ", "_")}.png')
        plt.savefig(save_path)
        plt.close()
        logging.info(f"SSA spectrum plot saved to {save_path}")
    
    plot_ssa_comparison(ssa_results, ssa_plot_dir)
    
    # Perform PCA and t-SNE visualizations
    vis_save_dir = os.path.join(args.save_dir, 'visualizations')
    os.makedirs(vis_save_dir, exist_ok=True)
    perform_pca_and_tsne_visualize(vector_groups, vis_save_dir)
    
    # Compute average cosine similarity and average magnitude for each group
    similarities = {}
    magnitudes = {}
    for key, vectors in tqdm(vector_groups.items(), desc="Computing similarities and magnitudes"):
        if len(vectors) > 1:
            similarities[str(key)] = compute_average_cosine_similarity(vectors)
            magnitudes[str(key)] = compute_average_magnitude(vectors)
    
    logging.info(f"Number of similarity results: {len(similarities)}")
    logging.info(f"Number of magnitude results: {len(magnitudes)}")

    # Save results including unembedding analysis
    save_path = os.path.join(args.save_dir, args.save_fname)
    with open(save_path, 'w') as f:
        json.dump({
            "average_similarities": similarities,
            "average_magnitudes": magnitudes,
            "ssa_results": ssa_results,
            # "unembedding_analysis": unembedding_analysis
        }, f, indent=2)
    
    logging.info(f"Analysis results saved to {save_path}")
    
    logging.info(f"Analysis results saved to {save_path}")

if __name__ == "__main__":
    main()