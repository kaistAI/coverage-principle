import numpy as np
from scipy.linalg import svd
import matplotlib.pyplot as plt

def perform_ssa(vector_group, n_components=10):
    # Ensure we have a 2D matrix
    matrix = np.array(vector_group)
    if matrix.ndim == 1:
        matrix = matrix.reshape(1, -1)
    
    # Perform SVD
    U, s, Vt = svd(matrix, full_matrices=False)
    
    # Return the singular values (spectrum)
    return s[:n_components]

def sample_and_perform_ssa(vector_groups, sample_size=1000, n_components=10):
    ssa_results = {}
    for key, vectors in vector_groups.items():
        if len(vectors) >= sample_size:
            sampled_vectors = vectors[np.random.choice(len(vectors), sample_size, replace=False)]
        else:
            sampled_vectors = vectors[np.random.choice(len(vectors), sample_size, replace=True)]
        spectrum = perform_ssa(sampled_vectors, n_components)
        ssa_results[str(key)] = spectrum.tolist()
    return ssa_results

def generate_random_matrix(rows, cols):
    return np.random.rand(rows, cols)

def plot_and_save_ssa(spectrum, output_file):
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(spectrum) + 1), spectrum, 'bo-')
    plt.title('Singular Spectrum Analysis (SSA) Plot')
    plt.xlabel('Component')
    plt.ylabel('Singular Value')
    plt.grid(True)
    plt.savefig(output_file)
    plt.close()

def main():
    # Generate a 768x1000 random matrix
    matrix = generate_random_matrix(768, 1000)
    
    # Perform SSA
    vector_groups = {'random_matrix': matrix}
    ssa_results = sample_and_perform_ssa(vector_groups, sample_size=1000, n_components=20)
    
    # Plot and save the results
    spectrum = ssa_results['random_matrix']
    plot_and_save_ssa(spectrum, 'ssa_plot.png')
    print("SSA plot has been saved as 'ssa_plot.png'")

if __name__ == "__main__":
    main()