import numpy as np
import os

# Replace with the actual path to a trajectory file
trajectory_path = "results/trajectories/test/test_inferred_iid/post_attention/<e_0><r_55><r_103>.npy"

trajectory = np.load(trajectory_path)
print(trajectory.shape)
print(trajectory)  # Expected Output: (number_of_checkpoints, vector_dimension)