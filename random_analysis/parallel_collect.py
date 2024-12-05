import os
import subprocess
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from itertools import cycle

def run_collect_vectors(ckpt, save_dir, save_fname, device, base_dir, dataset):
    """
    Executes the collect_vectors.py script with specified arguments.

    Args:
        ckpt (str): Path to the model checkpoint.
        save_dir (str): Directory to save the analysis results.
        save_fname (str): Filename to save the analysis results.
        device (str): GPU device to use (e.g., 'cuda:0').
        base_dir (str): Base directory path.
    
    Returns:
        tuple: (checkpoint_path, success, error_message)
    """
    dataset_path = os.path.join(base_dir, dataset)
    layer_pos_pairs = "[(5,1)]"

    command = [
        "python", "collect.py",
        "--ckpt", ckpt,
        "--dataset", dataset_path,
        "--layer_pos_pairs", layer_pos_pairs,
        "--save_dir", save_dir,
        "--save_fname", save_fname,
        "--device", device,
        "--no_deduplicate"
    ]

    try:
        subprocess.run(command, check=True)
        return (ckpt, True, None)
    except subprocess.CalledProcessError as e:
        return (ckpt, False, str(e))

def main():
    parser = argparse.ArgumentParser(description="Parallelize collect_vectors.py across multiple checkpoints and GPUs with per-GPU limits.")
    parser.add_argument("--base_dir", required=True, help="Base directory path")
    parser.add_argument("--checkpoints_dir", required=True, help="Directory containing all checkpoints")
    parser.add_argument("--layer_pos_pairs", default="[(5,1)]", help="List of (layer, position) tuples to evaluate")
    parser.add_argument("--save_dir_base", required=False, help="Base directory to save the analysis results")
    parser.add_argument("--save_fname_pattern", default="composition_{}.json", help="Filename pattern for saving results (use {} for step)")
    parser.add_argument("--device_ids", nargs='+', default=["cuda:0", "cuda:1", "cuda:2", "cuda:3"], help="List of GPU device IDs to use")
    # parser.add_argument("--device_ids", nargs='+', default=["cuda:0"], help="List of GPU device IDs to use")
    parser.add_argument("--max_workers_per_gpu", type=int, default=10, help="Maximum number of parallel workers per GPU")
    parser.add_argument("--dataset", type=str, default="data/composition.2000.200.9.0/test_wid.json")
    args = parser.parse_args()

    base_dir = args.base_dir
    checkpoints_dir = args.checkpoints_dir
    save_dir_base = args.save_dir_base or os.path.join(base_dir, "results/raw/5-1")
    os.makedirs(save_dir_base, exist_ok=True)
    save_fname_pattern = args.save_fname_pattern
    device_ids = args.device_ids
    max_workers_per_gpu = args.max_workers_per_gpu
    dataset = args.dataset

    # List all checkpoint directories
    checkpoints = sorted([
        os.path.join(checkpoints_dir, d) for d in os.listdir(checkpoints_dir)
        if os.path.isdir(os.path.join(checkpoints_dir, d))
    ])

    total_checkpoints = len(checkpoints)
    if total_checkpoints == 0:
        print(f"No checkpoints found in {checkpoints_dir}. Exiting.")
        return

    print(f"Found {total_checkpoints} checkpoints to process.")

    # Create a cycle iterator for GPU devices
    gpu_cycle = cycle(device_ids)

    # Initialize a list of ProcessPoolExecutors, one per GPU
    executors = []
    for gpu in device_ids:
        executor = ProcessPoolExecutor(max_workers=max_workers_per_gpu)
        executors.append({'executor': executor, 'gpu': gpu})

    # Assign checkpoints to executors in round-robin fashion
    futures = []
    for idx, ckpt in enumerate(checkpoints):
        # Assign to the next GPU in the cycle
        executor_info = executors[idx % len(executors)]
        executor = executor_info['executor']
        gpu = executor_info['gpu']

        checkpoint_name = os.path.basename(ckpt)
        step = checkpoint_name.split('-')[-1]  # Assuming checkpoint names like 'checkpoint-<step>'
        save_fname = save_fname_pattern.format(step)
        save_dir = save_dir_base  # All results saved in the same directory

        future = executor.submit(run_collect_vectors, ckpt, save_dir, save_fname, gpu, base_dir, dataset)
        futures.append(future)

    # Monitor the progress of the tasks
    for future in as_completed(futures):
        ckpt, success, error = future.result()
        if success:
            print(f"[SUCCESS] Processed checkpoint: {ckpt}")
        else:
            print(f"[FAILURE] Checkpoint: {ckpt}, Error: {error}")
            assert False

    # Shutdown all executors
    for executor_info in executors:
        executor_info['executor'].shutdown()

    print("All checkpoints have been processed.")

if __name__ == "__main__":
    main()