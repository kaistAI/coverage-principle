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
import random
import torch
import torch.nn.functional as F
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import plotly.express as px
from typing import Optional

def load_data(file_path):
    print(f"\nLoading data from: {file_path}")
    with open(file_path, 'r') as f:
        data = json.load(f)
    print(f"Successfully loaded. Number of top-level keys in JSON: {len(data)}")
    return data

def process_vectors(data, layer):
    """
    Convert the JSON data into grouped_vectors and all_vectors lists.
    Also store 'grouped_instances' => bridging -> list of (vector, input_text).
    """
    print(f"\nProcessing vectors for layer {layer}.")
    grouped_vectors = defaultdict(list)
    grouped_instances = defaultdict(list)
    all_vectors = []

    for target, instances in tqdm(data.items(), desc="Grouping vectors by target"):
        for instance in instances:
            hidden_states = instance['hidden_states']
            if not hidden_states:
                continue
            vector = hidden_states[0].get('post_mlp', None)
            if vector is not None:
                grouped_vectors[target].append(vector)
                # store (vector, input_text) for multi random references
                grouped_instances[target].append((vector, instance['input_text']))
                all_vectors.append(vector)
    
    print(f"Finished processing. Number of groups: {len(grouped_vectors)}. "
          f"Total vectors collected: {len(all_vectors)}")
    return grouped_vectors, grouped_instances, all_vectors

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

# -------------- EXISTING UTILITY FOR MULTIPLE RANDOM GROUPS --------------
def sample_references_and_collect_ranges_multi(grouped_instances, out_json_file, n_groups=3):
    """
    1) Randomly pick n_groups bridging groups from 'grouped_instances'.
    2) For each bridging group:
       a) pick a random reference instance (vector, input_text)
       b) compute cos sim with others in that group
       c) gather up to 10 items in each of 3 ranges:
           [-0.05..0.05], [0.30..0.40], [0.65..0.75]
    3) Save all bridging results in out_json_file as a list of bridging-blobs.

    NOTE: If fewer bridging groups exist or bridging has <2 items, skip it.
    """
    bridging_keys = [k for k,v in grouped_instances.items() if k!='unknown' and len(v)>1]
    random.shuffle(bridging_keys)
    bridging_keys = bridging_keys[:n_groups]  # pick up to n_groups if possible

    device = "cpu"
    results = []
    
    for bridging in bridging_keys:
        inst_list = grouped_instances[bridging]  # list of (vec, in_txt)
        if len(inst_list) < 2:
            continue
        ref_idx = random.randrange(len(inst_list))
        ref_vec, ref_txt = inst_list[ref_idx]

        # normalize the reference
        ref_t = torch.tensor(ref_vec, dtype=torch.float32, device=device)
        ref_t = F.normalize(ref_t.unsqueeze(0), p=2, dim=1)[0]

        # gather cos sims
        all_cos = []
        for i,(v,inp_txt) in enumerate(inst_list):
            if i == ref_idx:
                continue
            v_t = torch.tensor(v, dtype=torch.float32, device=device)
            v_t = F.normalize(v_t.unsqueeze(0), p=2, dim=1)[0]
            cos_sim = float(torch.dot(ref_t, v_t).item())
            all_cos.append((cos_sim, inp_txt))

        # define 2 ranges (customize as you wish)
        ranges = [
          (-0.05, 0.05),
          (0.90, 1.00)
        ]
        bridging_data = {
          "bridging_key": bridging,
          "reference_instance": ref_txt,
          "samples": []
        }
        for (rmin, rmax) in ranges:
            matched = [(c,txt) for (c,txt) in all_cos if rmin <= c <= rmax]
            matched.sort(key=lambda x: x[0])
            matched = matched[:30]
            bridging_data["samples"].append({
              "range": f"[{rmin:.2f},{rmax:.2f}]",
              "count": len(matched),
              "items": [{"cos": c, "input_text": txt} for (c,txt) in matched]
            })
        results.append(bridging_data)

    out_data = {
      "meta": f"Collected from {n_groups} random bridging groups",
      "bridgings": results
    }
    with open(out_json_file, "w") as f:
        json.dump(out_data, f, indent=2)
    print(f"Saved multi-range samples => {out_json_file}")

###############################################################################
# 1) A function to compute the embedding via PCA or t-SNE
###############################################################################
def compute_embedding(X, method='pca', dim=2):
    """
    X: np.array shape [N, D]
    method in {'pca','tsne'}
    dim in {2,3}
    
    Returns:
      X_emb shape [N, dim], a numpy array with the new embedding.
    """
    if method == 'pca':
        reducer = PCA(n_components=dim)
        X_emb = reducer.fit_transform(X)
    else:
        # t-SNE
        reducer = TSNE(n_components=dim, perplexity=30, 
                       n_iter=1000, verbose=1)
        X_emb = reducer.fit_transform(X)
    return X_emb

###############################################################################
# 2) The main plotting function: either 2D => Matplotlib PNG,
#    or 3D => Plotly HTML, for either PCA or t-SNE.
###############################################################################
def plot_embedding(
    grouped_vectors,
    output_path,
    m=3,
    n=5,
    title="Embedding Visualization",
    reduce_dim=2,
    reduce_method="pca",
    scope="global"
):
    """
    scope in {'global','local'} => see your existing logic
      if scope='global', gather *all* bridging groups
      if scope='local', pick bridging groups of size>=n, up to m groups,
                       from each group, we gather n vectors for uniform size.

    reduce_method in {'pca','tsne'}
    reduce_dim in {2,3}

    We produce a 2D or 3D scatter:
      - 2D => static PNG (Matplotlib)
      - 3D => interactive HTML (Plotly)
    """
    bridging_list = []
    all_points = []

    # 1) gather vectors
    if scope == "global":
        # gather all bridging vectors
        for bridging_key, vlist in grouped_vectors.items():
            for v in vlist:
                bridging_list.append(bridging_key)
                all_points.append(v)
    else:
        # local => only bridging groups of size >= n
        valid_groups = [k for k,v in grouped_vectors.items() if k!='unknown' and len(v)>=n]
        random.shuffle(valid_groups)
        valid_groups = valid_groups[:m]
        for bridging_key in valid_groups:
            vlist = grouped_vectors[bridging_key]
            random.shuffle(vlist)
            subset = vlist[:n]  # pick exactly n
            for vec in subset:
                bridging_list.append(bridging_key)
                all_points.append(vec)
    
    # print(bridging_list)
    if not all_points:
        print(f"No vectors => skip {reduce_method} scope={scope}.")
        return

    X = np.array(all_points)
    if X.shape[0] < 2:
        print("Fewer than 2 vectors => skip embedding.")
        return

    # 2) compute embedding
    print(f"Compute {reduce_method.upper()} with dim={reduce_dim} on {X.shape[0]} points...")
    X_emb = compute_embedding(X, method=reduce_method, dim=reduce_dim)

    # 3) We want to color bridging groups that have at least n items, up to m.
    #   Even in 'global' mode, we do the same for coloring.
    group_sizes= defaultdict(int)
    for b in bridging_list:
        group_sizes[b] += 1
    chosen_keys = [k for k in group_sizes if k!='unknown' and group_sizes[k]>=n]
    random.shuffle(chosen_keys)
    chosen_keys = chosen_keys[:m]

    # We'll skip bridging keys not in chosen_keys (or color them lightly if you prefer).
    # For minimal changes => skip them.
    # 4) 2D => static PNG, 3D => interactive HTML
    if reduce_dim == 2:
        fig, ax = plt.subplots(figsize=(8,6))
        palette = sns.color_palette("hls", len(chosen_keys))
        key2color = {}
        for i,k in enumerate(chosen_keys):
            key2color[k] = palette[i]

        for i,bkey in enumerate(bridging_list):
            if bkey not in key2color:
                continue
            c = key2color[bkey]
            ax.scatter(X_emb[i,0], X_emb[i,1], color=c, s=20, alpha=0.7)

        handles = []
        for i,k in enumerate(chosen_keys):
            c= palette[i]
            handles.append( plt.Line2D([],[], marker='o', color=c, label=str(k), linestyle='None') )
        ax.legend(handles=handles, bbox_to_anchor=(1.05,1), loc='upper left')
        ax.set_title(f"{reduce_method.upper()} {reduce_dim}D: {title}")
        plt.tight_layout()
        plt.savefig(output_path)
        plt.close()
        print(f"{reduce_method.upper()} 2D => saved {output_path}")

    elif reduce_dim == 3:
        import pandas as pd
        bridging_chosen = []
        for b in bridging_list:
            if b in chosen_keys:
                bridging_chosen.append(b)
            else:
                bridging_chosen.append(None)  # skip coloring

        df_data = {
          "x": X_emb[:,0],
          "y": X_emb[:,1],
          "z": X_emb[:,2],
          "bridging": bridging_chosen
        }
        df = pd.DataFrame(df_data)

        fig = px.scatter_3d(
            df, x="x", y="y", z="z",
            color="bridging",
            title=f"{reduce_method.upper()} {reduce_dim}D: {title}",
            opacity=0.7
        )
        fig.update_layout(width=900, height=700)
        html_file= str(output_path.with_suffix(".html"))
        fig.write_html(html_file)
        print(f"{reduce_method.upper()} 3D => interactive => {html_file}")
    else:
        print("reduce_dim must be 2 or 3.")

def main():
    parser = argparse.ArgumentParser(description='Calculate and visualize similarity metrics for ID, OOD, and Nonsense vectors')
    parser.add_argument('--id_train_file', required=True, help='Path to the ID Train vector file')
    parser.add_argument('--id_test_file', required=True, help='Path to the ID Test vector file')
    parser.add_argument('--ood_file', required=True, help='Path to the OOD vector file')
    parser.add_argument('--output_dir', required=True, help='Directory to save the plots and results')
    parser.add_argument('--save_plots', action='store_true', default=False,
                        help="If set, will generate and save plot images. Otherwise skip.")
    parser.add_argument('--num_random_groups', type=int, default=50,
                        help="Number of random bridging groups to sample for the new utility.")
    
    # PCA/TSNE arguments
    parser.add_argument('--reduce_dim',  type=Optional[int], choices=[2,3],
                        help="Dimension of PCA/t-SNE => 2 or 3")
    parser.add_argument('--reduce_method', type=str, default='pca',
                        choices=['pca','tsne'],
                        help="Which method => pca or tsne")
    parser.add_argument('--pca_scope', type=str, default="global", choices=["global","local"],
                        help="Perform PCA/t-SNE globally or only local sample bridging groups")
    parser.add_argument('--pca_m', type=int, default=5)
    parser.add_argument('--pca_n', type=int, default=20)
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    matches = re.findall(r"\((logit|prob|\d+),(\d+)\)", args.output_dir)
    assert len(matches) == 1, "Expected exactly one (layer, pos) pattern in output_dir name."
    target_layer = matches[0][0]

    metrics_json_file = output_dir / "metrics_results.json"

    # Load data
    id_train_data = load_data(args.id_train_file)
    id_test_data  = load_data(args.id_test_file)
    ood_data      = load_data(args.ood_file)

    # Process into grouped vectors
    id_train_vectors,  id_train_instances,  id_train_all_vectors = process_vectors(id_train_data,  target_layer)
    id_test_vectors,   id_test_instances,   id_test_all_vectors  = process_vectors(id_test_data,   target_layer)
    ood_vectors,       ood_instances,       ood_all_vectors      = process_vectors(ood_data,       target_layer)
    
    with open(metrics_json_file, 'r') as f:
        cossim_result = json.load(f)

    # PCA/t-SNE
    dr_dir = output_dir / "pca"
    dr_dir.mkdir(exist_ok=True)
    
    # ID Train
    if args.reduce_dim == None:
        out_id_train = dr_dir / f"id_train_dim2_{args.reduce_method}_{args.pca_scope}.png"
        plot_embedding(
            grouped_vectors=id_train_vectors,
            output_path=out_id_train,
            m=args.pca_m, n=args.pca_n,
            reduce_dim=2,
            reduce_method=args.reduce_method,
            scope=args.pca_scope,
            title=f"ID Train (Layer {target_layer})"
        )
        out_id_train = dr_dir / f"id_train_dim3_{args.reduce_method}_{args.pca_scope}.png"
        plot_embedding(
            grouped_vectors=id_train_vectors,
            output_path=out_id_train,
            m=args.pca_m, n=args.pca_n,
            reduce_dim=3,
            reduce_method=args.reduce_method,
            scope=args.pca_scope,
            title=f"ID Train (Layer {target_layer})"
        )
    else:
        out_id_train = dr_dir / f"id_train_dim{args.reduce_dim}_{args.reduce_method}_{args.pca_scope}.png"
        plot_embedding(
            grouped_vectors=id_train_vectors,
            output_path=out_id_train,
            m=args.pca_m, n=args.pca_n,
            reduce_dim=args.reduce_dim,
            reduce_method=args.reduce_method,
            scope=args.pca_scope,
            title=f"ID Train (Layer {target_layer})"
        )

    # ID Test
    if args.reduce_dim == None:
        out_id_test = dr_dir / f"id_test_dim2_{args.reduce_method}_{args.pca_scope}.png"
        plot_embedding(
            grouped_vectors=id_test_vectors,
            output_path=out_id_test,
            m=args.pca_m, n=args.pca_n,
            reduce_dim=2,
            reduce_method=args.reduce_method,
            scope=args.pca_scope,
            title=f"ID Test (Layer {target_layer})"
        )
        out_id_test = dr_dir / f"id_test_dim3_{args.reduce_method}_{args.pca_scope}.png"
        plot_embedding(
            grouped_vectors=id_test_vectors,
            output_path=out_id_test,
            m=args.pca_m, n=args.pca_n,
            reduce_dim=3,
            reduce_method=args.reduce_method,
            scope=args.pca_scope,
            title=f"ID Test (Layer {target_layer})"
        )
    else:
        out_id_test = dr_dir / f"id_test_dim{args.reduce_dim}_{args.reduce_method}_{args.pca_scope}.png"
        plot_embedding(
            grouped_vectors=id_test_vectors,
            output_path=out_id_test,
            m=args.pca_m, n=args.pca_n,
            reduce_dim=args.reduce_dim,
            reduce_method=args.reduce_method,
            scope=args.pca_scope,
            title=f"ID Test (Layer {target_layer})"
        )

    # OOD
    if args.reduce_dim == None:
        out_ood = dr_dir / f"ood_dim2_{args.reduce_method}_{args.pca_scope}.png"
        plot_embedding(
            grouped_vectors=ood_vectors,
            output_path=out_ood,
            m=args.pca_m, n=args.pca_n,
            reduce_dim=2,
            reduce_method=args.reduce_method,
            scope=args.pca_scope,
            title=f"OOD (Layer {target_layer})"
        )
        out_ood = dr_dir / f"ood_dim3_{args.reduce_method}_{args.pca_scope}.png"
        plot_embedding(
            grouped_vectors=ood_vectors,
            output_path=out_ood,
            m=args.pca_m, n=args.pca_n,
            reduce_dim=3,
            reduce_method=args.reduce_method,
            scope=args.pca_scope,
            title=f"OOD (Layer {target_layer})"
        )
    else:
        out_ood = dr_dir / f"ood_dim{args.reduce_dim}_{args.reduce_method}_{args.pca_scope}.png"
        plot_embedding(
            grouped_vectors=ood_vectors,
            output_path=out_ood,
            m=args.pca_m, n=args.pca_n,
            reduce_dim=args.reduce_dim,
            reduce_method=args.reduce_method,
            scope=args.pca_scope,
            title=f"OOD (Layer {target_layer})"
        )

    # -------------- Sample multiple random bridging groups --------------
    multi_range_file = output_dir / f"range_samples_layer{target_layer}.json"
    sample_references_and_collect_ranges_multi(id_train_instances, multi_range_file, n_groups=args.num_random_groups)

    if not args.save_plots:
        print("\nUser disabled plot saving; skipping figure generation.")
        return

    # =============== Create Plots ===============
    print("\nCreating plots...")

    # ID Train
    plot_comparison_distributions(
        cossim_result["id_train"]["within_sims"],
        cossim_result["id_train"]["between_sims"],
        ['Within-group', 'Between-group(centroids)'],
        f'ID Train Similarity Distributions (Layer {target_layer})',
        output_dir / f'id_train_similarities_layer{target_layer}.png'
    )
    # Also plot the new "between-group(all vectors)" distribution if it exists
    if cossim_result["id_train"]["between_allvec_sims"]:
        plot_comparison_distributions(
            cossim_result["id_train"]["within_sims"],
            cossim_result["id_train"]["between_allvec_sims"],
            ['Within-group', 'Between-group(allvec)'],
            f'ID Train Within vs. Between(AllVec) (Layer {target_layer})',
            output_dir / f'id_train_within_vs_betweenall_layer{target_layer}.png'
        )
    if cossim_result["id_train"]["all_sims"] and len(cossim_result["id_train"]["all_sims"])>0:
        plot_similarity_distribution(
            cossim_result["id_train"]["all_sims"],
            f'ID Train All-Vector Similarities (Layer {target_layer})',
            output_dir / f'id_train_all_similarities_layer{target_layer}.png',
            color='green'
        )

    # ID Test
    plot_comparison_distributions(
        cossim_result["id_test"]["within_sims"],
        cossim_result["id_test"]["between_sims"],
        ['Within-group', 'Between-group(centroids)'],
        f'ID Test Similarities (Layer {target_layer})',
        output_dir / f'id_test_similarities_layer{target_layer}.png'
    )
    if cossim_result["id_test"]["between_allvec_sims"]:
        plot_comparison_distributions(
            cossim_result["id_test"]["within_sims"],
            cossim_result["id_test"]["between_allvec_sims"],
            ['Within-group', 'Between-group(allvec)'],
            f'ID Test Within vs. Between(AllVec) (Layer {target_layer})',
            output_dir / f'id_test_within_vs_betweenall_layer{target_layer}.png'
        )
    if cossim_result["id_test"]["all_sims"] and len(cossim_result["id_test"]["all_sims"])>0:
        plot_similarity_distribution(
            cossim_result["id_test"]["all_sims"],
            f'ID Test All-Vector Similarities (Layer {target_layer})',
            output_dir / f'id_test_all_similarities_layer{target_layer}.png',
            color='green'
        )

    # OOD
    plot_comparison_distributions(
        cossim_result["ood"]["within_sims"],
        cossim_result["ood"]["between_sims"],
        ['Within-group', 'Between-group(centroids)'],
        f'OOD Similarities (Layer {target_layer})',
        output_dir / f'ood_similarities_layer{target_layer}.png'
    )
    if cossim_result["ood"]["between_allvec_sims"]:
        plot_comparison_distributions(
            cossim_result["ood"]["within_sims"],
            cossim_result["ood"]["between_allvec_sims"],
            ['Within-group', 'Between-group(allvec)'],
            f'OOD Within vs. Between(AllVec) (Layer {target_layer})',
            output_dir / f'ood_within_vs_betweenall_layer{target_layer}.png'
        )
    if cossim_result["ood"]["all_sims"] and len(cossim_result["ood"]["all_sims"])>0:
        plot_similarity_distribution(
            cossim_result["ood"]["all_sims"],
            f'OOD All-Vector Similarities (Layer {target_layer})',
            output_dir / f'ood_all_similarities_layer{target_layer}.png',
            color='green'
        )

    plot_similarity_distribution(
        cossim_result["cross_category"]["id_train_ood_sims"],
        f'ID_Train vs OOD Mean Similarities (Layer {target_layer})',
        output_dir / f'id_train_ood_mean_similarities_layer{target_layer}.png',
        color='purple'
    )
    plot_similarity_distribution(
        cossim_result["cross_category"]["id_test_ood_sims"],
        f'ID_Test vs OOD Mean Similarities (Layer {target_layer})',
        output_dir / f'id_test_ood_mean_similarities_layer{target_layer}.png',
        color='purple'
    )
    plot_comparison_distributions(
        cossim_result["id_train"]["within_sims"],
        cossim_result["ood"]["within_sims"],
        ['ID_Train Within-group', 'OOD Within-group'],
        f'ID_Train vs OOD Within-Group (Layer {target_layer})',
        output_dir / f'id_train_vs_ood_within_group_layer{target_layer}.png'
    )
    plot_comparison_distributions(
        cossim_result["id_test"]["within_sims"],
        cossim_result["ood"]["within_sims"],
        ['ID_Test Within-group', 'OOD Within-group'],
        f'ID_Test vs OOD Within-Group (Layer {target_layer})',
        output_dir / f'id_test_vs_ood_within_group_layer{target_layer}.png'
    )

    print("Finished all computations and plots!")

if __name__ == "__main__":
    main()