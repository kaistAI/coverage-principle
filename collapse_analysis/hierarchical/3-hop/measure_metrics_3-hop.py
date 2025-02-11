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
    from scipy.spatial.distance import cosine
    return 1 - cosine(v1, v2)

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

def calculate_within_group_similarity_gpu(group):
    k = len(group)
    if k <= 1:
        return None, []
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    X = torch.tensor(group, dtype=torch.float32, device=device)
    X = F.normalize(X, p=2, dim=1)
    sim_matrix= torch.matmul(X, X.t())
    idx= torch.triu_indices(k,k,offset=1)
    sims_upper= sim_matrix[idx[0], idx[1]]
    sims_list= sims_upper.cpu().numpy().tolist()
    return float(np.mean(sims_list)), sims_list

def calculate_between_group_similarity_gpu(group_means):
    M= len(group_means)
    if M<2:
        return None, []
    device= torch.device("cuda" if torch.cuda.is_available() else "cpu")
    X= torch.tensor(group_means,dtype=torch.float32,device=device)
    X= F.normalize(X,p=2,dim=1)
    sim_matrix= torch.matmul(X,X.t())
    idx= torch.triu_indices(M,M,offset=1)
    sims_upper= sim_matrix[idx[0], idx[1]]
    sims_list= sims_upper.cpu().numpy().tolist()
    return float(np.mean(sims_list)), sims_list

def compute_all_similarities_gpu(all_vectors):
    if len(all_vectors)<2:
        return []
    device= torch.device("cuda" if torch.cuda.is_available() else "cpu")
    X= torch.tensor(all_vectors,dtype=torch.float32,device=device)
    X= F.normalize(X,p=2,dim=1)
    sim_matrix= torch.matmul(X, X.t())
    N= sim_matrix.size(0)
    idx= torch.triu_indices(N,N,offset=1)
    sims_upper= sim_matrix[idx[0],idx[1]]
    return sims_upper.cpu().numpy().tolist()

def calculate_metrics(grouped_vectors, all_vectors, ood=False):
    within_all=[]
    between_all=[]
    group_means=[]
    for bridging, group in tqdm(grouped_vectors.items()):
        if bridging=='unknown':
            print('skip unknown bridging')
            continue
        if len(group)>1:
            _, sims= calculate_within_group_similarity_gpu(group)
            within_all.extend(sims)
        group_means.append(np.mean(group, axis=0))
    _, between_sims= calculate_between_group_similarity_gpu(group_means)
    between_all.extend(between_sims)
    all_sims= compute_all_similarities_gpu(all_vectors)
    return (np.mean(within_all) if within_all else 0,
            within_all,
            np.mean(between_all) if between_all else 0,
            between_all,
            group_means,
            all_sims)


# -------------- NEW UTILITY #1 FOR MULTIPLE RANDOM GROUPS --------------
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

    bridging_keys= [k for k,v in grouped_instances.items() if k!='unknown' and len(v)>1]
    random.shuffle(bridging_keys)
    bridging_keys= bridging_keys[:n_groups]  # pick up to n_groups if possible

    device= torch.device("cuda" if torch.cuda.is_available() else "cpu")
    results=[]
    
    for bridging in bridging_keys:
        inst_list= grouped_instances[bridging]  # list of (vec, in_txt)
        if len(inst_list)<2:
            continue
        ref_idx= random.randrange(len(inst_list))
        ref_vec, ref_txt= inst_list[ref_idx]

        # normalize the reference
        ref_t= torch.tensor(ref_vec,dtype=torch.float32,device=device)
        ref_t= F.normalize(ref_t.unsqueeze(0),p=2,dim=1)[0]

        # gather cos sims
        all_cos=[]
        for i,(v,inp_txt) in enumerate(inst_list):
            if i==ref_idx:
                continue
            v_t= torch.tensor(v,dtype=torch.float32,device=device)
            v_t= F.normalize(v_t.unsqueeze(0),p=2,dim=1)[0]
            cos_sim= float(torch.dot(ref_t,v_t).item())
            all_cos.append((cos_sim,inp_txt))

        # define 3 ranges
        ranges= [
          (-0.05, 0.05),
          (0.90, 1.00)
        ]
        bridging_data={
          "bridging_key": bridging,
          "reference_instance": ref_txt,
          "samples":[]
        }
        for (rmin,rmax) in ranges:
            matched= [(c,txt) for (c,txt) in all_cos if c>=rmin and c<=rmax]
            matched.sort(key=lambda x:x[0])
            matched= matched[:30]
            bridging_data["samples"].append({
              "range": f"[{rmin:.2f},{rmax:.2f}]",
              "count": len(matched),
              "items":[ {"cos": c, "input_text": txt} for (c,txt) in matched ]
            })
        results.append(bridging_data)

    out_data={
      "meta": f"Collected from {n_groups} random bridging groups in ID train_inferred",
      "bridgings": results
    }
    with open(out_json_file,"w") as f:
        json.dump(out_data,f,indent=2)
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
        # You can tune perplexity, etc. for your data
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
        group_sizes[b]+=1
    chosen_keys = [k for k in group_sizes if k!='unknown' and group_sizes[k]>=n]
    random.shuffle(chosen_keys)
    chosen_keys= chosen_keys[:m]

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
            c= key2color[bkey]
            ax.scatter(X_emb[i,0], X_emb[i,1], color=c, s=20, alpha=0.7)

        handles= []
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
        # produce 3D interactive plot => .html
        bridging_chosen= []
        for b in bridging_list:
            if b in chosen_keys:
                bridging_chosen.append(b)
            else:
                bridging_chosen.append(None)  # skip coloring

        import pandas as pd
        df_data= {
          "x": X_emb[:,0],
          "y": X_emb[:,1],
          "z": X_emb[:,2],
          "bridging": bridging_chosen
        }
        df= pd.DataFrame(df_data)

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
    
    # PCA arguments
    parser.add_argument('--pca_vis',  action='store_true', default=False)
    parser.add_argument('--reduce_dim',  type=int, default=2, choices=[2,3],
                        help="Dimension of PCA => 2 or 3")
    parser.add_argument('--reduce_method', type=str, default='pca',
                        choices=['pca','tsne'],
                        help="Which method => pca or tsne")
    parser.add_argument('--pca_scope', type=str, default="global", choices=["global","local"],
                        help="Perform PCA globally or only local sample bridging groups")
    parser.add_argument('--pca_m', type=int, default=5)
    parser.add_argument('--pca_n', type=int, default=20)

    
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    matches = re.findall(r"\(\d+,\s*\d+\)", args.output_dir)
    assert len(matches) == 1, "Expected exactly one (layer, pos) pattern in output_dir name."
    target_layer = matches[0].strip(")(").split(",")[0]

    results_file = output_dir / f"similarity_metrics_layer{target_layer}.txt"

    # Load data
    id_train_data = load_data(args.id_train_file)
    id_test_data  = load_data(args.id_test_file)
    ood_data      = load_data(args.ood_file)

    # Process into grouped vectors
    id_train_vectors,  id_train_instances,  id_train_all_vectors = process_vectors(id_train_data,  target_layer)
    id_test_vectors,   id_test_instances,   id_test_all_vectors  = process_vectors(id_test_data,   target_layer)
    ood_vectors,       ood_instances,       ood_all_vectors      = process_vectors(ood_data,       target_layer)

    print(f"\nAnalyzing layer {target_layer}...")

    # =============== ID Train Metrics ===============
    print("\nComputing ID Train metrics...")
    (id_train_within_sim, id_train_within_sims,
     id_train_between_sim, id_train_between_sims,
     id_train_group_means, id_train_all_sims) = calculate_metrics(id_train_vectors, id_train_all_vectors, ood=False)
    
    print(f"  ID within-group sim (mean): {id_train_within_sim:.4f}")
    print(f"  ID between-group sim (mean): {id_train_between_sim:.4f}")
    
    # =============== ID Test Metrics ===============
    print("\nComputing ID Test metrics...")
    (id_test_within_sim, id_test_within_sims,
     id_test_between_sim, id_test_between_sims,
     id_test_group_means, id_test_all_sims) = calculate_metrics(id_test_vectors, id_test_all_vectors, ood=False)
    
    print(f"  ID within-group sim (mean): {id_test_within_sim:.4f}")
    print(f"  ID between-group sim (mean): {id_test_between_sim:.4f}")

    # =============== OOD Metrics ===============
    print("\nComputing OOD metrics...")
    (ood_within_sim, ood_within_sims,
     ood_between_sim, ood_between_sims,
     ood_group_means, ood_all_sims) = calculate_metrics(ood_vectors, ood_all_vectors, ood=True)
    
    print(f"  OOD within-group sim (mean): {ood_within_sim:.4f}")
    print(f"  OOD between-group sim (mean): {ood_between_sim:.4f}")
    if ood_all_sims:
        print(f"  OOD all-vector sim (mean): {np.mean(ood_all_sims):.4f}")
    else:
        print("  OOD all-vector sim: None (only one OOD vector?)")

    # =============== Cross-Category Comparisons ===============
    print("\nCross-Category Similarities...")

    # ID Train group mean vs ID Test group mean (same bridging)
    id_train_gm= {}
    for be, vlist in id_train_vectors.items():
        id_train_gm[be]= np.mean(vlist, axis=0)
    id_test_gm= {}
    for be, vlist in id_test_vectors.items():
        id_test_gm[be]= np.mean(vlist, axis=0)
    intersec= set(id_train_gm.keys()) & set(id_test_gm.keys())
    from scipy.spatial.distance import cosine
    def cosim(a,b): return 1- cosine(a,b)
    id_train_test_sims= [ cosim(id_train_gm[be], id_test_gm[be]) for be in intersec ]
    avg_id_train_test_sims= np.mean(id_train_test_sims) if id_train_test_sims else 0
    print(f"  (ID Train group mean vs ID Test group mean): {avg_id_train_test_sims:.4f}")

    # OOD global mean vs ID train means
    if len(ood_group_means)>0:
        ood_gm= np.mean(np.vstack(ood_group_means),axis=0)
        id_train_ood_s= [ cosim(ood_gm,x) for x in id_train_group_means ]
        avg_id_train_ood_s= np.mean(id_train_ood_s)
    else:
        id_train_ood_s=[]
        avg_id_train_ood_s=0
    print(f"  (OOD global vs ID train means): {avg_id_train_ood_s:.4f}")

    # OOD global mean vs ID test means
    if len(ood_group_means)>0:
        id_test_ood_s= [cosim(ood_gm,x) for x in id_test_group_means]
        avg_id_test_ood_s= np.mean(id_test_ood_s)
    else:
        id_test_ood_s=[]
        avg_id_test_ood_s=0
    print(f"  (OOD global vs ID test means): {avg_id_test_ood_s:.4f}")

    # Save text results
    results_file_str= str(results_file)
    print(f"\nWriting results to {results_file_str}...")
    with open(results_file_str, 'w') as f:
        f.write(f"Analyzing layer {target_layer}\n\n")

        f.write("=== ID Train ===\n")
        f.write(f"Within-group sim (mean): {id_train_within_sim:.4f}\n")
        f.write(f"Between-group sim (mean): {id_train_between_sim:.4f}\n")
        if id_train_all_sims:
            f.write(f"All-vector sim (mean): {np.mean(id_train_all_sims):.4f}\n\n")
        else:
            f.write("All-vector sim (mean): None\n\n")

        f.write("=== ID Test ===\n")
        f.write(f"Within-group sim (mean): {id_test_within_sim:.4f}\n")
        f.write(f"Between-group sim (mean): {id_test_between_sim:.4f}\n")
        if id_test_all_sims:
            f.write(f"All-vector sim (mean): {np.mean(id_test_all_sims):.4f}\n\n")
        else:
            f.write("All-vector sim (mean): None\n\n")

        f.write("=== OOD ===\n")
        f.write(f"Within-group sim (mean): {ood_within_sim:.4f}\n")
        f.write(f"Between-group sim (mean): {ood_between_sim:.4f}\n")
        if ood_all_sims:
            f.write(f"All-vector sim (mean): {np.mean(ood_all_sims):.4f}\n\n")
        else:
            f.write("All-vector sim (mean): None\n\n")

        f.write("=== Cross-Category ===\n")
        f.write(f"(IDTrain vs IDTest) same bridging: {avg_id_train_test_sims:.4f}\n")
        f.write(f"(OOD global vs IDTrain means): {avg_id_train_ood_s:.4f}\n")
        f.write(f"(OOD global vs IDTest means): {avg_id_test_ood_s:.4f}\n")

    if args.pca_vis:
        # create subdir
        dr_dir= output_dir / "pca"
        dr_dir.mkdir(exist_ok=True)

        # e.g. "id_train_dim2_pca_global.png" or ".html"
        # Assume you have your id_train_vectors, etc. from process_vectors

        # ID train
        out_id_train= dr_dir / f"id_train_dim{args.reduce_dim}_{args.reduce_method}_{args.pca_scope}.png"
        plot_embedding(
            grouped_vectors=id_train_vectors,
            output_path=out_id_train,
            m=args.pca_m, n=args.pca_n,
            reduce_dim=args.reduce_dim,
            reduce_method=args.reduce_method,
            scope=args.pca_scope,
            title=f"ID Train (Layer {target_layer})"
        )
        # ID test
        out_id_test= dr_dir / f"id_test_dim{args.reduce_dim}_{args.reduce_method}_{args.pca_scope}.png"
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
        out_ood= dr_dir / f"ood_dim{args.reduce_dim}_{args.reduce_method}_{args.pca_scope}.png"
        plot_embedding(
            grouped_vectors=ood_vectors,
            output_path=out_ood,
            m=args.pca_m, n=args.pca_n,
            reduce_dim=args.reduce_dim,
            reduce_method=args.reduce_method,
            scope=args.pca_scope,
            title=f"OOD (Layer {target_layer})"
        )


    # -------------- NEW UTILITY: sample multiple random bridging groups --------------
    multi_range_file= output_dir / f"range_samples_layer{target_layer}.json"
    sample_references_and_collect_ranges_multi(id_train_instances, multi_range_file, n_groups=args.num_random_groups)

    # Optionally skip plots
    if not args.save_plots:
        print("\nUser disabled plot saving; skipping figure generation.")
        return

    # =============== Create Plots ===============
    print("\nCreating plots...")

    # ID Train
    plot_comparison_distributions(
        id_train_within_sims,
        id_train_between_sims,
        ['Within-group', 'Between-group'],
        f'ID Train Similarity Distributions (Layer {target_layer})',
        output_dir / f'id_train_similarities_layer{target_layer}.png'
    )
    if id_train_all_sims and len(id_train_all_sims)>0:
        plot_similarity_distribution(
            id_train_all_sims,
            f'ID Train All-Vector Similarities (Layer {target_layer})',
            output_dir / f'id_train_all_similarities_layer{target_layer}.png',
            color='green'
        )

    # ID Test
    plot_comparison_distributions(
        id_test_within_sims,
        id_test_between_sims,
        ['Within-group', 'Between-group'],
        f'ID Test Similarities (Layer {target_layer})',
        output_dir / f'id_test_similarities_layer{target_layer}.png'
    )
    if id_test_all_sims and len(id_test_all_sims)>0:
        plot_similarity_distribution(
            id_test_all_sims,
            f'ID Test All-Vector Similarities (Layer {target_layer})',
            output_dir / f'id_test_all_similarities_layer{target_layer}.png',
            color='green'
        )

    # OOD
    plot_comparison_distributions(
        ood_within_sims,
        ood_between_sims,
        ['Within-group', 'Between-group'],
        f'OOD Similarities (Layer {target_layer})',
        output_dir / f'ood_similarities_layer{target_layer}.png'
    )
    if ood_all_sims and len(ood_all_sims)>0:
        plot_similarity_distribution(
            ood_all_sims,
            f'OOD All-Vector Similarities (Layer {target_layer})',
            output_dir / f'ood_all_similarities_layer{target_layer}.png',
            color='green'
        )

    plot_similarity_distribution(
        id_train_ood_s,
        f'ID_Train vs OOD Mean Similarities (Layer {target_layer})',
        output_dir / f'id_train_ood_mean_similarities_layer{target_layer}.png',
        color='purple'
    )
    plot_similarity_distribution(
        id_test_ood_s,
        f'ID_Test vs OOD Mean Similarities (Layer {target_layer})',
        output_dir / f'id_test_ood_mean_similarities_layer{target_layer}.png',
        color='purple'
    )

    plot_comparison_distributions(
        id_train_within_sims,
        ood_within_sims,
        ['ID_Train Within-group', 'OOD Within-group'],
        f'ID_Train vs OOD Within-Group (Layer {target_layer})',
        output_dir / f'id_train_vs_ood_within_group_layer{target_layer}.png'
    )
    plot_comparison_distributions(
        id_test_within_sims,
        ood_within_sims,
        ['ID_Test Within-group', 'OOD Within-group'],
        f'ID_Test vs OOD Within-Group (Layer {target_layer})',
        output_dir / f'id_test_vs_ood_within_group_layer{target_layer}.png'
    )

    print("Finished all computations and plots!")

if __name__ == "__main__":
    main()