import argparse
import random
import os
import json
from collections import defaultdict
import numpy as np
from tqdm import tqdm
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


def load_and_process_data(id_test_file):
    """데이터를 로드하고 처리하는 함수"""
    with open(id_test_file, 'r') as f:
        id_test_data = json.load(f)
        
    grouped_vectors = defaultdict(list)
    grouped_instances = defaultdict(list)
    all_vectors = []

    for target, instances in tqdm(id_test_data.items(), desc="Grouping vectors by target"):
        for instance in instances:
            hidden_states = instance['hidden_states']
            if not hidden_states:
                continue
            vector = hidden_states[0].get('post_mlp', None)
            if vector is not None:
                grouped_vectors[target].append(vector)
                grouped_instances[target].append((vector, instance['input_text']))
                all_vectors.append(vector)
    
    equivalent_classes = []
    all_points = []

    for equivalent_class, vlist in grouped_vectors.items():
        for v in vlist:
            equivalent_classes.append(equivalent_class)
            all_points.append(v)
    
    return np.array(all_points), equivalent_classes, grouped_vectors


def compute_embedding(X, reduce_method, reduce_dim):
    """차원 축소를 수행하는 함수"""
    print(f"Compute {reduce_method.upper()} with dim={reduce_dim} on {X.shape[0]} points...")
    if reduce_method == 'pca':
        reducer = PCA(n_components=reduce_dim)
        X_emb = reducer.fit_transform(X)
    else:  # t-SNE
        reducer = TSNE(n_components=reduce_dim, perplexity=30, n_iter=1000, verbose=1)
        X_emb = reducer.fit_transform(X)
    
    # 각 차원별로 최대 자릿수를 구하고 그 값으로 나누어 정규화
    for i in range(reduce_dim):
        min_val = X_emb[:, i].min()
        max_val = X_emb[:, i].max()
        
        # 최대 자릿수 값 구하기
        max_digit = max(abs(min_val), abs(max_val))
        max_digit = 10 ** (int(np.log10(max_digit)))
        
        # 최대 자릿수 값으로 나누기
        X_emb[:, i] = X_emb[:, i] / max_digit
    
    return X_emb
    

def plot_3d_scatter(ax, X_emb, bridging_list, key2color, title):
    """3D 산점도를 그리는 함수"""
    # 그리드 라인 추가
    ax.grid(True, linestyle='--', alpha=0.3)
    
    # 배경색 설정
    ax.set_facecolor('white')
    
    # legend를 위한 scatter 객체 저장
    legend_elements = []
    
    # 각 그룹별로 데이터 포인트 그리기
    for key in key2color:
        # 해당 key를 가진 데이터 포인트의 인덱스 찾기
        indices = [i for i, bkey in enumerate(bridging_list) if bkey == key]
        if not indices:
            continue
            
        # scatter plot 생성
        scatter = ax.scatter(X_emb[indices,0], X_emb[indices,1], X_emb[indices,2], 
                           color=key2color[key], s=30, alpha=0.7,
                           edgecolor='black', linewidth=0.5,
                           label=key)
        legend_elements.append(scatter)
    
    # 축 눈금 설정
    ax.tick_params(axis='both', which='major', labelsize=14)
    
    # 제목 설정 (아래로 이동)
    ax.set_title(title, fontsize=20, pad=20, y=-0.2)
    
    # legend 추가 (마지막 subplot에만)
    if ax == plt.gcf().axes[-1]:
        ax.legend(handles=legend_elements, 
                 loc='center left',
                 bbox_to_anchor=(1.1, 0.5),
                 fontsize=14)


def visualize_pca(pca_files, reduce_method, reduce_dim, pca_m, pca_n, output_dir_path, local=False):
    
    random.seed(0)
    
    #########################################################
    # 0. 각각의 PCA plot의 제목
    #########################################################
    section_titles = ["$k < 3$", "$k > 3$", "OOD"]
    
    #########################################################
    # 1. PCA plot에 표시할 공통된 key 찾기
    #########################################################
    all_key_list = []
    min_instance_4_each_classes = {}
    for pca_file in pca_files:
        all_keys = set()
        all_points, equivalent_classes, grouped_vectors = load_and_process_data(pca_file)
        if all_points.shape[0] < 2:
            print(f"This file has less than 2 vectors => skip embedding.")
            continue
        all_keys.update(equivalent_classes)
        all_key_list.append(all_keys)
        for equivalent_class, vlist in grouped_vectors.items():
            if not equivalent_class in min_instance_4_each_classes:
                min_instance_4_each_classes[equivalent_class] = len(vlist)
            else:
                min_instance_4_each_classes[equivalent_class] = min(min_instance_4_each_classes[equivalent_class], len(vlist))
    
    print(min_instance_4_each_classes)
        
    # 모든 key_list의 교집합 찾기
    common_keys = list(set.intersection(*all_key_list))
    # 공통된 key 선택
    common_keys = [k for k in common_keys if min_instance_4_each_classes[k] >= pca_n]
    # min_instance_4_each_classes[x]를 기준으로 내림차순 정렬하고, 같은 값이 있을 경우 int(x.split("_")[-1])를 기준으로 정렬
    common_keys.sort(key=lambda x: (-min_instance_4_each_classes[x], int(x.split("_")[-1])))
    # random.shuffle(common_keys)
    chosen_keys = common_keys[:pca_m]
    print("\n선택된 keys:", chosen_keys, [min_instance_4_each_classes[chosen_key] for chosen_key in chosen_keys])
    
    palette = sns.color_palette("hls", len(chosen_keys))
    key2color = {k: palette[i] for i, k in enumerate(chosen_keys)}
    
    #########################################################
    # 2. PCA plot의 데이터 범위 계산
    #########################################################
    all_x_min, all_x_max = float('inf'), float('-inf')
    all_y_min, all_y_max = float('inf'), float('-inf')
    if reduce_dim == 3:
        all_z_min, all_z_max = float('inf'), float('-inf')
    
    # 선택된 key에 해당하는 데이터만 사용하여 범위 계산
    for pca_file in pca_files:
        all_points, equivalent_classes, grouped_vectors = load_and_process_data(pca_file)
        
        if local:
            # 각 키별로 정확히 pca_n개의 데이터만 선택
            filtered_points = []
            filtered_classes = []
            for key in chosen_keys:
                selected_indices = [i for i, eq_class in enumerate(equivalent_classes) if eq_class == key]
                # if len(indices) > pca_n:
                #     selected_indices = random.sample(indices, pca_n)
                # else:
                #     selected_indices = indices
                filtered_points.extend([all_points[i] for i in selected_indices])
                filtered_classes.extend([equivalent_classes[i] for i in selected_indices])
            all_points = np.array(filtered_points)
            equivalent_classes = filtered_classes
        
        X_emb = compute_embedding(all_points, reduce_method, reduce_dim)
        print(X_emb)
        
        # 선택된 key에 해당하는 데이터만 필터링
        mask = [equivalent_class in chosen_keys for equivalent_class in equivalent_classes]
        if not any(mask):
            continue
        filtered_X_emb = X_emb[mask]
        
        # 전체 데이터의 범위 업데이트
        all_x_min = min(all_x_min, filtered_X_emb[:, 0].min())
        all_x_max = max(all_x_max, filtered_X_emb[:, 0].max())
        all_y_min = min(all_y_min, filtered_X_emb[:, 1].min())
        all_y_max = max(all_y_max, filtered_X_emb[:, 1].max())
        if reduce_dim == 3:
            all_z_min = min(all_z_min, filtered_X_emb[:, 2].min())
            all_z_max = max(all_z_max, filtered_X_emb[:, 2].max())

    # 여백 추가
    margin = 0.1
    x_range = all_x_max - all_x_min
    y_range = all_y_max - all_y_min
    all_x_min -= x_range * margin
    all_x_max += x_range * margin
    all_y_min -= y_range * margin
    all_y_max += y_range * margin
    
    if reduce_dim == 3:
        z_range = all_z_max - all_z_min
        all_z_min -= z_range * margin
        all_z_max += z_range * margin

    #########################################################
    # 3. PCA plot 그리기
    #########################################################
    fig_height = 4
    fig_width = 4.25 * len(pca_files)
    fig = plt.figure(figsize=(fig_width, fig_height))
    gs = gridspec.GridSpec(1, len(pca_files), width_ratios=[1] * (len(pca_files)), wspace=0.25)
    axes = [plt.subplot(gs[i]) for i in range(len(pca_files))]
    
    for idx, pca_file in enumerate(pca_files):
        ax = axes[idx]
        all_points, equivalent_classes, grouped_vectors = load_and_process_data(pca_file)
        
        if local:
            # 각 키별로 정확히 pca_n개의 데이터만 선택
            filtered_points = []
            filtered_classes = []
            for key in chosen_keys:
                selected_indices = [i for i, eq_class in enumerate(equivalent_classes) if eq_class == key]
                # if len(indices) > pca_n:
                #     selected_indices = random.sample(indices, pca_n)
                # else:
                #     selected_indices = indices
                filtered_points.extend([all_points[i] for i in selected_indices])
                filtered_classes.extend([equivalent_classes[i] for i in selected_indices])
            all_points = np.array(filtered_points)
            equivalent_classes = filtered_classes
        
        X_emb = compute_embedding(all_points, reduce_method, reduce_dim)
        
        # 각 key별 데이터 포인트 개수 출력
        print(f"\nPlot {idx + 1} ({section_titles[idx]}) 데이터 분포:")
        for key in key2color:
            count = equivalent_classes.count(key)
            print(f"{key}: {count} points")
        
        if reduce_dim == 3:
            # 3D plot 생성
            plot_3d_scatter(ax, X_emb, equivalent_classes, key2color, 
                          f"{reduce_method.upper()} 3D: {section_titles[idx]}")
            # 모든 subplot에 동일한 범위 적용
            ax.set_xlim(all_x_min, all_x_max)
            ax.set_ylim(all_y_min, all_y_max)
            ax.set_zlim(all_z_min, all_z_max)
        else:
            # 2D plot 그리기
            legend_elements = []
            for key in key2color:
                # 해당 key를 가진 데이터 포인트의 인덱스 찾기
                indices = [i for i, bkey in enumerate(equivalent_classes) if bkey == key]
                if not indices:
                    continue
                    
                # scatter plot 생성
                scatter = ax.scatter(X_emb[indices,0], X_emb[indices,1], 
                                   color=key2color[key], s=20, alpha=0.8,
                                   label=key)
                legend_elements.append(scatter)
            
            ax.set_title(f"{section_titles[idx]}", fontsize=20, y=-0.25)
            ax.set_xlim(all_x_min, all_x_max)
            ax.set_ylim(all_y_min, all_y_max)
            
            # tick label fontsize 설정
            ax.tick_params(axis='both', which='major', labelsize=16)
            
            # # legend 추가 (마지막 subplot에만)
            # if idx == len(pca_files) - 1:
            #     ax.legend(handles=legend_elements, 
            #              loc='center left',
            #              bbox_to_anchor=(1.02, 0.5),
            #              fontsize=12)
            
    fig.add_artist(plt.Line2D([0.64, 0.64], [-0.075, 0.9], color='gray', linestyle='--', alpha=0.5, transform=fig.transFigure))
            
    # 전체 figure 저장
    output_path = os.path.join(output_dir_path, f"{reduce_method}_{reduce_dim}D_{'local' if local else 'global'}.pdf")
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()
    print(f"Saved combined plot to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Compute PCA and save sorted JSON")
    parser.add_argument('--pca_files', nargs='+',
                        help="PCA 파일 경로들")
    parser.add_argument('--reduce_method', type=str, default='pca',
                        choices=['pca','tsne'],
                        help="Which method => pca or tsne")
    parser.add_argument('--reduce_dim', type=int, default=2,
                        help="Reduce dimension")
    parser.add_argument('--pca_m', type=int, default=5)
    parser.add_argument('--pca_n', type=int, default=20)
    parser.add_argument('--local', action='store_true',
                        help="각 키별로 정확히 pca_n개의 데이터만 사용")
    parser.add_argument(
        "--output_dir", default="heatmaps",
        help="heatmap PDF 파일들을 저장할 디렉토리 (기본: heatmaps)"
    )
    args = parser.parse_args()
    
    # 모든 파일의 basename이 동일한지 확인
    settings = [f.split("residual/")[1].split("/")[0] for f in args.pca_files]
    if len(set(settings)) != 1:
        raise ValueError(f"모든 PCA 파일의 dataset setting은 동일해야 합니다. 현재 settings: {settings}")
    
    output_dir_path = os.path.join(args.output_dir, settings[0])
    
    visualize_pca(args.pca_files, args.reduce_method, args.reduce_dim, args.pca_m, args.pca_n, output_dir_path, args.local)


if __name__ == "__main__":
    main()
    
    