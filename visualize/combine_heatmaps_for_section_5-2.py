import argparse
import os
import glob
import re
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.gridspec as gridspec
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from collections import defaultdict
from tqdm import tqdm
import random
from sklearn.metrics import silhouette_score
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial.distance import pdist, squareform

def extract_diff(content, section):
    """
    콘텐츠에서 지정된 섹션의 'Within-group sim (mean)' 과 'Between-group (all vectors) sim (mean)'을 찾아
    차이만 반환합니다. 없으면 None.
    """
    # 지정된 섹션만 사용
    parts = content.split(f"=== {section} ===", 1)
    if len(parts) < 2:
        return None
    section_content = parts[1].split("===")[0]

    pat_within = re.compile(r"Within-group sim \(mean\):\s*([0-9.]+)")
    pat_between_all = re.compile(r"Between-group \(all vectors\) sim \(mean\):\s*([0-9.]+)")

    m_within = pat_within.search(section_content)
    m_between_all = pat_between_all.search(section_content)
    if not (m_within and m_between_all):
        return None

    return float(m_within.group(1)) - float(m_between_all.group(1))

def collect_diffs_twohop(root_dir, section):
    """
    2-hop 데이터셋을 위한 재귀 탐색으로 모든 similarity_metrics_layer*.txt 파일을 찾아
    detailed_grouping에 따라 (cutoff_type, f, layer, pos, step) 또는 (k, f, layer, pos, step) 별로 차이를 수집합니다.
    _detailed_grouping_이 포함된 경우에만 detailed_grouping을 체크하고, 그렇지 않은 경우 기본 구조로 저장합니다.
    """
    result = {}
    # 파일 경로에서 f, cutoff_type/k, (layer,pos), step 추출용 정규식
    re_f         = re.compile(r"[\\/](f\d+)[\\/]")
    re_cutoff    = re.compile(r"/(low_cutoff|mid_cutoff|high_cutoff)/")
    re_k         = re.compile(r"/covered_(\d+)/")
    re_layer_pos = re.compile(r"/\((\d+),(\d+)\)/")
    re_step      = re.compile(r"/step(\d+|final_checkpoint)/")
    
    # _detailed_grouping_ 포함 여부 확인
    has_detailed_grouping = "_detailed_grouping_" in root_dir
    
    pattern = os.path.join(root_dir, "**", "similarity_metrics_layer*.txt")
    for path in glob.glob(pattern, recursive=True):
        with open(path, encoding="utf-8") as f:
            diff = extract_diff(f.read(), section)
        assert diff is not None, f"No diff found in file: {path}"

        # path 에서 metadata 추출
        fm = re_f.search(path)
        cm = re_cutoff.search(path)
        km = re_k.search(path)
        lp = re_layer_pos.search(path)
        sm = re_step.search(path)
        assert fm != None and sm != None, f"All metadata should be present: {path}, {fm}, {sm}"
        if not lp:  # layer가 prob이거나 logit인 경우 무시
            continue    

        f_key        = fm.group(1)              # e.g. "f1"
        layer_idx    = lp.group(1)              # e.g. "6"
        pos_idx      = lp.group(2)              # e.g. "2"
        step_key     = "step" + sm.group(1)     # e.g. "step25000" or "stepfinal_checkpoint"
        layer_key    = "layer" + layer_idx
        pos_key      = "pos"   + pos_idx
        
        if has_detailed_grouping:
            # detailed_grouping 판단 및 데이터 구조화
            if cm:  # N=1 또는 N=2
                cutoff_key = cm.group(1)  # e.g. "low_cutoff" or "mid_cutoff" or "high_cutoff"
                result.setdefault(cutoff_key, {})\
                      .setdefault(f_key, {})\
                      .setdefault(step_key, {})\
                      .setdefault(layer_key, {})[pos_key] = diff
            elif km:  # N=3
                k_key = km.group(1)  # e.g. "2" or "3" or "4"
                result.setdefault(k_key, {})\
                      .setdefault(f_key, {})\
                      .setdefault(step_key, {})\
                      .setdefault(layer_key, {})[pos_key] = diff
            else:
                continue  # 알 수 없는 패턴은 무시
        else:
            # detailed_grouping이 없는 경우, 기본 구조로 저장
            result.setdefault("default", {})\
                  .setdefault(f_key, {})\
                  .setdefault(step_key, {})\
                  .setdefault(layer_key, {})[pos_key] = diff
              
    return result


def sort_nested_twohop(data):
    """
    2-hop 데이터셋을 위한 정렬 함수
    detailed_grouping에 따라 다른 방식으로 정렬된 새 dict 반환.
    모든 step의 데이터를 저장합니다.
    """
    out = {}
    
    # detailed_grouping 판단
    if "low_cutoff" in data and "mid_cutoff" in data and "high_cutoff" in data:
        # N=2: low_cutoff, mid_cutoff, high_cutoff
        for cutoff in ["low_cutoff", "mid_cutoff", "high_cutoff"]:
            if cutoff not in data:
                continue
            out[cutoff] = {}
            fdict = data[cutoff]
            # f1 데이터만 처리
            if "f1" not in fdict:
                continue
            stepdict = fdict["f1"]
            # 모든 step 처리
            for step in sorted(stepdict.keys(), key=lambda x: float('inf') if x == "stepfinal_checkpoint" else int(x.replace("step",""))):
                out[cutoff][step] = {}
                ldict = stepdict[step]
                # layer 키 정렬
                for layer in sorted(ldict.keys(), key=lambda x: int(x.replace("layer",""))):
                    out[cutoff][step][layer] = {}
                    pdict = ldict[layer]
                    # pos 키 정렬
                    for pos in sorted(pdict.keys(), key=lambda x: int(x.replace("pos",""))):
                        out[cutoff][step][layer][pos] = pdict[pos]
    elif "low_cutoff" in data and "high_cutoff" in data:
        # N=1: low_cutoff, high_cutoff
        for cutoff in ["low_cutoff", "high_cutoff"]:
            if cutoff not in data:
                continue
            out[cutoff] = {}
            fdict = data[cutoff]
            # f1 데이터만 처리
            if "f1" not in fdict:
                continue
            stepdict = fdict["f1"]
            # 모든 step 처리
            for step in sorted(stepdict.keys(), key=lambda x: float('inf') if x == "stepfinal_checkpoint" else int(x.replace("step",""))):
                out[cutoff][step] = {}
                ldict = stepdict[step]
                # layer 키 정렬
                for layer in sorted(ldict.keys(), key=lambda x: int(x.replace("layer",""))):
                    out[cutoff][step][layer] = {}
                    pdict = ldict[layer]
                    # pos 키 정렬
                    for pos in sorted(pdict.keys(), key=lambda x: int(x.replace("pos",""))):
                        out[cutoff][step][layer][pos] = pdict[pos]
    elif "2" in data and "3" in data and "4" in data:
        # N=3: 2, 3, 4
        for k in ["2", "3", "4"]:
            if k not in data:
                continue
            out[k] = {}
            fdict = data[k]
            # f1 데이터만 처리
            if "f1" not in fdict:
                continue
            stepdict = fdict["f1"]
            # 모든 step 처리
            for step in sorted(stepdict.keys(), key=lambda x: float('inf') if x == "stepfinal_checkpoint" else int(x.replace("step",""))):
                out[k][step] = {}
                ldict = stepdict[step]
                # layer 키 정렬
                for layer in sorted(ldict.keys(), key=lambda x: int(x.replace("layer",""))):
                    out[k][step][layer] = {}
                    pdict = ldict[layer]
                    # pos 키 정렬
                    for pos in sorted(pdict.keys(), key=lambda x: int(x.replace("pos",""))):
                        out[k][step][layer][pos] = pdict[pos]
    elif "default" in data:
        # detailed_grouping이 없는 경우
        out["default"] = {}
        fdict = data["default"]
        # 모든 f_key 처리
        if "f1" not in fdict:
            raise ValueError("No data found in default section")
        stepdict = fdict["f1"]
        # 모든 step 처리
        for step in sorted(stepdict.keys(), key=lambda x: float('inf') if x == "stepfinal_checkpoint" else int(x.replace("step",""))):
            out["default"][step] = {}
            ldict = stepdict[step]
            # layer 키 정렬
            for layer in sorted(ldict.keys(), key=lambda x: int(x.replace("layer",""))):
                out["default"][step][layer] = {}
                pdict = ldict[layer]
                # pos 키 정렬
                for pos in sorted(pdict.keys(), key=lambda x: int(x.replace("pos",""))):
                    out["default"][step][layer][pos] = pdict[pos]
    else:
        raise ValueError("Unknown detailed_grouping pattern in data")
    
    return out

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
    
    bridging_list = []
    all_points = []

    for bridging_key, vlist in grouped_vectors.items():
        for v in vlist:
            bridging_list.append(bridging_key)
            all_points.append(v)
            
    return np.array(all_points), bridging_list, grouped_vectors

def compute_embedding(X, reduce_method, reduce_dim):
    """차원 축소를 수행하는 함수"""
    print(f"Compute {reduce_method.upper()} with dim={reduce_dim} on {X.shape[0]} points...")
    if reduce_method == 'pca':
        reducer = PCA(n_components=reduce_dim)
        return reducer.fit_transform(X)
    else:  # t-SNE
        reducer = TSNE(n_components=reduce_dim, perplexity=30, n_iter=1000, verbose=1)
        return reducer.fit_transform(X)

def select_chosen_keys(bridging_list, pca_n, pca_m):
    """chosen_keys를 선택하는 함수"""
    group_sizes = defaultdict(int)
    for b in bridging_list:
        group_sizes[b] += 1
    chosen_keys = [k for k in group_sizes if k!='unknown' and group_sizes[k]>=pca_n]
    random.shuffle(chosen_keys)
    return chosen_keys[:pca_m]

def calculate_plot_range(X_emb, bridging_list, chosen_keys):
    """plot의 범위를 계산하는 함수"""
    chosen_indices = [i for i, b in enumerate(bridging_list) if b in chosen_keys]
    if not chosen_indices:
        return None, None, None, None
        
    chosen_X_emb = X_emb[chosen_indices]
    x_min = chosen_X_emb[:, 0].min()
    x_max = chosen_X_emb[:, 0].max()
    y_min = chosen_X_emb[:, 1].min()
    y_max = chosen_X_emb[:, 1].max()
    
    # 여백 추가
    margin = 0.1
    x_range = x_max - x_min
    y_range = y_max - y_min
    x_min -= x_range * margin
    x_max += x_range * margin
    y_min -= y_range * margin
    y_max += y_range * margin
    
    return x_min, x_max, y_min, y_max

def find_optimal_viewpoint(X_emb, labels):
    """silhouette_score를 사용하여 최적의 시점을 찾는 함수"""
    best_score = -1
    best_azim = 0
    best_elev = 0
    
    # 격자 탐색을 위한 각도 범위 설정
    azim_range = np.arange(0, 360, 45)
    elev_range = np.arange(-90, 91, 45)
    
    for azim in azim_range:
        for elev in elev_range:
            # 3D 좌표를 2D로 투영
            x = X_emb[:, 0] * np.cos(np.radians(azim)) * np.cos(np.radians(elev))
            y = X_emb[:, 1] * np.sin(np.radians(azim)) * np.cos(np.radians(elev))
            z = X_emb[:, 2] * np.sin(np.radians(elev))
            
            # 투영된 2D 좌표
            proj = np.column_stack((x, y))
            
            # silhouette score 계산
            score = silhouette_score(proj, labels)
            
            if score > best_score:
                best_score = score
                best_azim = azim
                best_elev = elev
    
    return best_azim, best_elev

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
    ax.tick_params(axis='both', which='major', labelsize=10)
    
    # 제목 설정
    ax.set_title(title, fontsize=16, pad=20)
    
    # legend 추가
    ax.legend(handles=legend_elements, 
             loc='upper right',
             bbox_to_anchor=(1.15, 1),
             fontsize=8)

def create_heatmaps_twohop(id_data, ood_data, output_dir, vmax, causal_input_pattern, args):
    """히트맵을 생성하는 메인 함수"""
    # 출력 디렉토리가 없으면 생성
    os.makedirs(output_dir, exist_ok=True)
    
    # vmin 설정
    vmin = 0
    cmap = sns.light_palette("seagreen", as_cmap=True)
    # cmap = sns.color_palette("viridis", as_cmap=True)
    
    # 전체 figure 생성
    fig_width = 22.5
    fig_height = 4
    fig = plt.figure(figsize=(fig_width, fig_height), constrained_layout=True)
    
    # 전체 figure를 3개의 영역으로 나누기 위한 GridSpec
    outer_gs = gridspec.GridSpec(1, 3, width_ratios=[20,3.5,30], wspace=0, hspace=0)
    
    # 파란색 테두리 추가
    for i in range(3):
        # 각 영역의 subplot 위치 계산
        ax = fig.add_subplot(outer_gs[i])
        bbox = ax.get_position()
        
        # 파란색 테두리 추가 (약간의 여백 추가)
        rect = plt.Rectangle(
            (bbox.x0, bbox.y0),
            bbox.width,
            bbox.height,
            fill=False,
            edgecolor='blue',
            linewidth=3,
            zorder=10,
            transform=fig.transFigure
        )
        fig.add_artist(rect)
        
        # subplot 제거하지 않고 유지
        ax.set_visible(False)  # 대신 보이지 않게 설정

    # 첫 번째 영역: (1,5) 레이아웃
    gs1 = gridspec.GridSpecFromSubplotSpec(1, 5, subplot_spec=outer_gs[0], wspace=0.1, width_ratios=[1, 1, 1, 1, 0.1])
    axes1 = []
    for i in range(5):
        axes1.append(fig.add_subplot(gs1[i]))
    
    # 두 번째 영역: (1,3) 레이아웃
    gs2 = gridspec.GridSpecFromSubplotSpec(1, 3, subplot_spec=outer_gs[2])
    axes2 = []
    for i in range(3):
        if args.reduce_dim == 3:
            axes2.append(fig.add_subplot(gs2[i], projection='3d'))
        else:
            axes2.append(fig.add_subplot(gs2[i]))
    
    # detailed_grouping 판단
    if "default" in id_data:
        # detailed_grouping이 없는 경우
        sections = ["default"]
        section_titles = ["ID_test"]
    elif "low_cutoff" in id_data and "mid_cutoff" in id_data and "high_cutoff" in id_data:
        sections = ["low_cutoff", "mid_cutoff", "high_cutoff"]
        section_titles = ["k $<$ 3", "k $=$ 3", "k $>$ 3"]
    elif "low_cutoff" in id_data and "high_cutoff" in id_data:
        sections = ["low_cutoff", "high_cutoff"]
        section_titles = ["k $<$ 3", "k $\geq$ 3"]
    elif "2" in id_data and "3" in id_data and "4" in id_data:
        sections = ["2", "3", "4"]
        section_titles = ["k $=$ 2", "k $=$ 3", "k $=$ 4"]
    else:
        raise ValueError("Unknown detailed_grouping pattern in data")
    
    # ID Test heatmap 그리기
    for idx, (section, title) in enumerate(zip(sections, section_titles)):
        if section not in id_data:
            continue
        step_data = id_data[section]
        # 마지막 step의 데이터 사용
        steps = sorted(step_data.keys(), key=lambda x: float('inf') if x == "stepfinal_checkpoint" else int(x.replace("step","")))
        step = steps[-1]
        layer_data = step_data[step]
        ax = axes1[idx]
        
        # heatmap 데이터 준비
        layers = sorted(layer_data.keys(), key=lambda x: int(x.replace("layer","")), reverse=True)
        positions = ["pos0", "pos1", "pos2"]
        heatmap_data = np.zeros((len(layers), len(positions)))
        for i, layer in enumerate(layers):
            for j, pos in enumerate(positions):
                heatmap_data[i, j] = layer_data[layer][pos]
        
        # heatmap 생성
        sns.heatmap(heatmap_data, 
                   annot=False,
                   cmap=cmap,
                   vmin=vmin,
                   vmax=vmax,
                   xticklabels=["x1", "x2", "x3"],
                   yticklabels=[f"Layer {i}" for i in range(len(layers), 0, -1)] if idx == 0 else [],  # 가장 왼쪽 heatmap에만 y축 레이블 표시
                   cbar=False,
                   linewidths=0.5,
                   linecolor='lightgray',
                   ax=ax,
                   square=True)
        
        # y축 레이블 회전 (가로로 표시)
        if idx == 0:  # 첫 번째 heatmap에만 적용
            ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=13)
        
        # 최대값 표시
        max_value = np.max(heatmap_data)
        max_indices = np.where(heatmap_data == max_value)
        for i, j in zip(*max_indices):
            ax.add_patch(plt.Rectangle((j, i), 1, 1, fill=False, edgecolor='black', lw=2))
            ax.text(j + 0.5, i + 0.5, f'{max_value:.3f}', 
                   ha='center', va='center', 
                   color='black', fontweight='bold', fontsize=9)
        
        # tick 제거
        ax.tick_params(axis='both', which='both', length=0)
        
        # x축 레이블 폰트 크기 설정
        ax.set_xticklabels(ax.get_xticklabels(), fontsize=13)
        
        # 테두리 및 제목
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_color('black')
            spine.set_linewidth(1.0)
        ax.text(0.5, -0.15, title, 
               ha='center', va='center', 
               transform=ax.transAxes, 
               fontsize=16)
    
    # 점선을 figure 전체에 걸쳐 그리기 위해 figure 좌표계 사용
    # fig.add_artist(plt.Line2D([0.65, 0.65], [0.05, 0.97], color='gray', linestyle='--', alpha=0.5, transform=fig.transFigure))
    
    # OOD heatmap 그리기 (마지막 subplot)
    ood_ax = axes1[-2]
    # OOD 데이터에서 첫 번째 cutoff의 데이터 사용 (어떤 cutoff든 같은 값)
    ood_cutoff = next(iter(ood_data))
    ood_step_data = ood_data[ood_cutoff]
    # 마지막 step의 데이터 사용
    ood_steps = sorted(ood_step_data.keys(), key=lambda x: float('inf') if x == "stepfinal_checkpoint" else int(x.replace("step","")))
    ood_step = ood_steps[-1]
    ood_layer_data = ood_step_data[ood_step]
    
    # OOD heatmap 데이터 준비
    layers = sorted(ood_layer_data.keys(), key=lambda x: int(x.replace("layer","")), reverse=True)
    positions = ["pos0", "pos1", "pos2"]
    ood_heatmap_data = np.zeros((len(layers), len(positions)))
    for i, layer in enumerate(layers):
        for j, pos in enumerate(positions):
            ood_heatmap_data[i, j] = ood_layer_data[layer][pos]
    
    # OOD 최대값 찾기
    ood_max_value = np.max(ood_heatmap_data)
    ood_max_indices = np.where(ood_heatmap_data == ood_max_value)
    
    # OOD heatmap 생성
    sns.heatmap(ood_heatmap_data,
                annot=False,
                cmap=cmap,
                vmin=vmin,
                vmax=vmax,
                xticklabels=["x1", "x2", "x3"],
                yticklabels=[],
                cbar=False,
                linewidths=0.5,
                linecolor='lightgray',
                ax=ood_ax,
                square=True)
    
    # OOD 최대값 표시
    ood_max_value = np.max(ood_heatmap_data)
    ood_max_indices = np.where(ood_heatmap_data == ood_max_value)
    for i, j in zip(*ood_max_indices):
        ood_ax.add_patch(plt.Rectangle((j, i), 1, 1, fill=False, edgecolor='black', lw=2))
        ood_ax.text(j + 0.5, i + 0.5, f'{ood_max_value:.3f}',
                   ha='center', va='center',
                   color='black', fontweight='bold', fontsize=9)
    
    # OOD 스타일 설정
    ood_ax.tick_params(axis='both', which='both', length=0)
    ood_ax.set_xticklabels(ood_ax.get_xticklabels(), fontsize=13)
    for spine in ood_ax.spines.values():
        spine.set_visible(True)
        spine.set_color('black')
        spine.set_linewidth(1.0)
    ood_ax.text(0.5, -0.15, "OOD",
                ha='center', va='center',
                transform=ood_ax.transAxes,
                fontsize=16)
    
    # colorbar 추가
    cbar_ax = axes1[-1]
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=vmin, vmax=vmax))
    cbar = fig.colorbar(sm, cax=cbar_ax)
    cbar.ax.tick_params(labelsize=14)
    
    # IICG 레이블 추가
    cbar.ax.text(0.5, 1.05, 'IICG', ha='center', va='bottom', fontsize=16, transform=cbar.ax.transAxes)
    
    # PCA/t-SNE plot 그리기 (두 번째 영역)
    random.seed(0)
    
    # 모든 데이터에서 공통된 key 찾기
    all_keys = set()
    for id_test_file in args.id_test_files:
        X, bridging_list, _ = load_and_process_data(id_test_file)
        if X.shape[0] < 2:
            print("Fewer than 2 vectors => skip embedding.")
            return
        all_keys.update(bridging_list)
    
    # 공통된 key 선택 및 색상 매핑
    chosen_keys = [k for k in all_keys if k != 'unknown']
    chosen_keys.sort()
    random.shuffle(chosen_keys)
    chosen_keys = chosen_keys[:args.pca_m]  # pca_m 개수만큼 선택
    print("\n선택된 keys:", chosen_keys)
    
    palette = sns.color_palette("hls", len(chosen_keys))
    key2color = {k: palette[i] for i, k in enumerate(chosen_keys)}
    
    # 데이터 범위 계산을 위한 초기화
    all_x_min, all_x_max = float('inf'), float('-inf')
    all_y_min, all_y_max = float('inf'), float('-inf')
    if args.reduce_dim == 3:
        all_z_min, all_z_max = float('inf'), float('-inf')
    
    # 선택된 key에 해당하는 데이터만 사용하여 범위 계산
    for id_test_file in args.id_test_files:
        X, bridging_list, _ = load_and_process_data(id_test_file)
        X_emb = compute_embedding(X, args.reduce_method, args.reduce_dim)
        
        # 선택된 key에 해당하는 데이터만 필터링
        mask = [bkey in chosen_keys for bkey in bridging_list]
        if not any(mask):
            continue
            
        filtered_X_emb = X_emb[mask]
        
        # 전체 데이터의 범위 업데이트
        all_x_min = min(all_x_min, filtered_X_emb[:, 0].min())
        all_x_max = max(all_x_max, filtered_X_emb[:, 0].max())
        all_y_min = min(all_y_min, filtered_X_emb[:, 1].min())
        all_y_max = max(all_y_max, filtered_X_emb[:, 1].max())
        if args.reduce_dim == 3:
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
    
    if args.reduce_dim == 3:
        z_range = all_z_max - all_z_min
        all_z_min -= z_range * margin
        all_z_max += z_range * margin

    # 실제 plot 그리기
    for idx, id_test_file in enumerate(args.id_test_files):
        ax = axes2[idx]
        X, bridging_list, grouped_vectors = load_and_process_data(id_test_file)
        if X.shape[0] < 2:
            continue
            
        X_emb = compute_embedding(X, args.reduce_method, args.reduce_dim)
        
        # 각 key별 데이터 포인트 개수 출력
        print(f"\nPlot {idx + 1} ({section_titles[idx]}) 데이터 분포:")
        for key in key2color:
            count = bridging_list.count(key)
            print(f"{key}: {count} points")
        
        if args.reduce_dim == 3:
            # 3D plot 생성
            plot_3d_scatter(ax, X_emb, bridging_list, key2color, 
                          f"{args.reduce_method.upper()} 3D: {section_titles[idx]}")
            # 모든 subplot에 동일한 범위 적용
            ax.set_xlim(all_x_min, all_x_max)
            ax.set_ylim(all_y_min, all_y_max)
            ax.set_zlim(all_z_min, all_z_max)
        else:
            # 2D plot 그리기
            legend_elements = []
            for key in key2color:
                # 해당 key를 가진 데이터 포인트의 인덱스 찾기
                indices = [i for i, bkey in enumerate(bridging_list) if bkey == key]
                if not indices:
                    continue
                    
                # scatter plot 생성
                scatter = ax.scatter(X_emb[indices,0], X_emb[indices,1], 
                                   color=key2color[key], s=20, alpha=0.8,
                                   label=key)
                legend_elements.append(scatter)
            
            ax.set_title(f"{args.reduce_method.upper()} {args.reduce_dim}D: {section_titles[idx]}", fontsize=16)
            ax.set_xlim(all_x_min, all_x_max)
            ax.set_ylim(all_y_min, all_y_max)
            
            # legend 추가
            ax.legend(handles=legend_elements, 
                     loc='upper right',
                     bbox_to_anchor=(1.15, 1),
                     fontsize=8)

    # 전체 figure 저장
    # plt.tight_layout()
    output_path = os.path.join(output_dir, f"twohop_combined.pdf")
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()
    print(f"Saved combined plot to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Compute Within–Between_all difference and save sorted JSON")
    parser.add_argument(
        "--root_dir", "-r", required=True,
        help="similarity_metrics_layer*.txt 들이 있는 최상위 디렉토리"
    )
    parser.add_argument('--reduce_method', type=str, default='pca',
                        choices=['pca','tsne'],
                        help="Which method => pca or tsne")
    parser.add_argument('--reduce_dim', type=int, default=2)
    parser.add_argument('--id_test_files', nargs='+',
                        help="ID Test JSON 파일 경로들 (low_cutoff와 high_cutoff에 대한 파일 경로를 순서대로 입력)")
    parser.add_argument('--pca_m', type=int, default=5)
    parser.add_argument('--pca_n', type=int, default=20)
    parser.add_argument(
        "--causal_input", "-c", default=None,
        help="Causal tracing JSON 파일 경로"
    )
    parser.add_argument(
        "--output_dir", "-d", default="heatmaps",
        help="heatmap PDF 파일들을 저장할 디렉토리 (기본: heatmaps)"
    )
    args = parser.parse_args()
    
    assert "results" in args.root_dir, "It should be results directory"

    # 데이터셋 타입 판단
    if "2-hop" not in args.root_dir:
        raise ValueError(f"Unknown dataset type in path: {args.root_dir}")

    # 모든 섹션의 데이터를 먼저 수집하여 전체 최대값 찾기
    all_values = []
    all_data = {}
    for section in ["ID Test", "OOD"]:
        raw = collect_diffs_twohop(args.root_dir, section)
        sorted_data = sort_nested_twohop(raw)
        all_data[section] = sorted_data
        
        # 각 섹션의 모든 값 수집
        for coverage, step_data in sorted_data.items():
            for step, layer_data in step_data.items():
                for layer, pos_data in layer_data.items():
                    for pos, value in pos_data.items():
                        if isinstance(value, (int, float)):
                            all_values.append(value)
                            
    print(all_data)
    
    # 전체 데이터의 최대값 계산
    vmax = max(all_values)
    
    output_dir_path = os.path.join(args.output_dir, args.root_dir.split("/")[-1])
    
    # ID Test와 OOD 데이터를 하나의 figure로 그리기
    create_heatmaps_twohop(all_data["ID Test"], all_data["OOD"], output_dir_path, vmax, args.causal_input, args)

if __name__ == "__main__":
    main()