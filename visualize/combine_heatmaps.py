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

def collect_diffs_twohop(root_dir, section, detailed_grouping):
    """
    2-hop 데이터셋을 위한 재귀 탐색으로 모든 similarity_metrics_layer*.txt 파일을 찾아
    (cutoff_type, f, layer, pos, step) 별로 차이를 수집합니다.
    """
    result = {}
    # 파일 경로에서 f, cutoff_type, (layer,pos), step 추출용 정규식
    re_f          = re.compile(r"[\\/](f\d+)[\\/]")
    if detailed_grouping:
        re_cutoff = re.compile(r"/covered_(\d+)/")
    else:
        re_cutoff = re.compile(r"/(low_cutoff|high_cutoff)/")
    re_layer_pos  = re.compile(r"/\((\d+),(\d+)\)/")
    re_step       = re.compile(r"/step(\d+)/")
    
    pattern = os.path.join(root_dir, "**", "similarity_metrics_layer*.txt")
    for path in glob.glob(pattern, recursive=True):
        with open(path, encoding="utf-8") as f:
            diff = extract_diff(f.read(), section)
        assert diff is not None, f"No diff found in file: {path}"

        # path 에서 metadata 추출
        fm = re_f.search(path)
        cm = re_cutoff.search(path)
        lp = re_layer_pos.search(path)
        sm = re_step.search(path)
        assert fm != None and sm != None, f"All metadata should be present: {path}, {fm}, {sm}"
        if not cm or not lp:
            continue    

        f_key        = fm.group(1)              # e.g. "f1"
        cutoff_key   = cm.group(1)              # e.g. "low_cutoff" or "high_cutoff"
        layer_idx    = lp.group(1)              # e.g. "6"
        pos_idx      = lp.group(2)              # e.g. "2"
        step_key     = "step" + sm.group(1)     # e.g. "step25000"
        layer_key    = "layer" + layer_idx
        pos_key      = "pos"   + pos_idx
        
        result.setdefault(cutoff_key, {})\
              .setdefault(f_key, {})\
              .setdefault(step_key, {})\
              .setdefault(layer_key, {})[pos_key] = diff
              
    return result

def collect_diffs_nontree(root_dir, section, detailed_grouping=None):
    """
    nontree 데이터셋을 위한 재귀 탐색으로 모든 similarity_metrics_layer*.txt 파일을 찾아
    (f, layer, pos, step) 별로 차이를 수집합니다.
    """
    result = {}
    # 파일 경로에서 f, (layer,pos), step 추출용 정규식
    re_f         = re.compile(r"[\\/](f\d+)[\\/]")
    re_layer_pos = re.compile(r"/\((\d+),(\d+)\)/")
    re_step      = re.compile(r"/step(\d+)/")
    
    pattern = os.path.join(root_dir, "**", "similarity_metrics_layer*.txt")
    for path in glob.glob(pattern, recursive=True):
        with open(path, encoding="utf-8") as f:
            diff = extract_diff(f.read(), section)
        assert diff is not None, f"No diff found in file: {path}"

        # path 에서 metadata 추출
        fm = re_f.search(path)
        lp = re_layer_pos.search(path)
        sm = re_step.search(path)
        assert fm != None and sm != None, f"All metadata should be present: {path}, {fm}, {sm}"
        if not lp:  # layer가 prob이거나 logit인 경우 무시
            continue    

        f_key        = fm.group(1)              # e.g. "f1"
        layer_idx    = lp.group(1)              # e.g. "6"
        pos_idx      = lp.group(2)              # e.g. "2"
        step_key     = "step" + sm.group(1)     # e.g. "step25000"
        layer_key    = "layer" + layer_idx
        pos_key      = "pos"   + pos_idx
        
        result.setdefault(f_key, {})\
              .setdefault(step_key, {})\
              .setdefault(layer_key, {})[pos_key] = diff
              
    return result

def sort_nested_twohop(data, detailed_grouping):
    """
    2-hop 데이터셋을 위한 정렬 함수
    low_cutoff, high_cutoff 순으로,
    f1에 해당하는 데이터만,
    각 cutoff 안의 layer1, layer2… 순으로,
    각 layer 안의 pos1, pos2… 순으로 정렬된 새 dict 반환.
    마지막 step의 데이터만 남깁니다.
    """
    out = {}
    # cutoff 키 정렬
    if detailed_grouping:
        cutoff_keys = sorted(data.keys(), key=lambda x: int(x))
    else:
        cutoff_keys = ["low_cutoff", "high_cutoff"]
    for cutoff in cutoff_keys:
        if cutoff not in data:
            continue
        out[cutoff] = {}
        fdict = data[cutoff]
        # f1 데이터만 처리
        if "f1" not in fdict:
            continue
        stepdict = fdict["f1"]
        # 마지막 step 선택
        last_step = sorted(stepdict.keys(), key=lambda x: int(x.replace("step","")))[-1]
        ldict = stepdict[last_step]
        # layer 키 정렬
        for layer in sorted(ldict.keys(), key=lambda x: int(x.replace("layer",""))):
            out[cutoff][layer] = {}
            pdict = ldict[layer]
            # pos 키 정렬
            for pos in sorted(pdict.keys(), key=lambda x: int(x.replace("pos",""))):
                out[cutoff][layer][pos] = pdict[pos]
    return out

def sort_nested_nontree(data, detailed_grouping=None):
    """
    nontree 데이터셋을 위한 정렬 함수
    f1과 f4에 해당하는 데이터를 각각 따로 처리하고,
    layer1, layer2… 순으로,
    각 layer 안의 pos1, pos2… 순으로 정렬된 새 dict 반환.
    마지막 step의 데이터만 남깁니다.
    """
    out = {}
    # f1과 f4 데이터 처리
    for f_key in ["f1", "f3", "f4"]:
        if f_key not in data:
            continue
        out[f_key] = {}
        stepdict = data[f_key]
        # 마지막 step 선택
        last_step = sorted(stepdict.keys(), key=lambda x: int(x.replace("step","")))[-1]
        ldict = stepdict[last_step]
        # layer 키 정렬
        for layer in sorted(ldict.keys(), key=lambda x: int(x.replace("layer",""))):
            out[f_key][layer] = {}
            pdict = ldict[layer]
            # pos 키 정렬
            for pos in sorted(pdict.keys(), key=lambda x: int(x.replace("pos",""))):
                out[f_key][layer][pos] = pdict[pos]
    return out

def extract_metrics(data, section):
    """
    JSON 데이터에서 특정 섹션의 both_layerN 값을 추출하고 퍼센트로 변환합니다.
    """
    result = {}
    checkpoint = next(iter(data.keys()))  # 첫 번째 체크포인트 사용
    section_data = data[checkpoint][section]
    
    for pos in section_data.keys():
        pos_data = section_data[pos]
        result[pos] = {}
        for layer in range(1, 8):  # layer1부터 layer7까지
            layer_key = f"layer{layer}"
            metric_key = f"both_{layer_key}"
            if not metric_key in pos_data:
                AssertionError(f"There is no result for {section}-({layer}, {pos})")
            # both_layerN 값을 퍼센트로 변환
            result[pos][layer_key] = (pos_data[metric_key] / pos_data["sample_num"]) * 100
    
    return result

def create_heatmaps_twohop(id_data, ood_data, output_dir, vmax, causal_input_pattern, args):
    """
    2-hop 데이터셋을 위한 ID Test와 OOD heatmap을 하나의 figure에 그립니다.
    ID Test는 low_cutoff와 high_cutoff별로 다른 heatmap을, OOD는 하나의 heatmap만 표시합니다.
    추가로 causal tracing 히트맵도 함께 표시합니다.
    figure_paths: IICG와 Causal Tracing 결과 사이에 삽입할 두 개의 figure 경로 리스트
    """
    # 출력 디렉토리가 없으면 생성
    os.makedirs(output_dir, exist_ok=True)
    
    # vmin 설정
    vmin = 0
    
    cmap = sns.light_palette("seagreen", as_cmap=True)
    
    # 전체 figure 생성
    if args.detailed_grouping:
        fig_width = 24
    else:
        fig_width = 19.5
    fig_height = 6
    fig = plt.figure(figsize=(fig_width, fig_height), constrained_layout=True)
    
    # 전체 figure를 3개의 영역으로 나누기 위한 GridSpec
    outer_gs = gridspec.GridSpec(1, 5, width_ratios=[13,2.7,4,1.5,12], wspace=0, hspace=0)
    
    # # 파란색 테두리 추가
    # for i in range(5):
    #     # 각 영역의 subplot 위치 계산
    #     ax = fig.add_subplot(outer_gs[i])
    #     bbox = ax.get_position()
        
    #     # 파란색 테두리 추가 (약간의 여백 추가)
    #     rect = plt.Rectangle(
    #         (bbox.x0, bbox.y0),
    #         bbox.width,
    #         bbox.height,
    #         fill=False,
    #         edgecolor='blue',
    #         linewidth=3,
    #         zorder=10,
    #         transform=fig.transFigure
    #     )
    #     fig.add_artist(rect)
        
    #     # subplot 제거하지 않고 유지
    #     ax.set_visible(False)  # 대신 보이지 않게 설정
    
    # 첫 번째 영역: (1,3) 레이아웃
    if args.detailed_grouping:
        gs1 = gridspec.GridSpecFromSubplotSpec(1, 5, subplot_spec=outer_gs[0], wspace=0.1, width_ratios=[1, 1, 1, 1, 0.1])
        axes1 = []
        for i in range(5):
            axes1.append(fig.add_subplot(gs1[i]))
    else:
        gs1 = gridspec.GridSpecFromSubplotSpec(1, 4, subplot_spec=outer_gs[0], wspace=0.1, width_ratios=[1, 1, 1, 0.1])
        axes1 = []
        for i in range(4):
            axes1.append(fig.add_subplot(gs1[i]))
    
    # 두 번째 영역: (2,1) 레이아웃
    gs2 = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=outer_gs[2], hspace=0.3)
    axes2 = []
    for i in range(2):
        axes2.append(fig.add_subplot(gs2[i]))
    
    # 세 번째 영역: (1,3) 레이아웃
    if args.detailed_grouping:
        gs3 = gridspec.GridSpecFromSubplotSpec(1, 4, subplot_spec=outer_gs[4], wspace=0.1, width_ratios=[1, 1, 1, 0.1])
        axes3 = []
        for i in range(4):
            axes3.append(fig.add_subplot(gs3[i]))
    else:
        gs3 = gridspec.GridSpecFromSubplotSpec(1, 3, subplot_spec=outer_gs[4], wspace=0.1, width_ratios=[1, 1, 0.1])
        axes3 = []
        for i in range(3):
            axes3.append(fig.add_subplot(gs3[i]))
    
    # cutoff를 순서대로 정렬
    if args.detailed_grouping:
        sorted_cutoffs = ["2", "3", "4"]
    else:
        sorted_cutoffs = ["low_cutoff", "high_cutoff"]
    
    # ID Test heatmap 그리기 (첫 번째 영역)
    for idx, cutoff in enumerate(sorted_cutoffs):
        if cutoff not in id_data:
            continue
        layer_data = id_data[cutoff]
        ax = axes1[idx]
        
        # 데이터를 numpy 배열로 변환
        layers = sorted(layer_data.keys(), key=lambda x: int(x.replace("layer","")), reverse=True)
        positions = ["pos0", "pos1", "pos2"]
        
        # heatmap 데이터 준비
        heatmap_data = np.zeros((len(layers), len(positions)))
        for i, layer in enumerate(layers):
            for j, pos in enumerate(positions):
                heatmap_data[i, j] = layer_data[layer][pos]
        
        # 최대값 찾기
        max_value = np.max(heatmap_data)
        max_indices = np.where(heatmap_data == max_value)
        
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
        
        # 최대값 셀에만 값 표시
        for i, j in zip(*max_indices):
            ax.add_patch(plt.Rectangle((j, i), 1, 1, fill=False, edgecolor='black', lw=2))
            ax.text(j + 0.5, i + 0.5, f'{max_value:.3f}', 
                   ha='center', va='center', 
                   color='black', fontweight='bold', fontsize=10)
        
        # tick 제거
        ax.tick_params(axis='both', which='both', length=0)
        
        # x축 레이블 폰트 크기 설정
        ax.set_xticklabels(ax.get_xticklabels(), fontsize=13)
        
        # subplot 영역 테두리 추가
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_color('black')
            spine.set_linewidth(1.0)
        
        # 제목을 heatmap 아래쪽에 추가
        if args.detailed_grouping:
            title = f"k={cutoff}"
        else:
            title = "k < 3" if cutoff == "low_cutoff" else "k >= 3"
        ax.text(0.5, -0.1, title, 
            ha='center', va='center', 
            transform=ax.transAxes, 
            fontsize=16)
    
    # 점선을 figure 전체에 걸쳐 그리기 위해 figure 좌표계 사용
    # fig.add_artist(plt.Line2D([0.65, 0.65], [0.05, 0.97], color='gray', linestyle='--', alpha=0.5, transform=fig.transFigure))
    
    # OOD heatmap 그리기 (마지막 subplot)
    ood_ax = axes1[-2]
    # OOD 데이터에서 첫 번째 cutoff의 데이터 사용 (어떤 cutoff든 같은 값)
    ood_cutoff = next(iter(ood_data))
    ood_layer_data = ood_data[ood_cutoff]
    
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
                yticklabels=[],  # OOD heatmap에는 y축 레이블 표시하지 않음
                cbar=False,
                linewidths=0.5,
                linecolor='lightgray',
                ax=ood_ax,
                square=True)
    
    # OOD 최대값 셀에만 값 표시
    for i, j in zip(*ood_max_indices):
        ood_ax.add_patch(plt.Rectangle((j, i), 1, 1, fill=False, edgecolor='black', lw=2))
        ood_ax.text(j + 0.5, i + 0.5, f'{ood_max_value:.3f}',
                   ha='center', va='center',
                   color='black', fontweight='bold', fontsize=10)
    
    # OOD tick 제거
    ood_ax.tick_params(axis='both', which='both', length=0)
    ood_ax.tick_params(axis='y', which='both', labelsize=13)  # y축 레이블 크기 설정
    
    # OOD x축 레이블 폰트 크기 설정
    ood_ax.set_xticklabels(ood_ax.get_xticklabels(), fontsize=13)
    
    # subplot 영역 테두리 추가
    for spine in ood_ax.spines.values():
        spine.set_visible(True)
        spine.set_color('black')
        spine.set_linewidth(1.0)
    
    # OOD 제목 추가
    ood_ax.text(0.5, -0.1, "OOD",
                ha='center', va='center',
                transform=ood_ax.transAxes,
                fontsize=16)
    
    # 첫 번째 영역(gs1)에 대한 colorbar 추가
    cbar_ax = axes1[-1]  # colorbar는 새로 추가된 네 번째 위치
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=vmin, vmax=vmax))
    cbar = fig.colorbar(sm, cax=cbar_ax)
    cbar.ax.tick_params(labelsize=14)
    
    # IICG 레이블 추가
    cbar.ax.text(0.5, 1.05, 'IICG', ha='center', va='bottom', fontsize=16, transform=cbar.ax.transAxes)
    
    # 추가 figure 삽입 (두 번째 영역)
    if args.detailed_grouping:
        cutoffs = ["4", "2"]
    else:
        cutoffs = ["high_cutoff", "low_cutoff"]
    for idx, (cutoff, id_test_file) in enumerate(zip(cutoffs, args.id_test_files)):
        ax = axes2[idx]
        
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
                    # store (vector, input_text) for multi random references
                    grouped_instances[target].append((vector, instance['input_text']))
                    all_vectors.append(vector)
        
        bridging_list = []
        all_points = []

        for bridging_key, vlist in grouped_vectors.items():
            for v in vlist:
                bridging_list.append(bridging_key)
                all_points.append(v)
        assert len(all_points) > 0, "No vectors found"
        
        X = np.array(all_points)
        assert X.shape[0] >= 2, "Fewer than 2 vectors => skip embedding."
        if X.shape[0] < 2:
            print("Fewer than 2 vectors => skip embedding.")
            return
        
        # 2) compute embedding
        print(f"Compute {args.reduce_method.upper()} with dim=2 on {X.shape[0]} points...")
        if args.reduce_method == 'pca':
            reducer = PCA(n_components=2)
            X_emb = reducer.fit_transform(X)
        else:
            # t-SNE
            reducer = TSNE(n_components=2, perplexity=30, 
                        n_iter=1000, verbose=1)
            X_emb = reducer.fit_transform(X)
        
        # PCA 계산
        reducer = PCA(n_components=2)
        X_emb = reducer.fit_transform(X)
        
        group_sizes = defaultdict(int)
        for b in bridging_list:
            group_sizes[b] += 1
        chosen_keys = [k for k in group_sizes if k!='unknown' and group_sizes[k]>=args.pca_n]
        random.shuffle(chosen_keys)
        chosen_keys = chosen_keys[:args.pca_m]
        
        # We'll skip bridging keys not in chosen_keys (or color them lightly if you prefer).
        # For minimal changes => skip them.
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
        # ax.legend(handles=handles, bbox_to_anchor=(1.05,1), loc='upper left')
        if args.detailed_grouping:
            ax.set_title(f"{args.reduce_method.upper()} 2D: k={cutoff}")
        else:
            ax.set_title(f"{args.reduce_method.upper()} 2D: {'k >= 3' if idx == 0 else 'k < 3'}")

        # # 제목 추가
        # title = "k < 3" if idx == 0 else "k >=3"
        # ax.text(0.5, -0.1, title, 
        #        ha='center', va='center', 
        #        transform=ax.transAxes, 
        #        fontsize=16)
        
        # 축 레이블 추가
        # ax.set_xlabel("PC1")
        # ax.set_ylabel("PC2")
    
    # Causal Tracing 히트맵 그리기 (세 번째 영역)
    vmax = 100
    
    with open(causal_input_pattern, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # 각 섹션에 대해 히트맵 생성
    if args.detailed_grouping:
        sections = ["test_inferred_covered_2", "test_inferred_covered_3", "test_inferred_covered_4"]
        section_titles = ["k=2", "k=3", "k=4"]
    else:
        sections = ["test_inferred_low_cutoff", "test_inferred_high_cutoff"]
        section_titles = ["k < 3", "k >= 3"]
    
    for idx, (section, title) in enumerate(zip(sections, section_titles)):
        ax = axes3[idx]
        
        # 데이터 추출
        section_data = extract_metrics(data, section)
        
        # 데이터를 numpy 배열로 변환
        positions = ["0", "1", "2"]
        layers = [f"layer{i}" for i in range(1, 8)]
        
        heatmap_data = np.zeros((len(layers), len(positions)))
        for i, layer in enumerate(layers):
            for j, pos in enumerate(positions):
                heatmap_data[i, j] = section_data[pos][layer]
        
        # 레이어 순서를 반대로 변경 (layer1이 맨 아래로)
        heatmap_data = np.flipud(heatmap_data)
        
        # heatmap 생성
        sns.heatmap(heatmap_data,
                    annot=True,  # 모든 셀의 값을 표시
                    fmt='.1f',   # 소수점 첫째 자리까지 표시
                    cmap=cmap,
                    vmin=vmin,
                    vmax=vmax,
                    xticklabels=["x1", "x2", "x3"],
                    yticklabels=[],
                    cbar=False,
                    linewidths=0.5,
                    linecolor='lightgray',
                    ax=ax,
                    square=True)
        
        # tick 제거
        ax.tick_params(axis='both', which='both', length=0)
        
        # x축 레이블 폰트 크기 설정
        ax.set_xticklabels(ax.get_xticklabels(), fontsize=13)
        
        # y축 레이블 회전 (가로로 표시)
        if idx == 0:  # 첫 번째 heatmap에만 적용
            ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=13)
        
        # subplot 영역 테두리 추가
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_color('black')
            spine.set_linewidth(1.0)
        
        # 제목 추가
        ax.text(0.5, -0.1, title,
                ha='center', va='center',
                transform=ax.transAxes,
                fontsize=16)
        
    # 세 번째 영역(gs3)에 대한 colorbar 추가
    cbar_ax = axes3[-1]  # colorbar는 새로 추가된 세 번째 위치
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=vmin, vmax=vmax))
    cbar = fig.colorbar(sm, cax=cbar_ax)
    cbar.ax.tick_params(labelsize=14)
    
    # 마지막에 tight_layout 적용 제거
    # plt.tight_layout()
    
    # Causal Tracing 레이블 추가
    cbar.ax.text(0.5, 1.05, 'Causal Tracing (%)', ha='center', va='bottom', fontsize=16, transform=cbar.ax.transAxes)
    
    # PDF로 저장
    output_path = os.path.join(output_dir, f"twohop_combined_heatmap.pdf")
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()
    print(f"Saved combined heatmap to {output_path}")

def create_heatmaps_nontree(id_data, ood_data, output_dir, vmax, causal_data, args):
    """
    nontree 데이터셋을 위한 ID Test와 OOD heatmap을 하나의 figure에 그립니다.
    f1과 f4의 히트맵을 각각 하나의 figure에 표시합니다.
    추가로 causal tracing 히트맵도 함께 표시합니다.
    figure_paths: IICG와 Causal Tracing 결과 사이에 삽입할 두 개의 figure 경로 리스트
    """
    # 출력 디렉토리가 없으면 생성
    os.makedirs(output_dir, exist_ok=True)
    
    # vmin 설정
    vmin = 0
    
    cmap = sns.light_palette("seagreen", as_cmap=True)
    
    # 전체 figure 생성
    fig_width = 20
    fig_height = 5
    fig = plt.figure(figsize=(fig_width, fig_height), constrained_layout=True)
    
    # 전체 figure를 3개의 영역으로 나누기 위한 GridSpec
    outer_gs = gridspec.GridSpec(1, 3, width_ratios=[9, 1, 3], wspace=0, hspace=0)
    
    # # 파란색 테두리 추가
    # for i in range(3):
    #     # 각 영역의 subplot 위치 계산
    #     ax = fig.add_subplot(outer_gs[i])
    #     bbox = ax.get_position()
        
    #     # 파란색 테두리 추가 (약간의 여백 추가)
    #     rect = plt.Rectangle(
    #         (bbox.x0, bbox.y0),
    #         bbox.width,
    #         bbox.height,
    #         fill=False,
    #         edgecolor='blue',
    #         linewidth=3,
    #         zorder=10,
    #         transform=fig.transFigure
    #     )
    #     fig.add_artist(rect)
        
    #     # subplot 제거하지 않고 유지
    #     ax.set_visible(False)  # 대신 보이지 않게 설정
    
    # 첫 번째 영역: (1,7) 레이아웃
    gs1 = gridspec.GridSpecFromSubplotSpec(1, 7, subplot_spec=outer_gs[0], wspace=0.1, width_ratios=[1, 1, 1, 1, 1, 1, 0.1])
    axes1 = []
    for i in range(7):
        axes1.append(fig.add_subplot(gs1[i]))
    
    # 두 번째 영역: (1,1) 레이아웃
    gs2 = gridspec.GridSpecFromSubplotSpec(1, 1, subplot_spec=outer_gs[2])
    axes2 = []
    for i in range(1):
        axes2.append(fig.add_subplot(gs2[i]))
    
    # 첫 번째 영역: (1,7) 레이아웃
    # f1, f4, f2 각각에 대해 처리
    for idx, f_key in enumerate(["f1", "f4", "f3"]):
        if f_key not in id_data or f_key not in ood_data:
            continue
            
        # ID Test heatmap 그리기
        id_ax = axes1[idx * 2]
        id_layer_data = id_data[f_key]
        
        # ID Test heatmap 데이터 준비
        layers = sorted(id_layer_data.keys(), key=lambda x: int(x.replace("layer","")), reverse=True)
        positions = ["pos0", "pos1", "pos2"]
        id_heatmap_data = np.zeros((len(layers), len(positions)))
        for i, layer in enumerate(layers):
            for j, pos in enumerate(positions):
                id_heatmap_data[i, j] = id_layer_data[layer][pos]
        
        # ID Test 최대값 찾기
        id_max_value = np.max(id_heatmap_data)
        id_max_indices = np.where(id_heatmap_data == id_max_value)
        
        # ID Test heatmap 생성
        sns.heatmap(id_heatmap_data,
                    annot=False,
                    cmap=cmap,
                    vmin=vmin,
                    vmax=vmax,
                    xticklabels=["x1", "x2", "x3"],
                    yticklabels=[f"Layer {i}" for i in range(len(layers), 0, -1)] if idx == 0 else [],  # 가장 왼쪽 heatmap에만 y축 레이블 표시
                    cbar=False,
                    linewidths=0.5,
                    linecolor='lightgray',
                    ax=id_ax,
                    square=True)
        
        # y축 레이블 회전 (가로로 표시)
        if idx == 0:  # 첫 번째 heatmap에만 적용
            id_ax.set_yticklabels(id_ax.get_yticklabels(), rotation=0, fontsize=13)
        
        # ID Test 최대값 셀에만 값 표시
        for i, j in zip(*id_max_indices):
            id_ax.add_patch(plt.Rectangle((j, i), 1, 1, fill=False, edgecolor='black', lw=2))
            id_ax.text(j + 0.5, i + 0.5, f'{id_max_value:.3f}',
                       ha='center', va='center',
                       color='black', fontweight='bold', fontsize=10)
        
        # ID Test tick 제거
        id_ax.tick_params(axis='both', which='both', length=0)
        
        # ID Test x축 레이블 폰트 크기 설정
        id_ax.set_xticklabels(id_ax.get_xticklabels(), fontsize=13)
        
        # subplot 영역 테두리 추가
        for spine in id_ax.spines.values():
            spine.set_visible(True)
            spine.set_color('black')
            spine.set_linewidth(1.0)
        
        # ID Test 제목 추가
        id_ax.text(0.5, -0.1, "ID Test",
                    ha='center', va='center',
                    transform=id_ax.transAxes,
                    fontsize=16)
        
        # OOD heatmap 그리기
        ood_ax = axes1[idx * 2 + 1]
        ood_layer_data = ood_data[f_key]
        
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
                    yticklabels=[],  # OOD heatmap에는 y축 레이블 표시하지 않음
                    cbar=False,
                    linewidths=0.5,
                    linecolor='lightgray',
                    ax=ood_ax,
                    square=True)
        
        # OOD 최대값 셀에만 값 표시
        for i, j in zip(*ood_max_indices):
            ood_ax.add_patch(plt.Rectangle((j, i), 1, 1, fill=False, edgecolor='black', lw=2))
            ood_ax.text(j + 0.5, i + 0.5, f'{ood_max_value:.3f}',
                       ha='center', va='center',
                       color='black', fontweight='bold', fontsize=10)
        
        # OOD tick 제거
        ood_ax.tick_params(axis='both', which='both', length=0)
        
        # OOD x축 레이블 폰트 크기 설정
        ood_ax.set_xticklabels(ood_ax.get_xticklabels(), fontsize=13)
        
        # subplot 영역 테두리 추가
        for spine in ood_ax.spines.values():
            spine.set_visible(True)
            spine.set_color('black')
            spine.set_linewidth(1.0)
        
        # OOD 제목 추가
        ood_ax.text(0.5, -0.1, "OOD",
                    ha='center', va='center',
                    transform=ood_ax.transAxes,
                    fontsize=16)
    # 첫 번째 영역(gs1)에 대한 colorbar 추가
    cbar_ax = axes1[-1]  # colorbar는 새로 추가된 네 번째 위치
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=vmin, vmax=vmax))
    cbar = fig.colorbar(sm, cax=cbar_ax)
    cbar.ax.tick_params(labelsize=14)
    
    # IICG 레이블 추가
    cbar.ax.text(0.5, 1.05, 'IICG', ha='center', va='bottom', fontsize=16, transform=cbar.ax.transAxes)
    
    # 각 히트맵 사이에 레이블 추가
    fig.text(0.2, -0.05, "b", ha='center', va='center', fontsize=16)
    fig.text(0.38, -0.05, "t2", ha='center', va='center', fontsize=16)
    fig.text(0.57, -0.05, "(b,t2)", ha='center', va='center', fontsize=16)
    
    # 히트맵 그룹 사이에 회색 세로 점선 추가
    fig.add_artist(plt.Line2D([0.298, 0.298], [-0.05, 0.95], color='gray', linestyle='--', alpha=0.5, transform=fig.transFigure))
    fig.add_artist(plt.Line2D([0.474, 0.474], [-0.05, 0.95], color='gray', linestyle='--', alpha=0.5, transform=fig.transFigure))
    
    # 두 번째 영역: (1,1) 레이아웃
    # assert len(args.id_test_files) == 1
    # ax = axes2[0]
    
    # id_test_file = args.id_test_files[0]
    # with open(id_test_file, 'r', encoding='utf-8') as f:
    #     id_test_data = json.load(f)
    
    
    
    
    
    
    
    
    # PDF로 저장
    output_path = os.path.join(output_dir, f"nontree_combined_heatmap.pdf")
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()
    print(f"Saved combined heatmap to {output_path}")

def main():
    parser = argparse.ArgumentParser(description="Compute Within–Between_all difference and save sorted JSON")
    parser.add_argument(
        "--root_dir", "-r", required=True,
        help="similarity_metrics_layer*.txt 들이 있는 최상위 디렉토리"
    )
    parser.add_argument("--detailed_grouping", action="store_true", help="Group covered_ examples by their specific coverage values")
    parser.add_argument('--reduce_method', type=str, default='pca',
                        choices=['pca','tsne'],
                        help="Which method => pca or tsne")
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

    # 데이터셋 타입 판단
    if "2-hop" in args.root_dir:
        dataset_type = "twohop"
        collect_diffs_fn = collect_diffs_twohop
        sort_nested_fn = sort_nested_twohop
        create_heatmaps_fn = create_heatmaps_twohop
    elif "nontree" in args.root_dir:
        dataset_type = "nontree"
        collect_diffs_fn = collect_diffs_nontree
        sort_nested_fn = sort_nested_nontree
        create_heatmaps_fn = create_heatmaps_nontree
    else:
        raise ValueError(f"Unknown dataset type in path: {args.root_dir}")

    # 모든 섹션의 데이터를 먼저 수집하여 전체 최대값 찾기
    all_values = []
    all_data = {}
    for section in ["ID Test", "OOD"]:
        raw = collect_diffs_fn(args.root_dir, section, args.detailed_grouping)
        print(raw) 
        sorted_data = sort_nested_fn(raw, args.detailed_grouping)
        all_data[section] = sorted_data
        
        # 각 섹션의 모든 값 수집
        if dataset_type == "twohop":
            for coverage, layer_data in sorted_data.items():
                for layer, pos_data in layer_data.items():
                    all_values.extend(pos_data.values())
        else:  # nontree
            for f_key, layer_data in sorted_data.items():
                for layer, pos_data in layer_data.items():
                    all_values.extend(pos_data.values())
    
    # 전체 데이터의 최대값 계산
    vmax = max(all_values)
    
    # ID Test와 OOD 데이터를 하나의 figure로 그리기
    create_heatmaps_fn(all_data["ID Test"], all_data["OOD"], args.output_dir, vmax, args.causal_input, args)

if __name__ == "__main__":
    main()