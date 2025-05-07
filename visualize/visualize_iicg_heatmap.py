import argparse
import os
import glob
import re
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns

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
    (cutoff_type, f, layer, pos, step) 별로 차이를 수집합니다.
    """
    result = {}
    # 파일 경로에서 f, cutoff_type, (layer,pos), step 추출용 정규식
    re_f         = re.compile(r"[\\/](f\d+)[\\/]")
    re_cutoff    = re.compile(r"/(low_cutoff|high_cutoff)/")
    re_layer_pos = re.compile(r"/\((\d+),(\d+)\)/")
    re_step      = re.compile(r"/step(\d+)/")
    
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

def collect_diffs_nontree(root_dir, section):
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

def sort_nested_twohop(data):
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
    for cutoff in ["low_cutoff", "high_cutoff"]:
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

def sort_nested_nontree(data):
    """
    nontree 데이터셋을 위한 정렬 함수
    f1과 f4에 해당하는 데이터를 각각 따로 처리하고,
    layer1, layer2… 순으로,
    각 layer 안의 pos1, pos2… 순으로 정렬된 새 dict 반환.
    마지막 step의 데이터만 남깁니다.
    """
    out = {}
    # f1과 f4 데이터 처리
    for f_key in ["f1", "f4"]:
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

def create_heatmaps_twohop(id_data, ood_data, output_dir, vmax):
    """
    2-hop 데이터셋을 위한 ID Test와 OOD heatmap을 하나의 figure에 그립니다.
    ID Test는 low_cutoff와 high_cutoff별로 다른 heatmap을, OOD는 하나의 heatmap만 표시합니다.
    """
    # 출력 디렉토리가 없으면 생성
    os.makedirs(output_dir, exist_ok=True)
    
    # vmin 설정
    vmin = 0
    
    cmap = sns.light_palette("seagreen", as_cmap=True)
    # cmap = sns.color_palette("viridis", as_cmap=True)
    
    # 전체 figure 생성 (ID Test 2개 + OOD 1개)
    fig_width = 2.2 * 3  # ID Test 2개 + OOD 1개
    fig_height = 5
    fig = plt.figure(figsize=(fig_width, fig_height))
    gs = gridspec.GridSpec(1, 3, width_ratios=[1, 1, 1], wspace=0.0)
    axes = [plt.subplot(gs[i]) for i in range(3)]
    
    # cutoff를 순서대로 정렬
    sorted_cutoffs = ["low_cutoff", "high_cutoff"]
    
    # ID Test heatmap 그리기
    for idx, cutoff in enumerate(sorted_cutoffs):
        if cutoff not in id_data:
            continue
        layer_data = id_data[cutoff]
        ax = axes[idx]
        
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
        title = "k < 3" if cutoff == "low_cutoff" else "k >= 3"
        ax.text(0.5, -0.2, title, 
               ha='center', va='center', 
               transform=ax.transAxes, 
               fontsize=16)
    
    # 점선을 figure 전체에 걸쳐 그리기 위해 figure 좌표계 사용
    fig.add_artist(plt.Line2D([0.65, 0.65], [0.05, 0.97], color='gray', linestyle='--', alpha=0.5, transform=fig.transFigure))
    
    # OOD heatmap 그리기 (마지막 subplot)
    ood_ax = axes[2]
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
    ood_ax.text(0.5, -0.2, "OOD",
                ha='center', va='center',
                transform=ood_ax.transAxes,
                fontsize=16)
    
    # 전체 figure에 대한 colorbar 추가
    cbar_ax = fig.add_axes([0.97, 0.23, 0.02, 0.7])
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=vmin, vmax=vmax))
    cbar = fig.colorbar(sm, cax=cbar_ax)
    cbar.ax.tick_params(labelsize=14)
    
    # IICG 레이블 추가
    cbar.ax.text(0.5, 1.05, 'IICG', ha='center', va='bottom', fontsize=16, transform=cbar.ax.transAxes)
    
    # 전체 figure 조정
    plt.subplots_adjust(wspace=0.001)  # subplot 간의 가로 간격을 매우 좁게 설정
    plt.tight_layout(rect=[0, 0, 0.95, 1], w_pad=0.005)
    
    # PDF로 저장
    output_path = os.path.join(output_dir, f"twohop_collapse_analysis.pdf")
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()
    print(f"Saved combined heatmap to {output_path}")

def create_heatmaps_nontree(id_data, ood_data, output_dir, vmax):
    """
    nontree 데이터셋을 위한 ID Test와 OOD heatmap을 하나의 figure에 그립니다.
    f1과 f4의 히트맵을 각각 하나의 figure에 표시합니다.
    """
    # 출력 디렉토리가 없으면 생성
    os.makedirs(output_dir, exist_ok=True)
    
    # vmin 설정
    vmin = 0
    
    cmap = sns.light_palette("seagreen", as_cmap=True)
    
    # 전체 figure 생성 (f1의 ID Test/OOD + f4의 ID Test/OOD)
    fig_width = 2.2 * 4  # f1의 ID Test/OOD + f4의 ID Test/OOD
    fig_height = 5
    fig = plt.figure(figsize=(fig_width, fig_height))
    gs = gridspec.GridSpec(1, 4, width_ratios=[1, 1, 1, 1], wspace=0.0)
    axes = [plt.subplot(gs[i]) for i in range(4)]
    
    # f1과 f4 각각에 대해 처리
    for idx, f_key in enumerate(["f1", "f4"]):
        if f_key not in id_data or f_key not in ood_data:
            continue
            
        # ID Test heatmap 그리기
        id_ax = axes[idx * 2]
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
        id_ax.text(0.5, -0.2, "ID Test",
                    ha='center', va='center',
                    transform=id_ax.transAxes,
                    fontsize=16)
        
        # OOD heatmap 그리기
        ood_ax = axes[idx * 2 + 1]
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
        ood_ax.text(0.5, -0.2, "OOD",
                    ha='center', va='center',
                    transform=ood_ax.transAxes,
                    fontsize=16)
    
    # f1과 f4 사이에 세로 점선 추가 (figure 전체에 걸쳐)
    fig.add_artist(plt.Line2D([0.5, 0.5], [0.05, 0.97], color='gray', linestyle='--', alpha=0.5, transform=fig.transFigure))
    
    # f1과 f4 텍스트 추가
    fig.text(0.25, -0.05, "b", ha='center', va='center', fontsize=16)
    fig.text(0.75, -0.05, "(b,x2)", ha='center', va='center', fontsize=16)
    
    # 전체 figure에 대한 colorbar 추가
    cbar_ax = fig.add_axes([0.97, 0.23, 0.02, 0.7])
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=vmin, vmax=vmax))
    cbar = fig.colorbar(sm, cax=cbar_ax)
    cbar.ax.tick_params(labelsize=14)
    
    # IICG 레이블 추가
    cbar.ax.text(0.5, 1.05, 'IICG', ha='center', va='bottom', fontsize=16, transform=cbar.ax.transAxes)
    
    # 전체 figure 조정
    plt.subplots_adjust(wspace=0.001)  # subplot 간의 가로 간격을 매우 좁게 설정
    plt.tight_layout(rect=[0, 0, 0.95, 1], w_pad=0.005)
    
    # PDF로 저장
    output_path = os.path.join(output_dir, f"nontree_collapse_analysis.pdf")
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()
    print(f"Saved combined heatmap to {output_path}")

def main():
    parser = argparse.ArgumentParser(description="Compute Within–Between_all difference and save sorted JSON")
    parser.add_argument(
        "--root_dir", "-r", required=True,
        help="similarity_metrics_layer*.txt 들이 있는 최상위 디렉토리"
    )
    parser.add_argument(
        "--output", "-o", default="result.json",
        help="저장할 JSON 파일 이름 (기본: result.json)"
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
        raw = collect_diffs_fn(args.root_dir, section)
        print(raw)
        sorted_data = sort_nested_fn(raw)
        print(sorted_data)
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
    # vmax=1
    
    # ID Test와 OOD 데이터를 하나의 figure로 그리기
    create_heatmaps_fn(all_data["ID Test"], all_data["OOD"], args.output_dir, vmax)
    
    # JSON 저장
    # for section in ["ID Test", "OOD"]:
    #     with open(f"{section.lower().replace(' ', '_')}_{args.output}", "w", encoding="utf-8") as outf:
    #         json.dump(all_data[section], outf, ensure_ascii=False, indent=4)
    #     print(f"Saved sorted JSON to {section.lower().replace(' ', '_')}_{args.output}")

if __name__ == "__main__":
    main()
