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
    (f, layer, pos, step) 별로 차이를 수집합니다.
    """
    result = {}
    re_f = re.compile(r"[\\/](f\d+)[\\/]")
    re_layer_pos = re.compile(r"/\((\d+),(\d+)\)/")
    re_step = re.compile(r"/step(\d+|final_checkpoint)/")
    
    pattern = os.path.join(root_dir, "**", "similarity_metrics_layer*.txt")
    for path in glob.glob(pattern, recursive=True):
        with open(path, encoding="utf-8") as f:
            diff = extract_diff(f.read(), section)
        assert diff is not None, f"No diff found in file: {path}"

        fm = re_f.search(path)
        lp = re_layer_pos.search(path)
        sm = re_step.search(path)
        assert fm != None and sm != None, f"All metadata should be present: {path}, {fm}, {sm}"
        if not lp:
            continue    

        f_key = fm.group(1)
        layer_idx = lp.group(1)
        pos_idx = lp.group(2)
        step_key = "step" + sm.group(1)
        layer_key = "layer" + layer_idx
        pos_key = "pos" + pos_idx
        
        result.setdefault(f_key, {})\
              .setdefault(step_key, {})\
              .setdefault(layer_key, {})[pos_key] = diff
              
    return result

def collect_diffs_nontree(root_dir, section):
    """
    nontree 데이터셋을 위한 재귀 탐색으로 모든 similarity_metrics_layer*.txt 파일을 찾아
    (f, layer, pos, step) 별로 차이를 수집합니다.
    """
    result = {}
    re_f = re.compile(r"[\\/](f\d+)[\\/]")
    re_layer_pos = re.compile(r"/\((?:(\d+)|(logit|prob)),(\d+)\)/")
    re_step = re.compile(r"/step(\d+|final_checkpoint)/")
    
    pattern = os.path.join(root_dir, "**", "similarity_metrics_layer*.txt")
    for path in glob.glob(pattern, recursive=True):
        with open(path, encoding="utf-8") as f:
            diff = extract_diff(f.read(), section)
        assert diff is not None, f"No diff found in file: {path}"

        fm = re_f.search(path)
        lp = re_layer_pos.search(path)
        sm = re_step.search(path)
        assert fm != None and sm != None, f"All metadata should be present: {path}, {fm}, {sm}"
        if not lp:
            continue    

        f_key = fm.group(1)
        layer_idx = lp.group(1) or lp.group(2)
        pos_idx = lp.group(3)
        step_key = "step" + sm.group(1)
        layer_key = "layer" + layer_idx if layer_idx.isdigit() else layer_idx
        pos_key = "pos" + pos_idx
        
        result.setdefault(f_key, {})\
              .setdefault(step_key, {})\
              .setdefault(layer_key, {})[pos_key] = diff
              
    return result

def sort_nested_data(data):
    """
    데이터를 정렬하여 반환합니다.
    마지막 step의 데이터만 사용합니다.
    """
    out = {}
    for f_key in sorted(data.keys()):
        out[f_key] = {}
        stepdict = data[f_key]
        last_step = sorted(stepdict.keys(), key=lambda x: float('inf') if x == "stepfinal_checkpoint" else int(x.replace("step","")))[-1]
        ldict = stepdict[last_step]
        
        for layer in sorted(ldict.keys(), key=lambda x: int(x.replace("layer","")) if x.startswith("layer") else (9 if x == "logit" else 10)):
            out[f_key][layer] = {}
            pdict = ldict[layer]
            for pos in sorted(pdict.keys(), key=lambda x: int(x.replace("pos",""))):
                out[f_key][layer][pos] = pdict[pos]
    
    return out

def create_comparison_heatmaps(twohop_id_data, twohop_ood_data, nontree_id_data, nontree_ood_data, output_dir, vmax=None):
    """
    twohop과 nontree의 ID_Test와 OOD 데이터를 비교하는 heatmap을 생성합니다.
    """
    # f1과 f4에 대한 데이터만 선택
    f_keys = ["f1", "f4"]
    
    # 데이터 정렬
    twohop_id_sorted = sort_nested_data(twohop_id_data)
    twohop_ood_sorted = sort_nested_data(twohop_ood_data)
    nontree_id_sorted = sort_nested_data(nontree_id_data)
    nontree_ood_sorted = sort_nested_data(nontree_ood_data)
    
    # 각 데이터셋의 layer와 position 정보 수집
    layers = sorted(set(
        list(twohop_id_sorted["f1"].keys()) +
        list(twohop_ood_sorted["f1"].keys()) +
        list(nontree_id_sorted["f1"].keys()) +
        list(nontree_ood_sorted["f1"].keys())
    ))
    positions = sorted(set(
        list(twohop_id_sorted["f1"][layers[0]].keys()) +
        list(twohop_ood_sorted["f1"][layers[0]].keys()) +
        list(nontree_id_sorted["f1"][layers[0]].keys()) +
        list(nontree_ood_sorted["f1"][layers[0]].keys())
    ))
    
    # 데이터를 numpy 배열로 변환
    def create_heatmap_data(data, f_key):
        heatmap_data = np.zeros((len(layers), len(positions)))
        for i, layer in enumerate(layers):
            for j, pos in enumerate(positions):
                if layer in data[f_key] and pos in data[f_key][layer]:
                    heatmap_data[i, j] = data[f_key][layer][pos]
        return heatmap_data
    
    # 모든 heatmap 데이터 생성
    heatmaps = {}
    for f_key in f_keys:
        heatmaps[f"twohop_{f_key}_id"] = create_heatmap_data(twohop_id_sorted, f_key)
        heatmaps[f"twohop_{f_key}_ood"] = create_heatmap_data(twohop_ood_sorted, f_key)
        heatmaps[f"nontree_{f_key}_id"] = create_heatmap_data(nontree_id_sorted, f_key)
        heatmaps[f"nontree_{f_key}_ood"] = create_heatmap_data(nontree_ood_sorted, f_key)
    
    # vmax 설정
    if vmax is None:
        vmax = max(np.max(data) for data in heatmaps.values())
    
    # figure 생성
    fig = plt.figure(figsize=(20, 5))
    gs = gridspec.GridSpec(1, 8, width_ratios=[1, 1, 1, 1, 1, 1, 1, 1])
    
    # heatmap 순서 정의
    heatmap_order = [
        "twohop_f1_id", "twohop_f4_id", "twohop_f1_ood", "twohop_f4_ood",
        "nontree_f1_id", "nontree_f4_id", "nontree_f1_ood", "nontree_f4_ood"
    ]
    
    # heatmap 생성
    for i, key in enumerate(heatmap_order):
        ax = plt.subplot(gs[i])
        sns.heatmap(heatmaps[key], cmap="RdBu_r", center=0, vmax=vmax, vmin=-vmax,
                   xticklabels=positions, yticklabels=layers, ax=ax)
        ax.set_title(key.replace("_", " ").title())
        if i % 2 == 0:  # 왼쪽 heatmap들에만 y축 레이블 표시
            ax.set_ylabel("Layer")
        else:
            ax.set_ylabel("")
        if i >= 6:  # 마지막 두 heatmap에만 x축 레이블 표시
            ax.set_xlabel("Position")
        else:
            ax.set_xlabel("")
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "iicg_comparison_heatmap.png"), dpi=300, bbox_inches="tight")
    plt.close()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--twohop_root_dir", type=str, required=True, help="2-hop 데이터셋의 루트 디렉토리")
    parser.add_argument("--nontree_root_dir", type=str, required=True, help="nontree 데이터셋의 루트 디렉토리")
    parser.add_argument("--output_dir", type=str, required=True, help="출력 디렉토리")
    parser.add_argument("--vmax", type=float, default=None, help="heatmap의 최대값")
    args = parser.parse_args()
    
    # ID_Test와 OOD 데이터 수집
    twohop_id_data = collect_diffs_twohop(args.twohop_root_dir, "ID_Test")
    twohop_ood_data = collect_diffs_twohop(args.twohop_root_dir, "OOD")
    nontree_id_data = collect_diffs_nontree(args.nontree_root_dir, "ID_Test")
    nontree_ood_data = collect_diffs_nontree(args.nontree_root_dir, "OOD")
    
    # heatmap 생성
    create_comparison_heatmaps(
        twohop_id_data, twohop_ood_data,
        nontree_id_data, nontree_ood_data,
        args.output_dir, args.vmax
    )

if __name__ == "__main__":
    main() 