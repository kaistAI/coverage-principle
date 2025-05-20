import argparse
import os
import glob
import re
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from collections import defaultdict


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


def collect_diffs_nontree(root_dir, section):
    """
    nontree 데이터셋을 위한 재귀 탐색으로 모든 similarity_metrics_layer*.txt 파일을 찾아
    (f, layer, pos, step) 별로 차이를 수집합니다.
    """
    result = {}
    # 파일 경로에서 f, (layer,pos), step 추출용 정규식
    re_f         = re.compile(r"[\\/](f\d+)[\\/]")
    re_layer_pos = re.compile(r"/\((?:(\d+)|(logit|prob)),(\d+)\)/")
    re_step      = re.compile(r"/step(\d+|final_checkpoint)/")
    
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
        layer_idx    = lp.group(1) or lp.group(2)  # 숫자 또는 "logit"/"prob"
        pos_idx      = lp.group(3)              # e.g. "2"
        step_key     = "step" + sm.group(1)     # e.g. "step25000" or "stepfinal_checkpoint"
        layer_key    = "layer" + layer_idx if layer_idx.isdigit() else layer_idx
        pos_key      = "pos"   + pos_idx
        
        result.setdefault(f_key, {})\
              .setdefault(step_key, {})\
              .setdefault(layer_key, {})[pos_key] = diff
              
    return result


def sort_nested_nontree(data):
    """
    nontree 데이터셋을 위한 정렬 함수
    모든 f_key에 대해 layer1, layer2… 순으로,
    각 layer 안의 pos1, pos2… 순으로 정렬된 새 dict 반환.
    마지막 step의 데이터만 남깁니다.
    """
    out = {}
    out["default"] = {}
    # 모든 f_key 처리
    for f_key in sorted(data.keys()):
        out["default"][f_key] = {}
        stepdict = data[f_key]
        # 모든 step 처리
        for step in sorted(stepdict.keys(), key=lambda x: float('inf') if x == "stepfinal_checkpoint" else int(x.replace("step",""))):
            out["default"][f_key][step] = {}
            ldict = stepdict[step]
            # layer 키 정렬 (logit은 9번째, prob은 10번째)
            for layer in sorted(ldict.keys(), key=lambda x: int(x.replace("layer","")) if x.startswith("layer") else (9 if x == "logit" else 10)):
                out["default"][f_key][step][layer] = {}
                pdict = ldict[layer]
                # pos 키 정렬
                for pos in sorted(pdict.keys(), key=lambda x: int(x.replace("pos",""))):
                    out["default"][f_key][step][layer][pos] = pdict[pos]
    return out


def filter_data(data, sections=None, cutoffs_or_ks=None, fs=None, steps=None, layers=None, positions=None):
    """
    nontree 데이터셋의 결과를 필터링합니다.
    sections: 포함할 section 리스트 (예: ["ID Test", "OOD"])
    cutoffs_or_funcs: 포함할 cutoff_type 또는 k 리스트 (예: ["low_cutoff", "high_cutoff"] 또는 ["2", "4"]) 또는 함수 리스트 (예: ["f1, f2"])
    fs: 포함할 f 리스트 (예: ["f1", "f4"])
    steps: 포함할 step 리스트 (예: ["step10000", "stepfinal_checkpoint"])
    positions: 포함할 position 리스트 (예: ["pos1", "pos3"])
    각 매개변수가 None이면 해당 차원의 모든 데이터를 유지합니다.
    """
    def tree():
        return defaultdict(tree)
    
    def dictify(d):
        """
        Recursively convert a (possibly nested) defaultdict to a regular dict.
        Any normal dicts encountered will also be converted at their sub-levels.
        """
        if isinstance(d, defaultdict):
            # First turn this defaultdict into a dict, but recurse into values
            d = dict(d)
        if isinstance(d, dict):
            # For every key in this dict, convert its value as well
            return {k: dictify(v) for k, v in d.items()}
        # Base case: not a dict, just return it
        return d
    
    result = tree()
    
    for section in data:
        if sections is not None and section not in sections:
            continue
        for cutoff_or_k in data[section]:
            if cutoffs_or_ks is not None and cutoff_or_k not in cutoffs_or_ks:
                continue
            for f_key in data[section][cutoff_or_k]:
                if fs is not None and f_key not in fs:
                    continue
                for step in data[section][cutoff_or_k][f_key]:
                    if steps is not None and step not in steps:
                        continue
                    for layer in data[section][cutoff_or_k][f_key][step]:
                        if layers is not None and layer not in layers:
                            continue
                        for pos in data[section][cutoff_or_k][f_key][step][layer]:
                            if positions is not None and pos not in positions:
                                continue
                            result[section][cutoff_or_k][f_key][step][layer][pos] = data[section][cutoff_or_k][f_key][step][layer][pos]
    return dictify(result)

def create_heatmaps_nontree(id_data, output_dir, vmax):
    """
    ID Test 데이터에서 f1과 f4에 대한 히트맵과 컬러바를 그립니다.
    """
    # 출력 디렉토리가 없으면 생성
    os.makedirs(output_dir, exist_ok=True)
    
    # vmin 설정
    vmin = 0
    
    cmap = sns.light_palette("seagreen", as_cmap=True)
    
    # f_key에 따른 텍스트 매핑
    f_key_to_text = {
        "f1": "b",
        "f2": "t",
        "f3": "x2",
        "f4": "(b, x2)",
        "f5": "(b, x1)"
    }
    
    print(id_data)
    
    available_f_keys = sorted([f for f in id_data["default"].keys()])
    num_f_keys = len(available_f_keys)
    
    if num_f_keys == 0:
        print("No valid f_key pairs found in both ID and OOD data")
        return
    
    for step in sorted(next(iter(next(iter(id_data.values())).values())).keys(), key=lambda x: float('inf') if x == "stepfinal_checkpoint" else int(x.replace("step",""))):
        # 전체 figure 생성 (ID Test N개 + colorbar)
        fig_height = 4
        fig_width = 1.8 * num_f_keys  # ID Test N개
        fig = plt.figure(figsize=(fig_width, fig_height))
        gs = gridspec.GridSpec(1, num_f_keys + 1, width_ratios=[1] * num_f_keys + [0.1], wspace=0.2)
        axes = [plt.subplot(gs[i]) for i in range(num_f_keys + 1)]
        
        # 각 f_key에 대해 처리
        for idx, (f_key, title) in enumerate(zip(available_f_keys, [f_key_to_text[f_key] for f_key in available_f_keys])):
            ax = axes[idx]
            layer_data = id_data["default"][f_key][step]
            
            # 데이터를 numpy 배열로 변환
            layers = sorted(layer_data.keys(), key=lambda x: int(x.replace("layer","")) if x.startswith("layer") else (9 if x == "logit" else 10), reverse=True)
            # positions를 데이터에서 동적으로 가져오기
            positions = sorted(set(pos for layer in layer_data.values() for pos in layer.keys()), 
                            key=lambda x: int(x.replace("pos","")))
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
                       fmt='.2f',
                       cmap=cmap,
                       vmin=vmin,
                       vmax=vmax,
                       xticklabels=["$x1$", "$x2$", "$x3$", "$b$"] if len(positions) == 4 else [f"$x{i+1}$" for i in range(len(positions))],
                       yticklabels=[f"Layer {i}" for i in range(len(layers), 0, -1)] if idx == 0 else [],  # 가장 왼쪽 heatmap에만 y축 레이블 표시
                       cbar=False,
                       linewidths=0.5,
                       linecolor='lightgray',
                       ax=ax,
                       square=True)
            
            # y축 레이블 회전 (가로로 표시)
            if idx == 0:  # 첫 번째 heatmap에만 적용
                ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=16)
               
            # x축 레이블 폰트 크기 설정
            ax.set_xticklabels(ax.get_xticklabels(), fontsize=16)
            
            # 최대값 셀에만 값 표시
            for i, j in zip(*max_indices):
                ax.add_patch(plt.Rectangle((j, i), 1, 1, fill=False, edgecolor='black', lw=2))
                ax.text(j + 0.5, i + 0.5, f'{max_value:.2f}', 
                       ha='center', va='center', 
                       color='black', fontweight='bold', fontsize=11)
                
            # tick 제거
            ax.tick_params(axis='both', which='both', length=0)
            
            # subplot 영역 테두리 추가
            for spine in ax.spines.values():
                spine.set_visible(True)
                spine.set_color('black')
                spine.set_linewidth(1.0)
            
            # 제목을 heatmap 아래쪽에 추가
            ax.text(0.5, -0.18, title, 
                   ha='center', va='center', 
                   transform=ax.transAxes, 
                   fontsize=20)
            
        # colorbar 추가
        cbar_ax = axes[-1]  # colorbar는 마지막 subplot
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=vmin, vmax=vmax))
        cbar = fig.colorbar(sm, cax=cbar_ax)
        cbar.ax.tick_params(labelsize=16)
        
        # IICG 레이블 추가
        cbar.ax.text(5.5, 0.5, 'IICG', ha='left', va='center', fontsize=20, transform=cbar.ax.transAxes)
        
        # # 점선을 figure 전체에 걸쳐 그리기 위해 figure 좌표계 사용
        # if "default" in id_data:
        #     fig.add_artist(plt.Line2D([0.525, 0.525], [0.05, 0.97], color='gray', linestyle='--', alpha=0.5, transform=fig.transFigure))
        # elif n_value == 1:
        #     fig.add_artist(plt.Line2D([0.65, 0.65], [0.05, 0.97], color='gray', linestyle='--', alpha=0.5, transform=fig.transFigure))
        # elif n_value == 2:
        #     fig.add_artist(plt.Line2D([0.682, 0.682], [-0.06, 0.87], color='gray', linestyle='--', alpha=0.5, transform=fig.transFigure))
        # elif n_value == 3:
        #     fig.add_artist(plt.Line2D([0.75, 0.75], [0.05, 0.97], color='gray', linestyle='--', alpha=0.5, transform=fig.transFigure))
        
        # 전체 figure 조정
        plt.subplots_adjust(wspace=0.001)  # subplot 간의 가로 간격을 매우 좁게 설정
        plt.tight_layout(rect=[0, 0, 0.95, 1], w_pad=0.005)
        
        # PDF로 저장
        output_path = os.path.join(output_dir, f"nontree_iicg_heatmap_section_5-4_step{step.replace('step','')}.pdf")
        # if "default" in id_data:
        #     output_path = os.path.join(output_dir, f"nontree_iicg_heatmap_step{step.replace('step','')}.pdf")
        # else:
        #     output_path = os.path.join(output_dir, f"twohop_iicg_heatmap_detailed_grouping_{n_value}_step{step.replace('step','')}.pdf")
        plt.savefig(output_path, bbox_inches='tight')
        plt.close()
        print(f"Saved combined heatmap to {output_path}")
            
        
    return
    
    # 전체 figure 생성 (f1, f4, colorbar)
    fig_height = 4
    fig_width = 2.2 * 2  # f1, f4, colorbar
    fig_height = 5
    fig = plt.figure(figsize=(fig_width, fig_height))
    gs = gridspec.GridSpec(1, 3, width_ratios=[1, 1, 0.1], wspace=0.05)
    axes = [plt.subplot(gs[i]) for i in range(3)]
    
    # f1과 f4에 대해 처리
    for idx, (f_key, title) in enumerate(zip(['f1', 'f4'], ['b', '(b, x2)'])):
        if f_key not in id_data:
            continue
            
        ax = axes[idx]
        layer_data = id_data[f_key]
        
        # 히트맵 데이터 준비
        layers = sorted(layer_data.keys(), key=lambda x: int(x.replace("layer","")) if x.startswith("layer") else (9 if x == "logit" else 10), reverse=True)
        # logit과 prob 제외
        layers = [layer for layer in layers if layer.startswith("layer")]
        positions = sorted(set(pos for layer in layer_data.values() for pos in layer.keys()), 
                        key=lambda x: int(x.replace("pos","")))
        heatmap_data = np.zeros((8, len(positions)))  # 8개의 layer만 사용
        for i, layer in enumerate(layers):
            for j, pos in enumerate(positions):
                if layer in layer_data and pos in layer_data[layer]:
                    heatmap_data[i, j] = layer_data[layer][pos]
        
        # 최대값 찾기
        max_value = np.max(heatmap_data)
        max_indices = np.where(heatmap_data == max_value)
        
        # 히트맵 생성
        sns.heatmap(heatmap_data,
                    annot=False,
                    cmap=cmap,
                    vmin=vmin,
                    vmax=vmax,
                    xticklabels=["x1", "x2", "x3"],
                    yticklabels=[f"Layer {i}" for i in range(8, 0, -1)] if idx == 0 else [],  # f1 히트맵에만 y축 레이블 표시
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
                   color='black', fontweight='bold', fontsize=8)
        
        # tick 제거
        ax.tick_params(axis='both', which='both', length=0)
        
        # x축 레이블 폰트 크기 설정
        ax.set_xticklabels(ax.get_xticklabels(), fontsize=13)
        
        # subplot 영역 테두리 추가
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_color('black')
            spine.set_linewidth(1.0)
        
        # 제목 추가
        ax.text(0.5, -0.15, f"{title}",
                ha='center', va='center',
                transform=ax.transAxes,
                fontsize=14)
    
    # 컬러바 추가
    cbar_ax = axes[2]
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=vmin, vmax=vmax))
    cbar = fig.colorbar(sm, cax=cbar_ax)
    cbar.ax.tick_params(labelsize=10)
    
    # IICG 레이블 추가
    cbar.ax.text(0.5, 1.02, 'IICG', ha='center', va='bottom', fontsize=12, transform=cbar.ax.transAxes)
    
    # 전체 figure 조정
    plt.subplots_adjust(wspace=0.001)  # subplot 간의 가로 간격을 매우 좁게 설정
    plt.tight_layout(rect=[0, 0, 0.95, 1], w_pad=0.005)
    
    # PDF로 저장
    output_path = os.path.join(output_dir, "nontree_iicg_heatmap_5-4.pdf")
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()
    print(f"Saved three heatmaps to {output_path}")

def main():
    parser = argparse.ArgumentParser(description="Create three heatmaps (ID_Test f1, ID_Test f4, and colorbar)")
    parser.add_argument(
        "--root_dir", "-r", required=True,
        help="similarity_metrics_layer*.txt 들이 있는 최상위 디렉토리"
    )
    parser.add_argument(
        "--output_dir", "-d", default="heatmaps",
        help="heatmap PDF 파일을 저장할 디렉토리 (기본: heatmaps)"
    )
    args = parser.parse_args()
    
    assert "results" in args.root_dir, "It should be results directory"
    assert "nontree" in args.root_dir, "This script is for nontree dataset only"

    all_data_nontree = {}
    for section in ["ID Test", "OOD"]:
        raw = collect_diffs_nontree(args.root_dir, section)
        sorted_data = sort_nested_nontree(raw)
        all_data_nontree[section] = sorted_data
    
    #########################################################
    # Should change here! 
    #########################################################
    filtered_data_nontree = filter_data(
        all_data_nontree, 
        sections=["ID Test"],
        fs=["f1", "f4"], 
        steps=["step50000"], 
        layers=["layer1", "layer2", "layer3", "layer4", "layer5", "layer6", "layer7", "layer8"]
    )
    print(filtered_data_nontree)
    
    all_values = []
    # Maximum value for figure's colorbar
    for section_dict in filtered_data_nontree.values():
        for cutoff_or_k_dict in section_dict.values():
            for f_key_dict in cutoff_or_k_dict.values():
                for step_dict in f_key_dict.values():
                    for layer_dict in step_dict.values():
                        for pos, value in layer_dict.items():
                            if isinstance(value, (int, float)):
                                all_values.append(value)
    vmax = max(all_values)
    vmax = 0.5
    
    output_dir_path = os.path.join(args.output_dir, args.root_dir.split("/")[-1])
    
    # 3개의 히트맵 생성
    create_heatmaps_nontree(filtered_data_nontree["ID Test"], output_dir_path, vmax)

if __name__ == "__main__":
    main() 