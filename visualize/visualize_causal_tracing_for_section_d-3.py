import json
import os
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np


def extract_metrics(data, checkpoint, section, metric_type):
    """
    JSON 데이터에서 특정 체크포인트와 섹션의 값을 추출합니다.
    rank의 경우 both_layerN 값을 퍼센트로 변환하고,
    prob의 경우 relative_prob_change_layerN 값을 그대로 사용합니다.
    """
    result = {}
    section_data = data[checkpoint][section]
    
    for pos in section_data.keys():
        pos_data = section_data[pos]
        result[pos] = {}
        for layer in range(1, 8):  # layer1부터 layer7까지
            layer_key = f"layer{layer}"
            if metric_type == "rank":
                metric_key = f"both_{layer_key}"
                if not metric_key in pos_data:
                    AssertionError(f"There is no result for {section}-({layer}, {pos})")
                # both_layerN 값을 퍼센트로 변환
                result[pos][layer_key] = (pos_data[metric_key] / pos_data["sample_num"]) * 100
            elif metric_type == "prob":
                metric_key = f"relative_prob_change_{layer_key}"
                if not metric_key in pos_data:
                    AssertionError(f"There is no result for {section}-({layer}, {pos})")
                # relative_prob_change_layerN 값을 그대로 사용
                result[pos][layer_key] = pos_data[metric_key]
            else:
                raise ValueError("metric_type must be either 'rank' or 'prob'")
    
    return result
    

def create_heatmaps_twohop(input_patterns, output_dir):
    """
    각 입력 파일의 마지막 체크포인트에 대한 test_inferred 섹션의 히트맵을 생성합니다.
    모든 히트맵을 하나의 figure에 가로로 배치합니다.
    """
    
    # rank/prob 판단
    if "_rank_" in input_patterns[0]:
        metric_type = "rank"
    elif "_prob_" in input_patterns[0]:
        metric_type = "prob"
    else:
        raise ValueError("Input pattern must contain either '_rank_' or '_prob_'")
    
    # 출력 디렉토리가 없으면 생성
    os.makedirs(output_dir, exist_ok=True)
    
    # vmin, vmax 설정
    vmin = 0
    vmax = 100 if metric_type == "rank" else None  # prob의 경우 vmin, vmax를 None으로 설정
    
    cmap = sns.light_palette("seagreen", as_cmap=True)
    
    # 전체 figure 생성
    fig_width = 2 * len(input_patterns)  # 각 히트맵당 2의 너비
    fig_height = 4
    fig = plt.figure(figsize=(fig_width, fig_height))
    gs = gridspec.GridSpec(1, len(input_patterns)+1, width_ratios=[1] * len(input_patterns) + [0.1], wspace=0.2)
    axes = [plt.subplot(gs[i]) for i in range(len(input_patterns)+1)]
    
    # 각 입력 파일에 대해 히트맵 생성
    for idx, input_pattern in enumerate(input_patterns):
        ax = axes[idx]
        
        with open(input_pattern, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # 마지막 체크포인트 가져오기
        checkpoint = list(sorted(data.keys(), key=lambda x: float('inf') if x == "final_checkpoint" else int(x.replace("checkpoint-",""))))[-1]
        
        # test_inferred 섹션의 데이터 추출
        section_data = extract_metrics(data, checkpoint, "test_inferred", metric_type)
        
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
                    fmt='.2f',   # 소수점 둘째 자리까지 표시
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
        
        # # 제목 추가 (체크포인트 정보)
        # ax.text(0.5, -0.2, f"Checkpoint {checkpoint}",
        #         ha='center', va='center',
        #         transform=ax.transAxes,
        #         fontsize=16)
    
    # colorbar 추가
    cbar_ax = axes[-1]
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=vmin, vmax=vmax))
    cbar = fig.colorbar(sm, cax=cbar_ax)
    cbar.ax.tick_params(labelsize=14)
    
    # 컬러바 레이블 추가
    cbar.ax.text(0.5, 1.05, '%' if metric_type == "rank" else '$\\Delta p$',
                ha='center', va='bottom',
                fontsize=16, transform=cbar.ax.transAxes)
    
    # 전체 figure 조정
    plt.tight_layout(rect=[0, 0, 0.95, 1], w_pad=0.005)
    
    # PDF로 저장
    output_path = os.path.join(output_dir, f"twohop_causal_tracing_{metric_type}_combined.pdf")
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()
    print(f"Saved combined heatmap to {output_path}")



def main():
    input_paths = ["/home/jinho/repos/GrokkedTransformer/collapse_analysis/tracing_results/2-hop/twohop.70.50000.diff-f12.inf_wd-0.1_layer-8_head-12_seed-42_residual_diff_f1_prob_refined.json", "/home/jinho/repos/GrokkedTransformer/collapse_analysis/tracing_results/2-hop/twohop.100.100000.diff-f12.inf_wd-0.1_layer-8_head-12_seed-42_residual_diff_f1_prob_refined.json", "/home/jinho/repos/GrokkedTransformer/collapse_analysis/tracing_results/2-hop/twohop.150.300000.diff-f12.inf_wd-0.1_layer-8_head-12_seed-42_residual_diff_f1_prob_refined.json", "/home/jinho/repos/GrokkedTransformer/collapse_analysis/tracing_results/2-hop/twohop.200.600000.diff-f12.inf_wd-0.1_layer-8_head-12_seed-42_residual_diff_f1_prob_refined.json"]

    output_dir_path = os.path.join("heatmaps", "combined")
    
    create_heatmaps_twohop(input_paths, output_dir_path)

if __name__ == "__main__":
    main()