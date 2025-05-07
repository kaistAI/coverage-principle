import argparse
import os
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import glob
import matplotlib.gridspec as gridspec

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

def create_heatmaps_twohop(input_pattern, output_dir):
    """
    twohop 데이터에 대한 히트맵을 생성합니다.
    """
    with open(input_pattern, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # 출력 디렉토리가 없으면 생성
    os.makedirs(output_dir, exist_ok=True)
    
    # vmin, vmax 설정
    vmin = 0
    vmax = 100
    
    cmap = sns.light_palette("seagreen", as_cmap=True)
    
    # 전체 figure 생성 (2개의 히트맵)
    fig_width = 2.5 * 2  # 2개의 히트맵
    fig_height = 5
    fig = plt.figure(figsize=(fig_width, fig_height))
    gs = gridspec.GridSpec(1, 2, width_ratios=[1, 1], wspace=0.0)
    axes = [plt.subplot(gs[i]) for i in range(2)]
    
    # 각 섹션에 대해 히트맵 생성
    sections = ["test_inferred_low_cutoff", "test_inferred_high_cutoff"]
    section_titles = ["k < 3", "k >= 3"]
    
    for idx, (section, title) in enumerate(zip(sections, section_titles)):
        ax = axes[idx]
        
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
        
        # 제목 추가
        ax.text(0.5, -0.2, title,
                ha='center', va='center',
                transform=ax.transAxes,
                fontsize=16)
    
    # 전체 figure에 대한 colorbar 추가
    cbar_ax = fig.add_axes([0.97, 0.23, 0.02, 0.7])
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=vmin, vmax=vmax))
    cbar = fig.colorbar(sm, cax=cbar_ax)
    cbar.ax.tick_params(labelsize=14)
    
    # 컬러바 레이블 추가
    cbar.ax.text(0.5, 1.05, '%',
                ha='center', va='bottom',
                fontsize=16, transform=cbar.ax.transAxes)
    
    # 전체 figure 조정
    plt.tight_layout(rect=[0, 0, 0.95, 1], w_pad=0.005)
    
    # PDF로 저장
    output_path = os.path.join(output_dir, "twohop_causal_tracing.pdf")
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()
    print(f"Saved combined heatmap to {output_path}")

def create_heatmaps_nontree(input_pattern, output_dir):
    """
    nontree 데이터에 대한 히트맵을 생성합니다.
    여러 JSON 파일을 처리하여 하나의 figure에 모든 히트맵을 생성합니다.
    """
    # 출력 디렉토리가 없으면 생성
    os.makedirs(output_dir, exist_ok=True)
    
    # 모든 JSON 파일 찾기
    json_files = glob.glob(input_pattern)
    
    if not json_files:
        raise ValueError(f"No JSON files found matching pattern: {input_pattern}")
    
    # 파일 이름에서 f 값으로 정렬
    json_files.sort(key=lambda x: int(os.path.basename(x).split('diff_f')[1].replace("_refined.json", "")))
    
    # vmin, vmax 설정
    vmin = 0
    vmax = 100
    
    cmap = sns.light_palette("seagreen", as_cmap=True)
    
    # 전체 figure 생성 (가로로 나열)
    n_files = len(json_files)
    fig_width = 2.5 * n_files  # 각 파일마다 2.5의 너비
    fig_height = 5  # 하나의 높이
    fig = plt.figure(figsize=(fig_width, fig_height))
    gs = gridspec.GridSpec(1, n_files, width_ratios=[1] * n_files, wspace=0.0)
    axes = [plt.subplot(gs[i]) for i in range(n_files)]
    
    # f 값에 따른 제목 매핑
    f_to_title = {
        "1": "b'",
        "4": "(b, x2')",
        "5": "(b', x2')"
    }
    
    # 각 JSON 파일에 대해 히트맵 생성
    for file_idx, json_file in enumerate(json_files):
        # 파일 이름에서 f 값 추출
        f_value = os.path.basename(json_file).split('diff_f')[1].replace("_refined.json", "")
        
        # JSON 파일 읽기
        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # 각 섹션에 대해 히트맵 생성
        sections = ["test_inferred"]
        section_titles = ["ID Test"]
        
        for idx, (section, title) in enumerate(zip(sections, section_titles)):
            ax = axes[file_idx]
            
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
                        yticklabels=[f"Layer {i}" for i in range(len(layers), 0, -1)] if file_idx == 0 else [],  # 가장 왼쪽 heatmap에만 y축 레이블 표시
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
            if file_idx == 0:  # 첫 번째 heatmap에만 적용
                ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=13)
            
            # subplot 영역 테두리 추가
            for spine in ax.spines.values():
                spine.set_visible(True)
                spine.set_color('black')
                spine.set_linewidth(1.0)
            
            # 제목 추가 (f 값에 따른 매핑된 제목 사용)
            title = f_to_title.get(f_value, f"f={f_value}")
            ax.text(0.5, -0.2, title,
                    ha='center', va='center',
                    transform=ax.transAxes,
                    fontsize=16)
    
    # 전체 figure에 대한 colorbar 추가
    cbar_ax = fig.add_axes([0.97, 0.23, 0.02, 0.7])
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=vmin, vmax=vmax))
    cbar = fig.colorbar(sm, cax=cbar_ax)
    cbar.ax.tick_params(labelsize=14)
    
    # 컬러바 레이블 추가
    cbar.ax.text(0.5, 1.05, '%',
                ha='center', va='bottom',
                fontsize=16, transform=cbar.ax.transAxes)
    
    # 전체 figure 조정
    plt.tight_layout(rect=[0, 0, 0.95, 1], w_pad=0.005)
    
    # PDF로 저장
    output_path = os.path.join(output_dir, "nontree_causal_tracing.pdf")
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()
    print(f"Saved combined heatmap to {output_path}")

def main():
    parser = argparse.ArgumentParser(description="Create causal tracing heatmaps from JSON results")
    parser.add_argument(
        "--input", "-i", required=True,
        help="입력 JSON 파일 경로 또는 패턴"
    )
    parser.add_argument(
        "--output_dir", "-o", default="heatmaps",
        help="히트맵 PDF 파일들을 저장할 디렉토리 (기본: heatmaps)"
    )
    args = parser.parse_args()
    
    if "twohop" in args.input:
        # twohop 데이터 처리
        create_heatmaps_twohop(args.input, args.output_dir)
    elif "nontree" in args.input:
        # nontree 데이터 처리
        create_heatmaps_nontree(args.input, args.output_dir)
    else:
        raise ValueError("Input path must contain either 'twohop' or 'nontree'")

if __name__ == "__main__":
    main()
