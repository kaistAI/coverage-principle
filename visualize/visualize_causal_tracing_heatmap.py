import argparse
import os
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import glob
import matplotlib.gridspec as gridspec

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

def create_heatmaps_twohop(input_pattern, output_dir):
    """
    twohop 데이터에 대한 히트맵을 생성합니다.
    각 체크포인트마다 별도의 히트맵을 생성합니다.
    """
    # 입력 파일 이름에서 N 값 추출
    n_value = int(os.path.basename(input_pattern).split('_detailed_grouping_')[1].split('_')[0])
    
    # N 값이 1, 2, 3이 아닌 경우 NotImplementedError 발생
    if n_value not in [1, 2, 3]:
        raise NotImplementedError(f"현재 구현은 N=1,2,3에 대해서만 지원됩니다. 입력된 N 값: {n_value}")
    
    # rank/prob 판단
    if "_rank_" in input_pattern:
        metric_type = "rank"
    elif "_prob_" in input_pattern:
        metric_type = "prob"
    else:
        raise ValueError("Input pattern must contain either '_rank_' or '_prob_'")
    
    with open(input_pattern, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # 출력 디렉토리가 없으면 생성
    os.makedirs(output_dir, exist_ok=True)
    
    # vmin, vmax 설정
    vmin = 0
    vmax = 100 if metric_type == "rank" else None  # prob의 경우 vmin, vmax를 None으로 설정
    
    cmap = sns.light_palette("seagreen", as_cmap=True)
    
    # N 값에 따른 섹션과 제목 설정
    if n_value == 1:
        sections = ["test_inferred_covered_low_cutoff", "test_inferred_covered_high_cutoff"]
        section_titles = ["k < 3", "k >= 3"]
    elif n_value == 2:
        sections = ["test_inferred_covered_low_cutoff", "test_inferred_covered_mid_cutoff", "test_inferred_covered_high_cutoff"]
        section_titles = ["k < 3", "k = 3", "k > 3"]
    else:  # n_value == 3
        sections = ["test_inferred_covered_2", "test_inferred_covered_3", "test_inferred_covered_4"]
        section_titles = ["k=2", "k=3", "k=4"]
    
    # 각 체크포인트에 대해 히트맵 생성
    for checkpoint in data.keys():
        # 전체 figure 생성 (N 값에 따라 히트맵 개수 조정)
        fig_width = 2.5 * len(sections)  # 각 히트맵당 2.5의 너비
        fig_height = 5
        fig = plt.figure(figsize=(fig_width, fig_height))
        gs = gridspec.GridSpec(1, len(sections), width_ratios=[1] * len(sections), wspace=0.0)
        axes = [plt.subplot(gs[i]) for i in range(len(sections))]
        
        # 각 섹션에 대해 히트맵 생성
        for idx, (section, title) in enumerate(zip(sections, section_titles)):
            ax = axes[idx]
            
            # 데이터 추출
            section_data = extract_metrics(data, checkpoint, section, metric_type)
            
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
        cbar.ax.text(0.5, 1.05, '%' if metric_type == "rank" else 'Δp',
                    ha='center', va='bottom',
                    fontsize=16, transform=cbar.ax.transAxes)
        
        # 전체 figure 조정
        plt.tight_layout(rect=[0, 0, 0.95, 1], w_pad=0.005)
        
        # PDF로 저장
        output_path = os.path.join(output_dir, f"twohop_causal_tracing_detailed_grouping_{n_value}_{metric_type}_{checkpoint}.png")
        plt.savefig(output_path, bbox_inches='tight')
        plt.close()
        print(f"Saved heatmap for checkpoint {checkpoint} to {output_path}")


def create_heatmaps_nontree(input_pattern, output_dir):
    """
    nontree 데이터에 대한 히트맵을 생성합니다.
    여러 JSON 파일을 처리하여 각 체크포인트마다 하나의 figure에 모든 히트맵을 생성합니다.
    """
    
    # input_pattern에서 데이터셋 이름 추출
    dataset_name = os.path.basename(input_pattern).split('_')[0]
    # 출력 디렉토리가 없으면 생성
    output_dir = os.path.join(output_dir, dataset_name)
    os.makedirs(output_dir, exist_ok=True)
    
    # rank/prob 판단
    if "_rank_" in input_pattern:
        metric_type = "rank"
    elif "_prob_" in input_pattern:
        metric_type = "prob"
    else:
        raise ValueError("Input pattern must contain either '_rank_' or '_prob_'")
    
    # 모든 JSON 파일 찾기
    json_files = glob.glob(input_pattern)
    
    if not json_files:
        raise ValueError(f"No JSON files found matching pattern: {input_pattern}")
    
    # 파일 이름에서 f 값으로 정렬
    json_files.sort(key=lambda x: int(os.path.basename(x).split('diff_f')[1].replace(f"_{metric_type}_refined.json", "")))
    
    # checkpoint 추출
    checkpoints = None
    with open(json_files[0], 'r', encoding='utf-8') as f:
        data = json.load(f)
        checkpoints = data.keys()
    for file in json_files[1:]:
        with open(file, 'r', encoding='utf-8') as f:
            data = json.load(f)
            assert checkpoints == data.keys(), f"checkpoin lists are not the same"
    
    # vmin, vmax 설정
    vmin = 0
    vmax = 100 if metric_type == "rank" else None  # prob의 경우 vmin, vmax를 None으로 설정
    
    cmap = sns.light_palette("seagreen", as_cmap=True)
    
    for checkpoint in checkpoints:
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
            f_value = os.path.basename(json_file).split('diff_f')[1].replace(f"_{metric_type}_refined.json", "")
            
            # JSON 파일 읽기
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # 각 섹션에 대해 히트맵 생성
            sections = ["test_inferred"]
            section_titles = ["ID Test"]
            
            for idx, (section, title) in enumerate(zip(sections, section_titles)):
                ax = axes[file_idx]
                
                # 데이터 추출
                section_data = extract_metrics(data, checkpoint, section, metric_type)
                
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
        cbar.ax.text(0.5, 1.05, '%' if metric_type == "rank" else 'Δp',
                    ha='center', va='bottom',
                    fontsize=16, transform=cbar.ax.transAxes)
        
        # 전체 figure 조정
        plt.tight_layout(rect=[0, 0, 0.95, 1], w_pad=0.005)
        
        # PDF로 저장
        output_path = os.path.join(output_dir, f"nontree_causal_tracing_{metric_type}_{checkpoint}.pdf")
        plt.savefig(output_path, bbox_inches='tight')
        plt.close()
        print(f"Saved combined heatmap to {output_path}")
        
    
def create_heatmaps_parallel2hop(input_pattern, output_dir):
    """
    nontree 데이터에 대한 히트맵을 생성합니다.
    여러 JSON 파일을 처리하여 각 체크포인트마다 하나의 figure에 모든 히트맵을 생성합니다.
    """
    
    # input_pattern에서 데이터셋 이름 추출
    dataset_name = os.path.basename(input_pattern).split('_')[0]
    # 출력 디렉토리가 없으면 생성
    output_dir = os.path.join(output_dir, dataset_name)
    os.makedirs(output_dir, exist_ok=True)
    
    # rank/prob 판단
    if "_rank_" in input_pattern:
        metric_type = "rank"
    elif "_prob_" in input_pattern:
        metric_type = "prob"
    else:
        raise ValueError("Input pattern must contain either '_rank_' or '_prob_'")
    
    # 모든 JSON 파일 찾기
    json_files = glob.glob(input_pattern)
    
    if not json_files:
        raise ValueError(f"No JSON files found matching pattern: {input_pattern}")
    
    # 파일 이름에서 f 값으로 정렬
    json_files.sort(key=lambda x: int(os.path.basename(x).split('diff_f')[1].replace(f"_{metric_type}_refined.json", "")))
    
    # checkpoint 추출
    checkpoints = None
    with open(json_files[0], 'r', encoding='utf-8') as f:
        data = json.load(f)
        checkpoints = data.keys()
    for file in json_files[1:]:
        with open(file, 'r', encoding='utf-8') as f:
            data = json.load(f)
            assert checkpoints == data.keys(), f"checkpoin lists are not the same"
    
    # vmin, vmax 설정
    vmin = 0
    vmax = 100 if metric_type == "rank" else None  # prob의 경우 vmin, vmax를 None으로 설정
    
    cmap = sns.light_palette("seagreen", as_cmap=True)
    
    for checkpoint in checkpoints:
        # 전체 figure 생성 (가로로 나열)
        n_files = len(json_files)
        print(n_files)
        fig_width = 2.6 * n_files  # 각 파일마다 2.5의 너비
        fig_height = 4  # 하나의 높이
        fig = plt.figure(figsize=(fig_width, fig_height))
        gs = gridspec.GridSpec(1, n_files + 1, width_ratios=[1] * n_files + [0.1], wspace=0.1)
        axes = [plt.subplot(gs[i]) for i in range(n_files+1)]
        
        # f 값에 따른 제목 매핑
        f_to_title = {
            "1": "b1'",
            "2": "b2'"
        }
        
        # 각 JSON 파일에 대해 히트맵 생성
        for file_idx, json_file in enumerate(json_files):
            # 파일 이름에서 f 값 추출
            f_value = os.path.basename(json_file).split('diff_f')[1].replace(f"_{metric_type}_refined.json", "")
            
            # JSON 파일 읽기
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # 각 섹션에 대해 히트맵 생성
            sections = ["test_inferred"]
            section_titles = ["ID Test"]
            
            for idx, (section, title) in enumerate(zip(sections, section_titles)):
                ax = axes[file_idx]
                
                # 데이터 추출
                section_data = extract_metrics(data, checkpoint, section, metric_type)
                
                # 데이터를 numpy 배열로 변환
                positions = ["0", "1", "2", "3"]
                layers = [f"layer{i}" for i in range(1, 8)]
                
                heatmap_data = np.zeros((len(layers), len(positions)))
                for i, layer in enumerate(layers):
                    for j, pos in enumerate(positions):
                        heatmap_data[i, j] = section_data[pos][layer]
                
                # 레이어 순서를 반대로 변경 (layer1이 맨 아래로)
                heatmap_data = np.flipud(heatmap_data)
                
                # heatmap 생성
                sns.heatmap(heatmap_data,
                            annot=True,
                            fmt='.2f',   # 소수점 첫째 자리까지 표시
                            cmap=cmap,
                            vmin=vmin,
                            vmax=vmax,
                            xticklabels=["x1", "x2", "x3", "x4"],
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
        
        # # colorbar 추가
        cbar_ax = axes[-1]
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=vmin, vmax=vmax))
        cbar = fig.colorbar(sm, cax=cbar_ax)
        cbar.ax.tick_params(labelsize=14)
        
        # 컬러바 레이블 추가
        cbar.ax.text(0.5, 1.05, '%' if metric_type == "rank" else 'Δp',
                    ha='center', va='bottom',
                    fontsize=16, transform=cbar.ax.transAxes)
        
        # 전체 figure 조정
        plt.tight_layout(rect=[0, 0, 0.95, 1], w_pad=0.005)
        
        # PDF로 저장
        output_path = os.path.join(output_dir, f"parallel2hop_causal-tracing_{metric_type}_{checkpoint}.pdf")
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
    elif "parallel2hop" in args.input:
        # parallel2hop 데이터 처리
        create_heatmaps_parallel2hop(args.input, args.output_dir)
    else:
        raise ValueError("Input path must contain either 'twohop' or 'nontree'")

if __name__ == "__main__":
    main()
