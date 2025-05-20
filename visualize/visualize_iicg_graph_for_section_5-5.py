import argparse
import os
import glob
import re
import matplotlib.pyplot as plt
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


def collect_diffs_twohop(root_dir, section):
    """
    2-hop 데이터셋을 위한 재귀 탐색으로 모든 similarity_metrics_layer*.txt 파일을 찾아
    detailed_grouping에 따라 (cutoff_type, f, layer, pos, step) 또는 (k, f, layer, pos, step) 별로 차이를 수집합니다.
    _detailed_grouping_이 포함된 경우에만 detailed_grouping을 체크하고, 그렇지 않은 경우 기본 구조로 저장합니다.
    """
    result = {}
    
    # _detailed_grouping_ 포함 여부 확인
    has_detailed_grouping = "_detailed_grouping_" in root_dir
    
    # 파일 경로에서 f, cutoff_type/k, (layer,pos), step 추출용 정규식
    re_f         = re.compile(r"[\\/](f\d+)[\\/]")
    re_layer_pos = re.compile(r"/\((\d+),(\d+)\)/")
    re_step      = re.compile(r"/step(\d+|final_checkpoint)/")
    if has_detailed_grouping:
        re_cutoff    = re.compile(r"/(low_cutoff|mid_cutoff|high_cutoff)/")
        re_k         = re.compile(r"/covered_(\d+)/")
    
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
        if has_detailed_grouping:
            cm = re_cutoff.search(path)
            km = re_k.search(path)

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
            # 모든 f 키 처리
            for f_key in sorted(fdict.keys()):
                out[cutoff][f_key] = {}
                stepdict = fdict[f_key]
                # 모든 step 처리
                for step in sorted(stepdict.keys(), key=lambda x: float('inf') if x == "stepfinal_checkpoint" else int(x.replace("step",""))):
                    out[cutoff][f_key][step] = {}
                    ldict = stepdict[step]
                    # layer 키 정렬
                    for layer in sorted(ldict.keys(), key=lambda x: int(x.replace("layer",""))):
                        out[cutoff][f_key][step][layer] = {}
                        pdict = ldict[layer]
                        # pos 키 정렬
                        for pos in sorted(pdict.keys(), key=lambda x: int(x.replace("pos",""))):
                            out[cutoff][f_key][step][layer][pos] = pdict[pos]
    elif "low_cutoff" in data and "high_cutoff" in data:
        # N=1: low_cutoff, high_cutoff
        for cutoff in ["low_cutoff", "high_cutoff"]:
            if cutoff not in data:
                continue
            out[cutoff] = {}
            fdict = data[cutoff]
            # 모든 f 키 처리
            for f_key in sorted(fdict.keys()):
                out[cutoff][f_key] = {}
                stepdict = fdict[f_key]
                # 모든 step 처리
                for step in sorted(stepdict.keys(), key=lambda x: float('inf') if x == "stepfinal_checkpoint" else int(x.replace("step",""))):
                    out[cutoff][f_key][step] = {}
                    ldict = stepdict[step]
                    # layer 키 정렬
                    for layer in sorted(ldict.keys(), key=lambda x: int(x.replace("layer",""))):
                        out[cutoff][f_key][step][layer] = {}
                        pdict = ldict[layer]
                        # pos 키 정렬
                        for pos in sorted(pdict.keys(), key=lambda x: int(x.replace("pos",""))):
                            out[cutoff][f_key][step][layer][pos] = pdict[pos]
    elif "2" in data and "3" in data and "4" in data:
        # N=3: 2, 3, 4
        for k in ["2", "3", "4"]:
            if k not in data:
                continue
            out[k] = {}
            fdict = data[k]
            # 모든 f 키 처리
            for f_key in sorted(fdict.keys()):
                out[k][f_key] = {}
                stepdict = fdict[f_key]
                # 모든 step 처리
                for step in sorted(stepdict.keys(), key=lambda x: float('inf') if x == "stepfinal_checkpoint" else int(x.replace("step",""))):
                    out[k][f_key][step] = {}
                    ldict = stepdict[step]
                    # layer 키 정렬
                    for layer in sorted(ldict.keys(), key=lambda x: int(x.replace("layer",""))):
                        out[k][f_key][step][layer] = {}
                        pdict = ldict[layer]
                        # pos 키 정렬
                        for pos in sorted(pdict.keys(), key=lambda x: int(x.replace("pos",""))):
                            out[k][f_key][step][layer][pos] = pdict[pos]
    elif "default" in data:
        # detailed_grouping이 없는 경우
        out["default"] = {}
        fdict = data["default"]
        # 모든 f_key 처리
        for f_key in sorted(fdict.keys()):
            out["default"][f_key] = {}
            stepdict = fdict[f_key]
            # 모든 step 처리
            for step in sorted(stepdict.keys(), key=lambda x: float('inf') if x == "stepfinal_checkpoint" else int(x.replace("step",""))):
                out["default"][f_key][step] = {}
                ldict = stepdict[step]
                # layer 키 정렬
                for layer in sorted(ldict.keys(), key=lambda x: int(x.replace("layer",""))):
                    out["default"][f_key][step][layer] = {}
                    pdict = ldict[layer]
                    # pos 키 정렬
                    for pos in sorted(pdict.keys(), key=lambda x: int(x.replace("pos",""))):
                        out["default"][f_key][step][layer][pos] = pdict[pos]
    else:
        raise ValueError("Unknown detailed_grouping pattern in data")
    
    return out


def sort_nested_nontree(data):
    """
    nontree 데이터셋을 위한 정렬 함수
    모든 f_key에 대해 layer1, layer2… 순으로,
    각 layer 안의 pos1, pos2… 순으로 정렬된 새 dict 반환.
    모든 step의 데이터를 저장합니다.
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
    2-hop 데이터셋의 결과를 필터링합니다.
    sections: 포함할 section 리스트 (예: ["ID Test", "OOD"])
    cutoffs_or_ks: 포함할 cutoff_type 또는 k 리스트 (예: ["low_cutoff", "high_cutoff"] 또는 ["2", "4"])
    fs: 포함할 f 리스트 (예: ["f1", "f4"])
    steps: 포함할 step 리스트 (예: ["step10000", "stepfinal_checkpoint"])   
    layers: 포함할 layer 리스트 (예: ["layer1", "layer2"])
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


def create_line_graph(filtered_data_twohop, filtered_data_nontree, output_dir):
    """
    filtered_data_twohop와 filtered_data_nontree에 있는 데이터를 사용하여 꺾은 선 그래프를 그립니다.
    가로축은 layer_num, 세로축은 실제 값입니다.
    각 step 단위의 dictionary가 하나의 line이 됩니다.
    """
    # 출력 디렉토리가 없으면 생성
    os.makedirs(output_dir, exist_ok=True)
    
    # Figure 생성
    plt.figure(figsize=(6, 4))
    
    # 라인 스타일과 색상 정의
    styles = {
        'twohop_f1': {'color': 'blue', 'marker': 'o', 'linestyle': '-', 'label': '2-Hop (b)'},
        'twohop_f4': {'color': 'blue', 'marker': 'D', 'linestyle': '-.', 'label': '2-Hop (b, x2)'},
        'nontree_f1': {'color': 'red', 'marker': 'o', 'linestyle': '-', 'label': 'Non-tree (b)'},
        'nontree_f4': {'color': 'red', 'marker': 'D', 'linestyle': '-.', 'label': 'Non-tree (b, x2)'}
    }
    
    # 2-hop 데이터 그리기
    for section in filtered_data_twohop:
        for cutoff_or_k in filtered_data_twohop[section]:  # key는 'default' 또는 cutoff 타입
            for f_key in filtered_data_twohop[section][cutoff_or_k]:
                for step in filtered_data_twohop[section][cutoff_or_k][f_key]:
                    layer_data = filtered_data_twohop[section][cutoff_or_k][f_key][step]
                    
                    # x축(layer 번호)과 y축(값) 데이터 준비
                    x_values = []
                    y_values = []
                    
                    for layer in sorted(layer_data.keys(), key=lambda x: int(x.replace("layer", ""))):
                        layer_num = int(layer.replace("layer", ""))
                        x_values.append(layer_num)
                        y_values.append(layer_data[layer]["pos3"])
                    
                    # 함수에 따라 다른 스타일 적용
                    style_key = 'twohop_f1' if f_key == 'f1' else 'twohop_f4'
                    plt.plot(x_values, y_values, **styles[style_key])
    
    # non-tree 데이터 그리기
    for section in filtered_data_nontree:
        for cutoff_or_k in filtered_data_nontree[section]:  # key는 'default' 또는 cutoff 타입
            for f_key in filtered_data_nontree[section][cutoff_or_k]:
                for step in filtered_data_nontree[section][cutoff_or_k][f_key]:
                    layer_data = filtered_data_nontree[section][cutoff_or_k][f_key][step]
                
                    # x축(layer 번호)과 y축(값) 데이터 준비
                    x_values = []
                    y_values = []
                    
                    for layer in sorted(layer_data.keys(), key=lambda x: int(x.replace("layer", ""))):
                        layer_num = int(layer.replace("layer", ""))
                        x_values.append(layer_num)
                        y_values.append(layer_data[layer]["pos3"])
                    
                    # 함수에 따라 다른 스타일 적용
                    style_key = 'nontree_f1' if f_key == 'f1' else 'nontree_f4'
                    plt.plot(x_values, y_values, **styles[style_key])
        
    # 그래프 꾸미기
    plt.xlabel('Layer', fontsize=16)
    plt.ylabel('IICG', fontsize=16)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    # plt.title('Layer-wise IICG Comparison', fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(fontsize=14)
    
    # 축 범위 설정
    plt.xlim(0.5, 8.5)  # Layer 1부터 8까지
    
    # PDF로 저장
    output_path = os.path.join(output_dir, "layer_comparison_line_graph.pdf")
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()
    
    print(f"Saved line graph to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Compute Within–Between_all difference and save sorted JSON")
    parser.add_argument(
        "--twohop_root_dir", required=True,
        help="similarity_metrics_layer*.txt 들이 있는 최상위 디렉토리"
    )
    parser.add_argument(
        "--nontree_root_dir", required=True,
        help="similarity_metrics_layer*.txt 들이 있는 최상위 디렉토리"
    )
    parser.add_argument(
        "--output_dir", "-d", default="heatmaps",
        help="heatmap PDF 파일들을 저장할 디렉토리 (기본: heatmaps)"
    )
    args = parser.parse_args()
    
    assert "results" in args.twohop_root_dir and "results" in args.nontree_root_dir,  "It should be results directory"
    
    all_data_twohop = {}
    all_data_nontree = {}
    # twohop IICG data 정리
    for section in ["ID Test", "OOD"]:
        raw = collect_diffs_twohop(args.twohop_root_dir, section)
        sorted_data = sort_nested_twohop(raw)
        all_data_twohop[section] = sorted_data
    
    # nontree IICG data 정리
    for section in ["ID Test", "OOD"]:
        raw = collect_diffs_nontree(args.nontree_root_dir, section)
        sorted_data = sort_nested_nontree(raw)
        all_data_nontree[section] = sorted_data
        
    print(all_data_twohop)
    # print(all_data_nontree)
    
    filtered_data_twohop = filter_data(
        all_data_twohop, 
        sections=["ID Test"],
        cutoffs_or_ks=["default"], 
        fs=["f1", "f4"],
        steps=["stepfinal_checkpoint"], 
        layers=["layer1", "layer2", "layer3", "layer4", "layer5", "layer6", "layer7", "layer8"], 
        positions=["pos3"]
    )
    
    filtered_data_nontree = filter_data(
        all_data_nontree, 
        sections=["ID Test"],
        cutoffs_or_ks=["default"],
        fs=["f1", "f4"], 
        steps=["stepfinal_checkpoint"], 
        layers=["layer1", "layer2", "layer3", "layer4", "layer5", "layer6", "layer7", "layer8"], 
        positions=["pos3"]
    )
    
    print("\n", filtered_data_twohop)
    print("\n", filtered_data_nontree)
    
    # 꺾은 선 그래프 생성
    create_line_graph(filtered_data_twohop, filtered_data_nontree, args.output_dir)


if __name__ == "__main__":
    main()