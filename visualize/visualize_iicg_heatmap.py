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


def collect_diffs_parallel(root_dir, section):
    """
    parallel-2-hop 데이터셋을 위한 재귀 탐색으로 모든 similarity_metrics_layer*.txt 파일을 찾아
    (f, layer, pos, step) 별로 차이를 수집합니다.
    """
    result = {}
    # 파일 경로에서 f, (layer,pos), step 추출용 정규식
    re_f         = re.compile(r"[\\/](f\d+)[\\/]")
    re_layer_pos = re.compile(r"\((\d+|logit|prob),\s*(\d+)\)")
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
        layer_idx    = lp.group(1)              # e.g. "6"
        pos_idx      = lp.group(2)              # e.g. "2"
        step_key     = "step" + sm.group(1)     # e.g. "step25000" or "stepfinal_checkpoint"
        layer_key    = "layer" + layer_idx if layer_idx.isdigit() else layer_idx
        pos_key      = "pos"   + pos_idx
        
        result.setdefault(f_key, {})\
              .setdefault(step_key, {})\
              .setdefault(layer_key, {})[pos_key] = diff
              
    return result


def collect_diffs_threehop(root_dir, section):
    """
    3-hop 데이터셋을 위한 재귀 탐색으로 모든 similarity_metrics_layer*.txt 파일을 찾아
    (f, layer, pos, step) 별로 차이를 수집합니다.
    """
    result = {}
    # 파일 경로에서 f, (layer,pos), step 추출용 정규식
    re_f         = re.compile(r"[\\/](f\d+)[\\/]")
    re_layer_pos = re.compile(r"\((\d+|logit|prob),\s*(\d+)\)")
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
        layer_idx    = lp.group(1)              # e.g. "6"
        pos_idx      = lp.group(2)              # e.g. "2"
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


def sort_nested_parallel(data):
    """
    parallel-2-hop 데이터셋을 위한 정렬 함수
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
            # layer 키 정렬
            for layer in sorted(ldict.keys(), key=lambda x: int(x.replace("layer","")) if x.startswith("layer") else (9 if x == "logit" else 10)):
                out["default"][f_key][step][layer] = {}
                pdict = ldict[layer]
                # pos 키 정렬
                for pos in sorted(pdict.keys(), key=lambda x: int(x.replace("pos",""))):
                    out["default"][f_key][step][layer][pos] = pdict[pos]
    return out


def sort_nested_threehop(data):
    """
    3-hop 데이터셋을 위한 정렬 함수
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
            # layer 키 정렬
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


def create_heatmaps_twohop(id_data, ood_data, output_dir, vmax):
    """
    2-hop 데이터셋을 위한 ID Test와 OOD heatmap을 하나의 figure에 그립니다.
    ID Test는 detailed_grouping에 따라 다른 heatmap을, OOD는 하나의 heatmap만 표시합니다.
    모든 step에 대해 heatmap을 생성합니다.
    """
    # 출력 디렉토리가 없으면 생성
    os.makedirs(output_dir, exist_ok=True)
    
    # vmin 설정
    vmin = 0
    
    cmap = sns.light_palette("seagreen", as_cmap=True)
    # cmap = sns.color_palette("viridis", as_cmap=True)
    
    # detailed_grouping 판단
    if "default" in id_data:
        # detailed_grouping이 없는 경우
        sections = ["default"]
        section_titles = ["ID_test"]
    elif "low_cutoff" in id_data and "mid_cutoff" in id_data and "high_cutoff" in id_data:
        n_value = 2
        sections = ["low_cutoff", "mid_cutoff", "high_cutoff"]
        section_titles = ["$k < 3$", "$k = 3$", "$k > 3$"]
    elif "low_cutoff" in id_data and "high_cutoff" in id_data:
        n_value = 1
        sections = ["low_cutoff", "high_cutoff"]
        section_titles = ["$k < 3$", "$k \geq 3$"]
    elif "2" in id_data and "3" in id_data and "4" in id_data:
        n_value = 3
        sections = ["2", "3", "4"]
        section_titles = ["$k = 2$", "$k = 3$", "$k = 4$"]
    else:
        raise ValueError("Unknown detailed_grouping pattern in data")
    
    # 사용 가능한 f_key 목록 가져오기
    available_f_keys = sorted([f for f in next(iter(id_data.values())).keys() if f in next(iter(id_data.values())).keys()])
    print(available_f_keys)
    num_f_keys = len(available_f_keys)
    
    if num_f_keys == 0:
        print("No valid f_key pairs found in both ID and OOD data")
        return
    
    # 모든 step에 대해 처리
    for step in sorted(next(iter(next(iter(id_data.values())).values())).keys(), key=lambda x: float('inf') if x == "stepfinal_checkpoint" else int(x.replace("step",""))):
        # 전체 figure 생성 (ID Test N개 + OOD 1개 + colorbar)
        fig_height = 4
        if "default" in id_data:
            fig_width = 2.7 * (len(sections) + 1)  # ID Test N개 + OOD 1개
        else:
            fig_width = 1.8 * (len(sections) + 1)  # ID Test N개 + OOD 1개
        # fig_height = fig_width * 0.66
        fig = plt.figure(figsize=(fig_width, fig_height))
        gs = gridspec.GridSpec(1, len(sections) + 2, width_ratios=[1] * (len(sections) + 1) + [0.1], wspace=0.2)
        axes = [plt.subplot(gs[i]) for i in range(len(sections) + 2)]
        
        # ID Test heatmap 그리기
        for idx, (section, title) in enumerate(zip(sections, section_titles)):
            if section not in id_data:
                continue
            layer_data = id_data[section]["f1"][step]
            ax = axes[idx]
            
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
                       xticklabels=["$x1$", "$x2$", "$x3$", "$b$"] if "default" in id_data else [f"$x{i+1}$" for i in range(len(positions))],
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
        
        # OOD heatmap 그리기 (마지막 subplot)
        ood_ax = axes[-2]
        # OOD 데이터에서 첫 번째 cutoff의 데이터 사용 (어떤 cutoff든 같은 값)
        ood_cutoff = next(iter(ood_data))
        ood_layer_data = ood_data[ood_cutoff]["f1"][step]
        
        # OOD heatmap 데이터 준비
        layers = sorted(ood_layer_data.keys(), key=lambda x: int(x.replace("layer","")), reverse=True)
        positions = sorted(set(pos for layer in ood_layer_data.values() for pos in layer.keys()), 
                        key=lambda x: int(x.replace("pos","")))
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
                    fmt='.2f',
                    cmap=cmap,
                    vmin=vmin,
                    vmax=vmax,
                    xticklabels=["$x1$", "$x2$", "$x3$", "$b$"] if "default" in id_data else [f"$x{i+1}$" for i in range(len(positions))],
                    yticklabels=[],  # OOD heatmap에는 y축 레이블 표시하지 않음
                    cbar=False,
                    linewidths=0.5,
                    linecolor='lightgray',
                    ax=ood_ax,
                    square=True)
        
        # OOD 최대값 셀에만 값 표시
        for i, j in zip(*ood_max_indices):
            ood_ax.add_patch(plt.Rectangle((j, i), 1, 1, fill=False, edgecolor='black', lw=2))
            ood_ax.text(j + 0.5, i + 0.5, f'{ood_max_value:.2f}',
                       ha='center', va='center',
                       color='black', fontweight='bold', fontsize=11)
        
        # OOD tick 제거
        ood_ax.tick_params(axis='both', which='both', length=0)
        
        # OOD x축 레이블 폰트 크기 설정
        ood_ax.set_xticklabels(ood_ax.get_xticklabels(), fontsize=16)
        
        # subplot 영역 테두리 추가
        for spine in ood_ax.spines.values():
            spine.set_visible(True)
            spine.set_color('black')
            spine.set_linewidth(1.0)
        
        # OOD 제목 추가
        ood_ax.text(0.5, -0.18, "OOD",
                    ha='center', va='center',
                    transform=ood_ax.transAxes,
                    fontsize=20)
        
        # colorbar 추가
        cbar_ax = axes[-1]  # colorbar는 마지막 subplot
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=vmin, vmax=vmax))
        cbar = fig.colorbar(sm, cax=cbar_ax)
        cbar.ax.tick_params(labelsize=16)
        
        # IICG 레이블 추가
        cbar.ax.text(5.5, 0.5, 'IICG', ha='left', va='center', fontsize=20, transform=cbar.ax.transAxes)
        
        # 점선을 figure 전체에 걸쳐 그리기 위해 figure 좌표계 사용
        if "default" in id_data:
            fig.add_artist(plt.Line2D([0.525, 0.525], [0.05, 0.97], color='gray', linestyle='--', alpha=0.5, transform=fig.transFigure))
        elif n_value == 1:
            fig.add_artist(plt.Line2D([0.65, 0.65], [0.05, 0.97], color='gray', linestyle='--', alpha=0.5, transform=fig.transFigure))
        elif n_value == 2:
            fig.add_artist(plt.Line2D([0.682, 0.682], [-0.05, 0.89], color='gray', linestyle='--', alpha=0.5, transform=fig.transFigure))
        elif n_value == 3:
            fig.add_artist(plt.Line2D([0.75, 0.75], [0.05, 0.97], color='gray', linestyle='--', alpha=0.5, transform=fig.transFigure))
        
        # 전체 figure 조정
        plt.subplots_adjust(wspace=0.001)  # subplot 간의 가로 간격을 매우 좁게 설정
        plt.tight_layout(rect=[0, 0, 0.95, 1], w_pad=0.005)
        
        # PDF로 저장
        if "default" in id_data:
            output_path = os.path.join(output_dir, f"twohop_iicg_heatmap_step{step.replace('step','')}.pdf")
        else:
            output_path = os.path.join(output_dir, f"twohop_iicg_heatmap_detailed_grouping_{n_value}_step{step.replace('step','')}.pdf")
        plt.savefig(output_path, bbox_inches='tight')
        plt.close()
        print(f"Saved combined heatmap to {output_path}")

def create_heatmaps_nontree(id_data, ood_data, output_dir, vmax):
    """
    nontree 데이터셋을 위한 ID Test와 OOD heatmap을 하나의 figure에 그립니다.
    f_key의 개수에 따라 동적으로 히트맵을 생성합니다.
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
    
    # 사용 가능한 f_key 목록 가져오기
    available_f_keys = sorted([f for f in id_data["default"].keys() if f in ood_data["default"].keys()])
    num_f_keys = len(available_f_keys)
    
    if num_f_keys == 0:
        print("No valid f_key pairs found in both ID and OOD data")
        return
    
    for step in sorted(next(iter(next(iter(id_data.values())).values())).keys(), key=lambda x: float('inf') if x == "stepfinal_checkpoint" else int(x.replace("step",""))):
        # 전체 figure 생성 (각 f_key마다 ID Test/OOD)
        fig_width = 1.8 * (num_f_keys * 2)  # 각 f_key마다 ID Test와 OOD
        fig_height = 4
        fig = plt.figure(figsize=(fig_width, fig_height))
        gs = gridspec.GridSpec(1, num_f_keys * 2, width_ratios=[1] * (num_f_keys * 2), wspace=0.0)
        axes = [plt.subplot(gs[i]) for i in range(num_f_keys * 2)]
        
        # 각 f_key에 대해 처리
        for idx, f_key in enumerate(available_f_keys):
            id_ax = axes[idx * 2]
            id_layer_data = id_data["default"][f_key][step]
            
            # ID Test heatmap 데이터 준비
            layers = sorted(id_layer_data.keys(), key=lambda x: int(x.replace("layer","")) if x.startswith("layer") else (9 if x == "logit" else 10), reverse=True)
            positions = sorted(set(pos for layer in id_layer_data.values() for pos in layer.keys()), 
                            key=lambda x: int(x.replace("pos","")))
            # heatmap 데이터 준비
            id_heatmap_data = np.zeros((len(layers), len(positions)))
            for i, layer in enumerate(layers):
                for j, pos in enumerate(positions):
                    if layer in id_layer_data and pos in id_layer_data[layer]:
                        id_heatmap_data[i, j] = id_layer_data[layer][pos]
            
            # ID Test 최대값 찾기
            id_max_value = np.max(id_heatmap_data)
            id_max_indices = np.where(id_heatmap_data == id_max_value)
            
            # ID Test heatmap 생성
            sns.heatmap(id_heatmap_data,
                        annot=False,
                        fmt='.2f',
                        cmap=cmap,
                        vmin=vmin,
                        vmax=vmax,
                        xticklabels=["x1\npos1", "x2\npos2", "x3\npos3", "b\npos4"] if len(positions) == 4 else [f"x{i+1}\npos{i+1}" for i in range(len(positions))],
                        yticklabels=[f"Layer {i}" for i in range(len(layers), 0, -1)] if idx == 0 else [],  # 가장 왼쪽 heatmap에만 y축 레이블 표시
                        cbar=False,
                        linewidths=0.5,
                        linecolor='lightgray',
                        ax=id_ax,
                        square=True)
            
            # y축 레이블 회전 (가로로 표시)
            if idx == 0:  # 첫 번째 heatmap에만 적용
                id_ax.set_yticklabels(id_ax.get_yticklabels(), rotation=0, fontsize=16)
                
            # x축 레이블 폰트 크기 설정
            id_ax.set_xticklabels(id_ax.get_xticklabels(), fontsize=16)
            
            # ID Test 최대값 셀에만 값 표시
            for i, j in zip(*id_max_indices):
                id_ax.add_patch(plt.Rectangle((j, i), 1, 1, fill=False, edgecolor='black', lw=2))
                id_ax.text(j + 0.5, i + 0.5, f'{id_max_value:.2f}',
                        ha='center', va='center',
                        color='black', fontweight='bold', fontsize=11)
            
            # ID Test tick 제거
            id_ax.tick_params(axis='both', which='both', length=0)
            
            # subplot 영역 테두리 추가
            for spine in id_ax.spines.values():
                spine.set_visible(True)
                spine.set_color('black')
                spine.set_linewidth(1.0)
            
            # ID Test 제목 추가
            id_ax.text(0.5, -0.18, "ID Test",
                        ha='center', va='center',
                        transform=id_ax.transAxes,
                        fontsize=20)
            
            # OOD heatmap 그리기
            ood_ax = axes[idx * 2 + 1]
            ood_layer_data = ood_data["default"][f_key][step]
            
            # OOD heatmap 데이터 준비
            layers = sorted(ood_layer_data.keys(), key=lambda x: int(x.replace("layer","")) if x.startswith("layer") else (9 if x == "logit" else 10), reverse=True)
            positions = sorted(set(pos for layer in ood_layer_data.values() for pos in layer.keys()), 
                            key=lambda x: int(x.replace("pos","")))
            ood_heatmap_data = np.zeros((len(layers), len(positions)))
            for i, layer in enumerate(layers):
                for j, pos in enumerate(positions):
                    if layer in ood_layer_data and pos in ood_layer_data[layer]:
                        ood_heatmap_data[i, j] = ood_layer_data[layer][pos]
            
            # OOD 최대값 찾기
            ood_max_value = np.max(ood_heatmap_data)
            ood_max_indices = np.where(ood_heatmap_data == ood_max_value)
            
            # OOD heatmap 생성
            sns.heatmap(ood_heatmap_data,
                        annot=False,
                        fmt='.2f',
                        cmap=cmap,
                        vmin=vmin,
                        vmax=vmax,
                        xticklabels=["x1\npos1", "x2\npos2", "x3\npos3", "b\npos4"] if len(positions) == 4 else [f"x{i+1}\npos{i+1}" for i in range(len(positions))],
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
                        color='black', fontweight='bold', fontsize=11)
            
            # OOD tick 제거
            ood_ax.tick_params(axis='both', which='both', length=0)
            
            # OOD x축 레이블 폰트 크기 설정
            ood_ax.set_xticklabels(ood_ax.get_xticklabels(), fontsize=16)
            
            # subplot 영역 테두리 추가
            for spine in ood_ax.spines.values():
                spine.set_visible(True)
                spine.set_color('black')
                spine.set_linewidth(1.0)
            
            # OOD 제목 추가
            ood_ax.text(0.5, -0.2, "OOD",
                        ha='center', va='center',
                        transform=ood_ax.transAxes,
                        fontsize=20)
        
        # # f_key 사이에 세로 점선 추가 (figure 전체에 걸쳐)
        # for i in range(1, num_f_keys):
        #     x_pos = i * 2 / (num_f_keys * 2)
        #     fig.add_artist(plt.Line2D([x_pos, x_pos], [0.05, 0.97], color='gray', linestyle='--', alpha=0.5, transform=fig.transFigure))
        
        # f_key 텍스트 추가
        if num_f_keys == 5:
            start_point = 0.113
            interval = 0.185
        elif num_f_keys == 4:
            start_point = 0.138
            interval = 0.231
        elif num_f_keys == 3:
            start_point = 0.113
            interval = 0.265
        else:
            raise NotImplementedError(f"Interval for num_f_keys: {num_f_keys} not implemented")
            
        for i, f_key in enumerate(available_f_keys):
            x_pos = start_point + i * interval
            display_text = f_key_to_text.get(f_key, f_key)  # 매핑된 텍스트가 없으면 f_key 사용
            fig.text(x_pos, -0.05, display_text, ha='center', va='center', fontsize=16)
        
        # 전체 figure에 대한 colorbar 추가
        cbar_ax = fig.add_axes([0.97, 0.23, 0.02, 0.7])
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=vmin, vmax=vmax))
        cbar = fig.colorbar(sm, cax=cbar_ax)
        cbar.ax.tick_params(labelsize=16)
        
        # IICG 레이블 추가
        cbar.ax.text(5.5, 0.5, 'IICG', ha='left', va='center', fontsize=20, transform=cbar.ax.transAxes)
        
        # 점선을 figure 전체에 걸쳐 그리기 위해 figure 좌표계 사용
        fig.add_artist(plt.Line2D([0.32, 0.32], [-0.18, 0.86], color='gray', linestyle='--', alpha=0.5, transform=fig.transFigure))
        fig.add_artist(plt.Line2D([0.55, 0.55], [-0.18, 0.86], color='gray', linestyle='--', alpha=0.5, transform=fig.transFigure))
        
        # 전체 figure 조정
        plt.subplots_adjust(wspace=0.001)  # subplot 간의 가로 간격을 매우 좁게 설정
        plt.tight_layout(rect=[0, 0, 0.95, 1], w_pad=0.005)
        
        # PDF로 저장
        output_path = os.path.join(output_dir, f"nontree_iicg_heatmap_step{step.replace('step','')}.pdf")
        plt.savefig(output_path, bbox_inches='tight')
        plt.close()
        print(f"Saved combined heatmap to {output_path}")


def create_heatmaps_parallel(id_data, ood_data, output_dir, vmax):
    """
    parallel-2-hop 데이터셋을 위한 ID Test와 OOD heatmap을 하나의 figure에 그립니다.
    f_key의 개수에 따라 동적으로 히트맵을 생성합니다.
    """
    # 출력 디렉토리가 없으면 생성
    os.makedirs(output_dir, exist_ok=True)
    
    # vmin 설정
    vmin = 0
    
    cmap = sns.light_palette("seagreen", as_cmap=True)
    
    # f_key에 따른 텍스트 매핑
    f_key_to_text = {
        "f1": "b1",
        "f2": "b2",
        "f3": "t"
    }
    
    # 사용 가능한 f_key 목록 가져오기
    available_f_keys = sorted([f for f in id_data["default"].keys() if f in ood_data["default"].keys()])
    num_f_keys = len(available_f_keys)
    
    if num_f_keys == 0:
        print("No valid f_key pairs found in both ID and OOD data")
        return
    
    for step in sorted(next(iter(next(iter(id_data.values())).values())).keys(), key=lambda x: float('inf') if x == "stepfinal_checkpoint" else int(x.replace("step",""))):
        # 전체 figure 생성
        fig_width = 2.4 * (num_f_keys * 2)  # 각 f_key마다 하나의 subplot
        fig_height = 4
        fig = plt.figure(figsize=(fig_width, fig_height))
        
        # outer GridSpec 생성 (1행, num_f_keys열)
        outer_gs = gridspec.GridSpec(1, num_f_keys+1, width_ratios=[1] * num_f_keys + [0.1], wspace=0.2)
        outer_axes = [plt.subplot(outer_gs[i]) for i in range(num_f_keys + 1)]
        
        # outer_axes 숨기기
        for ax in outer_axes:
            if ax != outer_axes[-1]:
                ax.set_axis_off()
                ax.set_xticks([])
                ax.set_yticks([])
                for spine in ax.spines.values():
                    spine.set_visible(False)
        
        # 각 f_key에 대해 처리
        for idx, f_key in enumerate(available_f_keys):
            # 각 f_key에 대한 inner GridSpec 생성 (1행, 2열)
            inner_gs = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=outer_gs[idx], wspace=0.1)
            
            # ID Test heatmap
            id_ax = plt.subplot(inner_gs[0])
            id_layer_data = id_data["default"][f_key][step]
            
            # ID Test heatmap 데이터 준비
            layers = sorted(id_layer_data.keys(), key=lambda x: int(x.replace("layer","")) if x.startswith("layer") else (9 if x == "logit" else 10), reverse=True)
            positions = sorted(set(pos for layer in id_layer_data.values() for pos in layer.keys()), 
                            key=lambda x: int(x.replace("pos","")))
            # heatmap 데이터 준비
            id_heatmap_data = np.zeros((len(layers), len(positions)))
            for i, layer in enumerate(layers):
                for j, pos in enumerate(positions):
                    if layer in id_layer_data and pos in id_layer_data[layer]:
                        id_heatmap_data[i, j] = id_layer_data[layer][pos]
            
            # ID Test 최대값 찾기
            id_max_value = np.max(id_heatmap_data)
            id_max_indices = np.where(id_heatmap_data == id_max_value)
            
            # ID Test heatmap 생성
            sns.heatmap(id_heatmap_data,
                        annot=False,
                        fmt='.2f',
                        cmap=cmap,
                        vmin=vmin,
                        vmax=vmax,
                        xticklabels=["x1", "x2", "x3", "x4"],
                        yticklabels=[f"Layer {i}" for i in range(len(layers), 0, -1)] if idx == 0 else [],  # 가장 왼쪽 heatmap에만 y축 레이블 표시
                        cbar=False,
                        linewidths=0.5,
                        linecolor='lightgray',
                        ax=id_ax,
                        square=True)
            
            # y축 레이블 회전 (가로로 표시)
            if idx == 0:  # 첫 번째 heatmap에만 적용
                id_ax.set_yticklabels(id_ax.get_yticklabels(), rotation=0, fontsize=16)
                
            # x축 레이블 폰트 크기 설정
            id_ax.set_xticklabels(id_ax.get_xticklabels(), fontsize=16)
            
            # ID Test 최대값 셀에만 값 표시
            for i, j in zip(*id_max_indices):
                id_ax.add_patch(plt.Rectangle((j, i), 1, 1, fill=False, edgecolor='black', lw=2))
                id_ax.text(j + 0.5, i + 0.5, f'{id_max_value:.2f}',
                        ha='center', va='center',
                        color='black', fontweight='bold', fontsize=11)
            
            # ID Test tick 제거
            id_ax.tick_params(axis='both', which='both', length=0)
            
            # subplot 영역 테두리 추가
            for spine in id_ax.spines.values():
                spine.set_visible(True)
                spine.set_color('black')
                spine.set_linewidth(1.0)
            
            # ID Test 제목 추가
            id_ax.text(0.5, -0.18, "ID Test",
                        ha='center', va='center',
                        transform=id_ax.transAxes,
                        fontsize=20)
            
            # OOD heatmap
            ood_ax = plt.subplot(inner_gs[1])
            ood_layer_data = ood_data["default"][f_key][step]
            
            # OOD heatmap 데이터 준비
            layers = sorted(ood_layer_data.keys(), key=lambda x: int(x.replace("layer","")) if x.startswith("layer") else (9 if x == "logit" else 10), reverse=True)
            positions = sorted(set(pos for layer in ood_layer_data.values() for pos in layer.keys()), 
                            key=lambda x: int(x.replace("pos","")))
            ood_heatmap_data = np.zeros((len(layers), len(positions)))
            for i, layer in enumerate(layers):
                for j, pos in enumerate(positions):
                    if layer in ood_layer_data and pos in ood_layer_data[layer]:
                        ood_heatmap_data[i, j] = ood_layer_data[layer][pos]
            
            # OOD 최대값 찾기
            ood_max_value = np.max(ood_heatmap_data)
            ood_max_indices = np.where(ood_heatmap_data == ood_max_value)
            
            # OOD heatmap 생성
            sns.heatmap(ood_heatmap_data,
                        annot=False,
                        fmt='.2f',
                        cmap=cmap,
                        vmin=vmin,
                        vmax=vmax,
                        xticklabels=["x1", "x2", "x3", "x4"],
                        yticklabels=[],  # OOD heatmap에는 y축 레이블 표시하지 않음
                        cbar=False,
                        linewidths=0.5,
                        linecolor='lightgray',
                        ax=ood_ax,
                        square=True)
            
            # OOD 최대값 셀에만 값 표시
            for i, j in zip(*ood_max_indices):
                ood_ax.add_patch(plt.Rectangle((j, i), 1, 1, fill=False, edgecolor='black', lw=2))
                ood_ax.text(j + 0.5, i + 0.5, f'{ood_max_value:.2f}',
                        ha='center', va='center',
                        color='black', fontweight='bold', fontsize=11)
            
            # OOD tick 제거
            ood_ax.tick_params(axis='both', which='both', length=0)
            
            # OOD x축 레이블 폰트 크기 설정
            ood_ax.set_xticklabels(ood_ax.get_xticklabels(), fontsize=16)
            
            # subplot 영역 테두리 추가
            for spine in ood_ax.spines.values():
                spine.set_visible(True)
                spine.set_color('black')
                spine.set_linewidth(1.0)
            
            # OOD 제목 추가
            ood_ax.text(0.5, -0.18, "OOD",
                        ha='center', va='center',
                        transform=ood_ax.transAxes,
                        fontsize=20)
            
            outer_axes[idx].text(0.5, -0.3, f_key_to_text[f_key],
                        ha='center', va='center',
                        transform=outer_axes[idx].transAxes,
                        fontsize=20)
        
        # colorbar 추가
        cbar_ax = plt.subplot(outer_gs[-1])
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=vmin, vmax=vmax))
        cbar = fig.colorbar(sm, cax=cbar_ax)
        cbar.ax.tick_params(labelsize=16)
        
        # IICG 레이블 추가
        cbar.ax.text(3.5, 0.5, 'IICG', ha='left', va='center', fontsize=20, transform=cbar.ax.transAxes)
        
        # 점선을 figure 전체에 걸쳐 그리기 위해 figure 좌표계 사용
        fig.add_artist(plt.Line2D([0.36, 0.36], [-0.17, 0.86], color='gray', linestyle='--', alpha=0.5, transform=fig.transFigure))
        fig.add_artist(plt.Line2D([0.61, 0.61], [-0.17, 0.86], color='gray', linestyle='--', alpha=0.5, transform=fig.transFigure))
        
        # 전체 figure 조정
        plt.subplots_adjust(wspace=0.001)  # subplot 간의 가로 간격을 매우 좁게 설정
        plt.tight_layout(rect=[0, 0, 0.95, 1], w_pad=0.005)
        
        # PDF로 저장
        output_path = os.path.join(output_dir, f"parallel2hop_iicg_heatmap_step{step.replace('step','')}.pdf")
        plt.savefig(output_path, bbox_inches='tight')
        plt.close()
        print(f"Saved combined heatmap to {output_path}")


def create_heatmaps_threehop(id_data, ood_data, output_dir, vmax):
    """
    3-hop 데이터셋을 위한 ID Test와 OOD heatmap을 하나의 figure에 그립니다.
    f_key의 개수에 따라 동적으로 히트맵을 생성합니다.
    """
    # 출력 디렉토리가 없으면 생성
    os.makedirs(output_dir, exist_ok=True)
    
    # vmin 설정
    vmin = 0
    
    cmap = sns.light_palette("seagreen", as_cmap=True)
    
    # f_key에 따른 텍스트 매핑
    f_key_to_text = {
        "f1": "b1",
        "f2": "b2",
        "f3": "t"
    }
    
    # 사용 가능한 f_key 목록 가져오기
    available_f_keys = sorted([f for f in id_data["default"].keys() if f in ood_data["default"].keys()])
    num_f_keys = len(available_f_keys)
    
    if num_f_keys == 0:
        print("No valid f_key pairs found in both ID and OOD data")
        return
    
    for step in sorted(next(iter(next(iter(id_data.values())).values())).keys(), key=lambda x: float('inf') if x == "stepfinal_checkpoint" else int(x.replace("step",""))):
        # 전체 figure 생성
        fig_width = 2.4 * (num_f_keys * 2)  # 각 f_key마다 하나의 subplot
        fig_height = 4
        fig = plt.figure(figsize=(fig_width, fig_height))
        
        # outer GridSpec 생성 (1행, num_f_keys열)
        outer_gs = gridspec.GridSpec(1, num_f_keys+1, width_ratios=[1] * num_f_keys + [0.1], wspace=0.2)
        outer_axes = [plt.subplot(outer_gs[i]) for i in range(num_f_keys + 1)]
        
        # outer_axes 숨기기
        for ax in outer_axes:
            if ax != outer_axes[-1]:
                ax.set_axis_off()
                ax.set_xticks([])
                ax.set_yticks([])
                for spine in ax.spines.values():
                    spine.set_visible(False)
        
        # 각 f_key에 대해 처리
        for idx, f_key in enumerate(available_f_keys):
            # 각 f_key에 대한 inner GridSpec 생성 (1행, 2열)
            inner_gs = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=outer_gs[idx], wspace=0.1)
            
            # ID Test heatmap
            id_ax = plt.subplot(inner_gs[0])
            id_layer_data = id_data["default"][f_key][step]
            
            # ID Test heatmap 데이터 준비
            layers = sorted(id_layer_data.keys(), key=lambda x: int(x.replace("layer","")) if x.startswith("layer") else (9 if x == "logit" else 10), reverse=True)
            positions = sorted(set(pos for layer in id_layer_data.values() for pos in layer.keys()), 
                            key=lambda x: int(x.replace("pos","")))
            # heatmap 데이터 준비
            id_heatmap_data = np.zeros((len(layers), len(positions)))
            for i, layer in enumerate(layers):
                for j, pos in enumerate(positions):
                    if layer in id_layer_data and pos in id_layer_data[layer]:
                        id_heatmap_data[i, j] = id_layer_data[layer][pos]
            
            # ID Test 최대값 찾기
            id_max_value = np.max(id_heatmap_data)
            id_max_indices = np.where(id_heatmap_data == id_max_value)
            
            # ID Test heatmap 생성
            sns.heatmap(id_heatmap_data,
                        annot=False,
                        fmt='.2f',
                        cmap=cmap,
                        vmin=vmin,
                        vmax=vmax,
                        xticklabels=["x1", "x2", "x3", "x4"],
                        yticklabels=[f"Layer {i}" for i in range(len(layers), 0, -1)] if idx == 0 else [],  # 가장 왼쪽 heatmap에만 y축 레이블 표시
                        cbar=False,
                        linewidths=0.5,
                        linecolor='lightgray',
                        ax=id_ax,
                        square=True)
            
            # y축 레이블 회전 (가로로 표시)
            if idx == 0:  # 첫 번째 heatmap에만 적용
                id_ax.set_yticklabels(id_ax.get_yticklabels(), rotation=0, fontsize=16)
                
            # x축 레이블 폰트 크기 설정
            id_ax.set_xticklabels(id_ax.get_xticklabels(), fontsize=16)
            
            # ID Test 최대값 셀에만 값 표시
            for i, j in zip(*id_max_indices):
                id_ax.add_patch(plt.Rectangle((j, i), 1, 1, fill=False, edgecolor='black', lw=2))
                id_ax.text(j + 0.5, i + 0.5, f'{id_max_value:.2f}',
                        ha='center', va='center',
                        color='black', fontweight='bold', fontsize=11)
            
            # ID Test tick 제거
            id_ax.tick_params(axis='both', which='both', length=0)
            
            # subplot 영역 테두리 추가
            for spine in id_ax.spines.values():
                spine.set_visible(True)
                spine.set_color('black')
                spine.set_linewidth(1.0)
            
            # ID Test 제목 추가
            id_ax.text(0.5, -0.18, "ID Test",
                        ha='center', va='center',
                        transform=id_ax.transAxes,
                        fontsize=20)
            
            # OOD heatmap
            ood_ax = plt.subplot(inner_gs[1])
            ood_layer_data = ood_data["default"][f_key][step]
            
            # OOD heatmap 데이터 준비
            layers = sorted(ood_layer_data.keys(), key=lambda x: int(x.replace("layer","")) if x.startswith("layer") else (9 if x == "logit" else 10), reverse=True)
            positions = sorted(set(pos for layer in ood_layer_data.values() for pos in layer.keys()), 
                            key=lambda x: int(x.replace("pos","")))
            ood_heatmap_data = np.zeros((len(layers), len(positions)))
            for i, layer in enumerate(layers):
                for j, pos in enumerate(positions):
                    if layer in ood_layer_data and pos in ood_layer_data[layer]:
                        ood_heatmap_data[i, j] = ood_layer_data[layer][pos]
            
            # OOD 최대값 찾기
            ood_max_value = np.max(ood_heatmap_data)
            ood_max_indices = np.where(ood_heatmap_data == ood_max_value)
            
            # OOD heatmap 생성
            sns.heatmap(ood_heatmap_data,
                        annot=False,
                        fmt='.2f',
                        cmap=cmap,
                        vmin=vmin,
                        vmax=vmax,
                        xticklabels=["x1", "x2", "x3", "x4"],
                        yticklabels=[],  # OOD heatmap에는 y축 레이블 표시하지 않음
                        cbar=False,
                        linewidths=0.5,
                        linecolor='lightgray',
                        ax=ood_ax,
                        square=True)
            
            # OOD 최대값 셀에만 값 표시
            for i, j in zip(*ood_max_indices):
                ood_ax.add_patch(plt.Rectangle((j, i), 1, 1, fill=False, edgecolor='black', lw=2))
                ood_ax.text(j + 0.5, i + 0.5, f'{ood_max_value:.2f}',
                        ha='center', va='center',
                        color='black', fontweight='bold', fontsize=11)
            
            # OOD tick 제거
            ood_ax.tick_params(axis='both', which='both', length=0)
            
            # OOD x축 레이블 폰트 크기 설정
            ood_ax.set_xticklabels(ood_ax.get_xticklabels(), fontsize=16)
            
            # subplot 영역 테두리 추가
            for spine in ood_ax.spines.values():
                spine.set_visible(True)
                spine.set_color('black')
                spine.set_linewidth(1.0)
            
            # OOD 제목 추가
            ood_ax.text(0.5, -0.18, "OOD",
                        ha='center', va='center',
                        transform=ood_ax.transAxes,
                        fontsize=20)
            
            outer_axes[idx].text(0.5, -0.3, f_key_to_text[f_key],
                        ha='center', va='center',
                        transform=outer_axes[idx].transAxes,
                        fontsize=20)
        
        # colorbar 추가
        cbar_ax = plt.subplot(outer_gs[-1])
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=vmin, vmax=vmax))
        cbar = fig.colorbar(sm, cax=cbar_ax)
        cbar.ax.tick_params(labelsize=16)
        
        # IICG 레이블 추가
        cbar.ax.text(3.5, 0.5, 'IICG', ha='left', va='center', fontsize=20, transform=cbar.ax.transAxes)
        
        # 점선을 figure 전체에 걸쳐 그리기 위해 figure 좌표계 사용
        fig.add_artist(plt.Line2D([0.32, 0.32], [-0.18, 0.86], color='gray', linestyle='--', alpha=0.5, transform=fig.transFigure))
        fig.add_artist(plt.Line2D([0.55, 0.55], [-0.18, 0.86], color='gray', linestyle='--', alpha=0.5, transform=fig.transFigure))
        
        # 전체 figure 조정
        plt.subplots_adjust(wspace=0.001)  # subplot 간의 가로 간격을 매우 좁게 설정
        plt.tight_layout(rect=[0, 0, 0.95, 1], w_pad=0.005)
        
        # PDF로 저장
        output_path = os.path.join(output_dir, f"threehop_iicg_heatmap_step{step.replace('step','')}.pdf")
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
        "--output_dir", "-d", default="heatmaps",
        help="heatmap PDF 파일들을 저장할 디렉토리 (기본: heatmaps)"
    )
    args = parser.parse_args()
    
    assert "results" in args.root_dir, "It should be results directory"

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
    elif "parallel" in args.root_dir:
        dataset_type = "parallel"
        collect_diffs_fn = collect_diffs_parallel
        sort_nested_fn = sort_nested_parallel
        create_heatmaps_fn = create_heatmaps_parallel
    elif "3-hop" in args.root_dir:
        dataset_type = "threehop"
        collect_diffs_fn = collect_diffs_threehop
        sort_nested_fn = sort_nested_threehop
        create_heatmaps_fn = create_heatmaps_threehop
    else:
        raise ValueError(f"Unknown dataset type in path: {args.root_dir}")

    all_data = {}
    for section in ["ID Test", "OOD"]:
        raw = collect_diffs_fn(args.root_dir, section)
        sorted_data = sort_nested_fn(raw)
        all_data[section] = sorted_data
    
    #########################################################
    # Should change here! 
    #########################################################
    filtered_data = filter_data(
        all_data, 
        sections=["ID Test", "OOD"],
        fs=["f1", "f2", "f3"],
        steps=["stepfinal_checkpoint"], 
        layers=["layer1", "layer2", "layer3", "layer4", "layer5", "layer6", "layer7", "layer8"]
    )
    print(filtered_data)
    
    all_values = []
    # Maximum value for figure's colorbar
    for section_dict in filtered_data.values():
        for cutoff_or_k_dict in section_dict.values():
            for f_key_dict in cutoff_or_k_dict.values():
                for step_dict in f_key_dict.values():
                    for layer_dict in step_dict.values():
                        for pos, value in layer_dict.items():
                            if isinstance(value, (int, float)):
                                all_values.append(value)
    # vmax = max(all_values)
    vmax=0.5
    
    output_dir_path = os.path.join(args.output_dir, args.root_dir.split("/")[-1])
    print(output_dir_path)
    
    # ID Test와 OOD 데이터를 하나의 figure로 그리기
    create_heatmaps_fn(filtered_data["ID Test"], filtered_data["OOD"], output_dir_path, vmax)

if __name__ == "__main__":
    main()