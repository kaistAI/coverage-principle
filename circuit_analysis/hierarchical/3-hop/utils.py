import logging
import json

###############################################################################
def setup_logging(debug_mode):
    """Set up logging with DEBUG level if debug_mode is True, otherwise INFO level."""
    level = logging.DEBUG if debug_mode else logging.INFO
    logging.basicConfig(level=level, format='%(levelname)s - %(message)s')
###############################################################################

###############################################################################
def parse_tokens(text):
    tokens = text.replace("</a>", "").strip("><").split("><")
    return tokens
###############################################################################

###############################################################################
def load_atomic_facts_3hop(f1_path, f2_path, f3_path):
    """
    For 3-hop logic, parse the atomic facts files:
      - Subcomponent 1: (t1, t2) -> b1
      - Subcomponent 2: (b1, t3) -> b2
      - Subcomponent 3: (b2, t4) -> t_final
    Returns three dictionaries: f1_dict, f2_dict, f3_dict.
    """
    f1_dict, f2_dict, f3_dict = {}, {}, {}

    def parse_atomic_facts(file_path):
        """
        Parse an atomic fact file with the following format:
          "input_text": "<t_N1><t_N2>"
          "target_text": "<t_N1><t_N2><t_N3></a>"
        This corresponds to the mapping (t_N1, t_N2) -> t_N3
        """
        with open(file_path, "r") as f:
            facts = json.load(f)
        out_dict = {}
        for item in facts:
            inp_tokens = parse_tokens(item["input_text"])
            assert len(inp_tokens) == 2
            tgt_tokens = parse_tokens(item["target_text"])
            assert len(tgt_tokens) == 3 and inp_tokens == tgt_tokens[:2]
            out_dict[(tgt_tokens[0], tgt_tokens[1])] = tgt_tokens[-1]
        return out_dict

    f1_dict = parse_atomic_facts(f1_path)
    f2_dict = parse_atomic_facts(f2_path)
    f3_dict = parse_atomic_facts(f3_path)
    
    return f1_dict, f2_dict, f3_dict
###############################################################################

###############################################################################
def deduplicate_grouped_data(grouped_data, atomic_idx):
    """
    grouped_data: 그룹핑된 데이터. 형식은 { group_key: [entry, entry, ...] }이며,
                  각 entry는 "input_text"와 "target_text"를 포함하는 dict입니다.
    atomic_idx: deduplication 기준을 결정하는 인덱스
                - 1이면, target_text의 처음 두 토큰(t1, t2) 기준 deduplication
                - 2이면, 처음 세 토큰(t1, t2, t3) 기준 deduplication
                - 3이면, 처음 네 토큰(t1, t2, t3, t4) 기준 deduplication

    Returns:
        중복 제거된 entry들의 리스트. 동일한 deduplication 키를 가진 entry들은 하나만 남게 됩니다.
    """
    output = {}
    for group_key, entries in grouped_data.items():
        deduped = {}
        for entry in entries:
            tokens = parse_tokens(entry["target_text"])
            if atomic_idx == 1:
                dedup_key = tuple(tokens[:2])  # (t1, t2)
            elif atomic_idx == 2:
                dedup_key = tuple(tokens[:3])  # (t1, t2, t3)
            elif atomic_idx == 3:
                dedup_key = tuple(tokens[:4])  # (t1, t2, t3, t4)
            else:
                raise ValueError("atomic_idx must be 1, 2, or 3")

            if dedup_key not in deduped:
                deduped[dedup_key] = entry
        output[group_key] = list(deduped.values())

    return output
###############################################################################