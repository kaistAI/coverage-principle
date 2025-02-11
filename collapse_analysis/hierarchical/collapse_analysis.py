import argparse
import torch
import json
from tqdm import tqdm
import os
from transformers import GPT2Config, GPT2LMHeadModel, GPT2Tokenizer
import logging
from collections import defaultdict
import re

###############################################################################
# 1) Loading atomic facts for 3-hop: (h1,h2)->b1, (b1,h3)->b2, (b2,h4)->t
###############################################################################
def load_atomic_facts_3hop(f1_path, f2_path, f3_path):
    """
    For 3-hop logic, we parse:
      subcomp1: (h1,h2)-> b1
      subcomp2: (b1,h3)-> b2
      subcomp3: (b2,h4)-> t
    Return three dicts: f1_dict, f2_dict, f3_dict
    """
    f1_dict, f2_dict, f3_dict = {}, {}, {}

    def parse_atomic_facts(file_path):
        """
        Format: "input_text": "<x><y>", "target_text": "<x><y><z></a>"
        => (x,y)-> z
        """
        with open(file_path, "r") as f:
            facts = json.load(f)
        out_dict = {}
        for item in facts:
            inp = item["input_text"].strip("<>").split("><")
            if len(inp) == 2:
                x, y = inp
                tgt = item["target_text"].replace("</a>", "").strip("<>").split("><")
                if len(tgt) == 3:
                    z = tgt[-1]
                    out_dict[(x,y)] = z
        return out_dict

    f1_dict = parse_atomic_facts(f1_path)
    f2_dict = parse_atomic_facts(f2_path)
    f3_dict = parse_atomic_facts(f3_path)
    
    return f1_dict, f2_dict, f3_dict

###############################################################################
# 2) Helpers for 3-hop input parsing & grouping
###############################################################################
def parse_3hop_input(input_text):
    """
    e.g. "<t_7><t_11><t_99><t_25>" => ["t_7","t_11","t_99","t_25"]
    returns None if not 4 tokens
    """
    tokens = input_text.strip("<>").split("><")
    if len(tokens) != 4:
        return None
    return tokens

def group_data_by_b1(examples, f1_dict):
    """
    For each example: parse (h1,h2,h3,h4),
    b1= f1_dict.get((h1,h2),'unknown'),
    group by b1
    => group_dict[b1] = list of examples
    """
    group_dict = defaultdict(list)
    for ex in examples:
        inp_tokens = parse_3hop_input(ex["input_text"])
        if not inp_tokens:
            continue
        h1,h2,h3,h4 = inp_tokens
        b1 = f1_dict.get((h1,h2), "unknown")
        group_dict[b1].append(ex)
    return dict(group_dict)

def group_data_by_b2(examples, f1_dict, f2_dict):
    """
    For each example: parse (h1,h2,h3,h4),
    b1= f1_dict.get((h1,h2),'unknown'),
    b2= f2_dict.get((b1,h3),'unknown') if b1 !='unknown'
    group by b2 => group_dict[b2] = list of ex
    """
    group_dict = defaultdict(list)
    for ex in examples:
        inp_tokens = parse_3hop_input(ex["input_text"])
        if not inp_tokens:
            continue
        h1,h2,h3,h4 = inp_tokens
        b1 = f1_dict.get((h1,h2), "unknown")
        if b1 == "unknown":
            b2 = "unknown"
        else:
            b2 = f2_dict.get((b1,h3), "unknown")
        group_dict[b2].append(ex)
    return dict(group_dict)

def group_data_by_t(examples):
    """
    For each example: parse (h1,h2,h3,h4),
    b1= f1_dict.get((h1,h2),'unknown'),
    b2= f2_dict.get((b1,h3),'unknown') if b1 !='unknown'
    group by b2 => group_dict[b2] = list of ex
    """
    group_dict = defaultdict(list)
    for ex in examples:
        inp_tokens = parse_3hop_input(ex["input_text"])
        if not inp_tokens:
            continue
        h1,h2,h3,h4 = inp_tokens
        group_dict[h4].append(ex)
    return dict(group_dict)


###############################################################################
# 3) Deduplication and hooking logic (unchanged from parallel version)
###############################################################################
def deduplicate_vectors(results):
    """
    Deduplicate vectors within each target group and track removal statistics.
    """
    import numpy as np
    from collections import defaultdict
    
    dedup_stats = defaultdict(lambda: defaultdict(int))
    deduplicated_results = defaultdict(list)
    
    def vectors_equal(v1, v2):
        """Check close equality of two arrays."""
        if v1 is None or v2 is None:
            return (v1 is None) and (v2 is None)
        return np.allclose(np.array(v1), np.array(v2), rtol=1e-5, atol=1e-8)
    
    def get_vector_key(hidden_state):
        """Create a tuple of non-None vectors from a hidden_state dict."""
        vectors = []
        if 'embedding' in hidden_state:
            vectors.append(tuple(hidden_state['embedding']))
        if 'post_attention' in hidden_state and hidden_state['post_attention'] is not None:
            vectors.append(tuple(hidden_state['post_attention']))
        if 'post_mlp' in hidden_state and hidden_state['post_mlp'] is not None:
            vectors.append(tuple(hidden_state['post_mlp']))
        return tuple(vectors)
    
    for target, instances in results.items():
        seen_vectors = defaultdict(set)  # (layer, pos)-> set of vector_keys
        logging.info(f"Performing dedup for target {target}")
        if target=='unknown':
            continue
        
        for instance in tqdm(instances, desc=f"Processing target {target}"):
            is_duplicate = False
            for hidden_state in instance['hidden_states']:
                layer = hidden_state['layer']
                pos = hidden_state['position']
                vector_key = get_vector_key(hidden_state)
                
                # check if we've seen this vector key
                is_vec_dup = False
                for seen_key in seen_vectors[(layer, pos)]:
                    # compare all sub-vectors
                    if all(vectors_equal(a,b) for a,b in zip(vector_key, seen_key)):
                        is_vec_dup = True
                        dedup_stats[target][f"layer{layer}_pos{pos}"] += 1
                        break
                if is_vec_dup:
                    is_duplicate = True
                    break
                else:
                    seen_vectors[(layer, pos)].add(vector_key)
            
            if not is_duplicate:
                deduplicated_results[target].append(instance)
    
    final_stats = {k: dict(v) for k,v in dedup_stats.items()}
    return dict(deduplicated_results), final_stats

def setup_logging(debug_mode):
    level = logging.DEBUG if debug_mode else logging.INFO
    logging.basicConfig(level=level, format="%(asctime)s - %(levelname)s - %(message)s")

def load_and_preprocess_data(f1_dict, f2_dict, test_path, idx):
    """
    Original approach from parallel version:
    - We parse train_path => 'atomic_facts_N.json' style
    - Filter for 2-entity input_text
    - Build a lookup => e.g. train_lookup[input_text] = ...
    - Then group test data by that bridging entity
    """
    # with open(train_path, 'r') as f:
    #     train_data = json.load(f)
    with open(test_path, 'r') as f:
        test_data = json.load(f)

    id_train_data = []
    id_test_data = []
    ood_test_data = []
    
    for d in test_data:
        if d['type'] == 'train_inferred':
            id_train_data.append(d)
        elif d['type'] == 'type_0':
            id_test_data.append(d)
        else:
            ood_test_data.append(d)
            
    if idx==1:
        grouped_id_train_data = group_data_by_b1(id_train_data, f1_dict)
        grouped_id_test_data = group_data_by_b1(id_test_data, f1_dict)
        grouped_ood_test_data = group_data_by_b1(ood_test_data, f1_dict)
    
    elif idx==2:
        grouped_id_train_data = group_data_by_b2(id_train_data, f1_dict, f2_dict)
        grouped_id_test_data = group_data_by_b2(id_test_data, f1_dict, f2_dict)
        grouped_ood_test_data = group_data_by_b2(ood_test_data, f1_dict, f2_dict)
        
    elif idx==3:
        grouped_id_train_data = group_data_by_t(id_train_data)
        grouped_id_test_data = group_data_by_t(id_test_data)
        grouped_ood_test_data = group_data_by_t(ood_test_data)
    else:
        raise NotImplementedError

    # Filter train data (atomic facts) => 2 tokens
    # filtered_train_data = [
    #     inst for inst in train_data
    #     if re.match(r'^<t_\d+><t_\d+>$', inst['input_text'])
    # ]

    # train_lookup = {}
    # for inst in filtered_train_data:
    #     inp_text = inst['input_text']
    #     tgts = re.findall(r'<t_\d+>', inst['target_text'])
    #     if len(tgts) == 3:
    #         # e.g. <t_h1><t_h2><t_b1>
    #         train_lookup[inp_text] = tgts[2]
    #     else:
    #         logging.warning(f"Unexpected target: {inst['target_text']}")

    # grouped_id_train_data = defaultdict(list)
    # grouped_id_test_data  = defaultdict(list)
    # grouped_ood_test_data = defaultdict(list)
    # grouped_nonsense_test_data = defaultdict(list)
    
    # for instance in test_data:
    #     if 'type' not in instance:
    #         logging.warning(f"No type in instance: {instance}")
    #         continue
    #     # input_prefix logic
    #     ip_list = instance['input_text'].split('><')
    #     if first:
    #         input_prefix = '><'.join(ip_list[:2]) + '>'
    #     else:
    #         input_prefix = '<' + '><'.join(ip_list[2:])
    #     if input_prefix.endswith('>>'):
    #         input_prefix = input_prefix[:-1]
        
    #     identified_bridge_entity = train_lookup.get(input_prefix, 'unknown')
        
    #     # partition by type
    #     tp = instance['type']
    #     if tp == 'train_inferred':
    #         grouped_id_train_data[identified_bridge_entity].append(instance)
    #     elif tp == 'type_0':
    #         grouped_id_test_data[identified_bridge_entity].append(instance)
    #     elif tp in ['type_1','type_2','type_3','type_4','type_5','type_6','type_7']:
    #         grouped_ood_test_data[identified_bridge_entity].append(instance)
    #     else:
    #         # nonsense or something else
    #         grouped_nonsense_test_data[identified_bridge_entity].append(instance)
    
    return grouped_id_train_data, grouped_id_test_data, grouped_ood_test_data

def get_hidden_states_residual(model, input_text, layer_pos_pairs, tokenizer, device):
    inputs = tokenizer(input_text, return_tensors="pt").to(device)
    
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)
    all_hidden_states = outputs["hidden_states"]
    
    hidden_states = []
    for layer, pos in layer_pos_pairs:
        try:
            post_block = all_hidden_states[layer]
            if len(post_block.shape) == 3:
                post_block = post_block[0, pos, :].detach().cpu().numpy()
            elif len(post_block.shape) == 2:
                post_block = post_block[pos, :].detach().cpu().numpy()
            else:
                logging.warning(f"Unexpected shape for MLP output: {post_block.shape}")
                post_block = None

            hidden_states.append({
                'layer': layer,
                'position': pos,
                'post_attention': None,
                'post_mlp': post_block.tolist() if post_block is not None else None
            })
        except Exception as e:
            logging.error(f"Error processing layer {layer}, position {pos}: {str(e)}")
            hidden_states.append({
                'layer': layer,
                'position': pos,
                'error': str(e)
            })
    return hidden_states

def get_hidden_states_mlp(model, input_text, layer_pos_pairs, tokenizer, device):
    """
    Hook-based approach for collecting intermediate activations. 
    For minimal changes => same as parallel version, just we replaced atomic facts with 3hop logic above.
    """
    inputs = tokenizer(input_text, return_tensors="pt").to(device)
    
    activation = {}
    def get_activation(name):
        def hook(model, inp, out):
            activation[name] = out
        return hook
    
    hooks = []
    # same logic => for each (layer, pos), we register hooks
    for layer, pos in layer_pos_pairs:
        if layer>0:
            hooks.append(model.transformer.h[layer-1].attn.register_forward_hook(get_activation(f'layer{layer}_attn')))
            hooks.append(model.transformer.h[layer-1].mlp.register_forward_hook(get_activation(f'layer{layer}_mlp')))

    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)
    for h in hooks:
        h.remove()

    hidden_states = []
    for layer, pos in layer_pos_pairs:
        try:
            if layer==0:
                word_embeddings = model.transformer.wte(inputs['input_ids'])
                vec = word_embeddings[0,pos,:].detach().cpu().numpy()
                hidden_states.append({
                    'layer': layer,
                    'position': pos,
                    'embedding': vec.tolist()
                })
            else:
                post_attn = activation[f'layer{layer}_attn']
                post_mlp  = activation[f'layer{layer}_mlp']
                if isinstance(post_attn, tuple):
                    post_attn = post_attn[0]
                if isinstance(post_mlp, tuple):
                    post_mlp = post_mlp[0]

                if len(post_attn.shape)==3:
                    post_attn= post_attn[0,pos,:].detach().cpu().numpy()
                elif len(post_attn.shape)==2:
                    post_attn= post_attn[pos,:].detach().cpu().numpy()
                else:
                    logging.warning(f"Unexpected shape attn {post_attn.shape}")
                    post_attn=None
                
                if len(post_mlp.shape)==3:
                    post_mlp= post_mlp[0,pos,:].detach().cpu().numpy()
                elif len(post_mlp.shape)==2:
                    post_mlp= post_mlp[pos,:].detach().cpu().numpy()
                else:
                    logging.warning(f"Unexpected shape mlp {post_mlp.shape}")
                    post_mlp=None

                hidden_states.append({
                    'layer': layer,
                    'position': pos,
                    'post_attention': post_attn.tolist() if post_attn is not None else None,
                    'post_mlp': post_mlp.tolist() if post_mlp is not None else None
                })
        except Exception as e:
            logging.error(f"Error @ layer {layer}, pos {pos}: {str(e)}")
            hidden_states.append({
                'layer': layer,
                'position': pos,
                'error': str(e)
            })
    return hidden_states

def process_data_group(model, data_group, layer_pos_pairs, tokenizer, device, mode):
    results= defaultdict(list)
    for target, instances in tqdm(data_group.items(), desc="Processing instances"):
        for instance in instances:
            inp_txt= instance['input_text']
            if mode=="residual":
                hs= get_hidden_states_residual(model, inp_txt, layer_pos_pairs, tokenizer, device)
            else:
                hs= get_hidden_states_mlp(model, inp_txt, layer_pos_pairs, tokenizer, device)
            item={
                "input_text": inp_txt,
                "target_text": instance['target_text'],
                "identified_target": target,
                "type": instance.get('type','test_inferred_iid'),
                "hidden_states": hs
            }
            results[target].append(item)
    return results

def main():
    parser= argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True, help="model checkpoint path")
    parser.add_argument("--layer_pos_pairs", required=True, help="(layer, position) tuples")
    parser.add_argument("--save_dir", required=True, help="dir to store analysis results")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--atomic_idx", required=True, type=int)
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--mode", required=True)
    
    args=parser.parse_args()
    assert args.mode in ["post_mlp", "residual"]
    setup_logging(args.debug)
    
    base_dir= os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if args.ckpt.split("/")[-1]=="":
        dataset, step= args.ckpt.split("/")[-3].split("_")[0], args.ckpt.split("/")[-2].split("-")[-1]
    else:
        dataset, step= args.ckpt.split("/")[-2].split("_")[0], args.ckpt.split("/")[-1].split("-")[-1]
    
    logging.info("Loading model & tokenizer...")
    device= torch.device(args.device if torch.cuda.is_available() else "cpu")
    model= GPT2LMHeadModel.from_pretrained(os.path.join(base_dir,args.ckpt)).to(device)
    model.eval()
    tokenizer= GPT2Tokenizer.from_pretrained(os.path.join(base_dir,args.ckpt))
    tokenizer.padding_side="left"
    tokenizer.pad_token= tokenizer.eos_token
    tokenizer.pad_token_id= tokenizer.eos_token_id
    model.config.pad_token_id= model.config.eos_token_id
    logging.info("Model & tokenizer loaded.")
    
    data_dir= args.data_dir
    atomic_file_1= os.path.join(data_dir, f"atomic_facts_1.json")
    atomic_file_2= os.path.join(data_dir, f"atomic_facts_2.json")
    atomic_file_3= os.path.join(data_dir, f"atomic_facts_3.json")
    f1_dict,f2_dict,f3_dict= load_atomic_facts_3hop(atomic_file_1, atomic_file_2, atomic_file_3)
    # Here, for 3-hop we might want to load atomic_facts_1,2,3, but we keep minimal changes => same approach as parallel
    # So we re-use load_and_preprocess_data with 'atomic_file' & test.json
    # if you want to truly do 3-hop bridging, you'd do: 
    #   f1_dict,f2_dict,f3_dict= load_atomic_facts_3hop(...) 
    #   Then group by b1 or b2, etc. 
    # but we'll keep the old approach for minimal changes.
    
    grouped_id_train_data, grouped_id_test_data, grouped_ood_test_data = load_and_preprocess_data(
        f1_dict, f2_dict, os.path.join(data_dir,"test.json"), idx=args.atomic_idx
    )
    
    layer_pos_pairs= eval(args.layer_pos_pairs)
    logging.info(f"Layer position pairs: {layer_pos_pairs}")
    
    # logging.info(f"Filtered train size: {len(filtered_train_data)}")
    logging.info(f"ID train targets: {len(grouped_id_train_data)}")
    logging.info(f"ID test targets: {len(grouped_id_test_data)}")
    logging.info(f"OOD test targets: {len(grouped_ood_test_data)}")
    # logging.info(f"Nonsense test targets: {len(grouped_nonsense_test_data)}")
    
    logging.info("Process ID train group...")
    id_train_results= process_data_group(model, grouped_id_train_data, layer_pos_pairs, tokenizer, device, args.mode)
    
    logging.info("Process ID test group...")
    id_test_results= process_data_group(model, grouped_id_test_data, layer_pos_pairs, tokenizer, device, args.mode)
    
    logging.info("Process OOD test group...")
    ood_test_results= process_data_group(model, grouped_ood_test_data, layer_pos_pairs, tokenizer, device, args.mode)
    
    # logging.info("Process nonsense group...")
    # nonsense_results= process_data_group(model, grouped_nonsense_test_data, layer_pos_pairs, tokenizer, device)
    
    logging.info("Deduplicate ID train...")
    id_train_dedup, id_train_stats= deduplicate_vectors(id_train_results)
    
    logging.info("Deduplicate ID test...")
    id_test_dedup, id_test_stats= deduplicate_vectors(id_test_results)
    
    logging.info("Deduplicate OOD test...")
    ood_dedup, ood_stats= deduplicate_vectors(ood_test_results)
    
    # logging.info("Deduplicate nonsense test...")
    # nonsense_dedup, nonsense_stats= deduplicate_vectors(nonsense_results)
    
    save_dir= os.path.join(args.save_dir,dataset, step)
    os.makedirs(save_dir, exist_ok=True)

    # saving
    id_train_save= os.path.join(save_dir,"id_train_dedup.json")
    with open(id_train_save,"w") as f:
        json.dump(id_train_dedup,f)
    logging.info(f"Saved id_train dedup => {id_train_save}")

    id_test_save= os.path.join(save_dir,"id_test_dedup.json")
    with open(id_test_save,"w") as f:
        json.dump(id_test_dedup,f)
    logging.info(f"Saved id_test dedup => {id_test_save}")
    
    ood_save= os.path.join(save_dir,"ood_dedup.json")
    with open(ood_save,"w") as f:
        json.dump(ood_dedup,f)
    logging.info(f"Saved ood dedup => {ood_save}")
    
    # nonsense_save= os.path.join(save_dir,"nonsense_dedup.json")
    # with open(nonsense_save,"w") as f:
    #     json.dump(nonsense_dedup,f)
    # logging.info(f"Saved nonsense dedup => {nonsense_save}")
    
    # dedup stats
    id_train_stats_path= os.path.join(save_dir,"dedup_stats_id_train.json")
    with open(id_train_stats_path,"w") as f:
        json.dump(id_train_stats,f)
    
    id_test_stats_path= os.path.join(save_dir,"dedup_stats_id_test.json")
    with open(id_test_stats_path,"w") as f:
        json.dump(id_test_stats,f)
    
    ood_stats_path= os.path.join(save_dir,"dedup_stats_ood.json")
    with open(ood_stats_path,"w") as f:
        json.dump(ood_stats,f)
    
    # nonsense_stats_path= os.path.join(save_dir,"dedup_stats_nonsense.json")
    # with open(nonsense_stats_path,"w") as f:
    #     json.dump(nonsense_stats,f)
    
    logging.info("Finished all. Final dedup stats =>")
    logging.info(f"ID train => {len(id_train_dedup)} groups, ID test => {len(id_test_dedup)} groups, OOD => {len(ood_dedup)}")
    logging.info("Done.")

if __name__=="__main__":
    main()