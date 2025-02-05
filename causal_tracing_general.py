import numpy as np
import json
import os
import random
import torch
import torch.nn.functional as F
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from tqdm import tqdm
import argparse

def build_position_map(train_file, max_positions):
    """
    Build a position map from the training data.

    This function loads the training data from 'train_file' (a JSON file)
    and parses each "input_text" (assumed to be in the format 
    "<t_x><t_y><t_z>...") to build a mapping for each token position.

    For each position (0, 1, ..., max_positions-1), it collects all tokens
    that appear in that slot using a set (to avoid duplicates). Finally, it
    converts each set into a sorted list so that random.choice returns a 
    consistent order of tokens.

    Parameters:
      - train_file: path to the JSON file containing training data.
      - max_positions: the number of token positions to track.

    Returns:
      A list of lists, where each inner list contains the sorted unique tokens
      observed at that position.
    """
    pos2valid_tokens = [set() for _ in range(max_positions)]
    
    with open(train_file, "r") as f:
        train_data = json.load(f)
    
    for item in train_data:
        # Parse input_text: e.g. "<t_23><t_187><t_28><t_154>" -> ["t_23", "t_187", "t_28", "t_154"]
        tokens = item["input_text"].strip("<>").split("><")
        for i, tok in enumerate(tokens):
            if i < max_positions:
                pos2valid_tokens[i].add(tok)
    
    # Convert each set to a sorted list for reproducibility and easier sampling.
    pos2valid_tokens = [sorted(list(tokens_set)) for tokens_set in pos2valid_tokens]
    return pos2valid_tokens


def return_rank(hidden_states, word_emb, gold_token_ids, metric='dot'):
    """
    hidden_states: [B, seq_len, hidden_dim]
    word_emb: [vocab_size, hidden_dim]
    gold_token_ids: [B] (the ID of the gold token per example)
    metric: 'dot' or 'cos'

    Returns a list of ranks, one per example.
    """
    if metric == 'cos':
        # normalize
        word_emb = F.normalize(word_emb, p=2, dim=1)
        hidden_states = F.normalize(hidden_states, p=2, dim=2)
    # => shape [B, seq_len, vocab_size]
    logits = torch.matmul(hidden_states, word_emb.T)
    
    batch_size, seq_len, vocab_size = logits.shape
    
    # We'll assume we want the last position in seq_len => -1
    # If you want a custom position, adapt here.
    # shape => [B, vocab_size]
    final_logits = logits[:, -1, :]
    
    sorted_idx = final_logits.argsort(dim=-1, descending=True)
    ranks = []
    for i in range(batch_size):
        gold_id = gold_token_ids[i]
        rank_pos = (sorted_idx[i] == gold_id).nonzero(as_tuple=True)
        if len(rank_pos[0])>0:
            r = rank_pos[0].item()
        else:
            r = vocab_size  # if not found (shouldn't happen)
        ranks.append(r)
    return ranks

def parse_final_entity(target_text):
    """
    Example: given target_text = "<t_78><t_143><t_163><t_194><t_6></a>"
    parse out the last entity (like "t_6").
    """
    s = target_text.replace("</a>", "")
    tokens = s.strip("<>").split("><")  # => e.g. ["t_78","t_143","t_163","t_194","t_6"]
    return tokens[-1]  # "t_6"

def parse_input_tokens(input_text):
    """
    E.g. input_text = "<t_23><t_187><t_28><t_154>"
    => ["t_23","t_187","t_28","t_154"]
    Adjust if your data is 3 tokens or 5 tokens, etc.
    """
    tokens = input_text.strip("<>").split("><")
    return tokens

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", type=str, required=True)
    parser.add_argument("--data_file", type=str, required=True, 
                        help="Path to train.json or test.json, whichever you want to do tracing on.")
    parser.add_argument("--save_path", type=str, required=True)
    parser.add_argument("--target_layer", type=int, default=8)
    parser.add_argument("--num_layer", type=int, default=8)
    parser.add_argument("--step_list", nargs="+", default=None,
                        help="Which checkpoint steps to do perturbation on, e.g. 1000 2000")
    parser.add_argument("--type", type=str)
    parser.add_argument("--max_positions", type=int, default=4)
    args = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Build pos2valid_tokens from atomic_facts
    pos2valid_tokens = build_position_map(args.data_file, args.max_positions)
    
    # Load data
    with open(args.data_file) as f:
        dataset = json.load(f)  # train or test data
        if args.type:
            dataset = [d for d in dataset if d["type"]==args.type]
        print(f"example: {dataset[0]}")
    print("Loaded data size:", len(dataset))
    
    # Find all checkpoints
    all_ckpts = [ck for ck in os.listdir(args.model_dir) if ck.startswith("checkpoint-")]
    # sort by step
    def step_number(ck):
        return int(ck.split("-")[1])
    all_ckpts.sort(key=step_number)
    
    step_list = [f"checkpoint-{s}" for s in args.step_list] if args.step_list else []
    
    BATCH_SIZE = 128
    
    results = {}
    
    for ck in tqdm(all_ckpts):
        if ck not in step_list:
            continue
        ckpt_path = os.path.join(args.model_dir, ck)
        print(f"\nNow checkpoint = {ck}")
        
        model = GPT2LMHeadModel.from_pretrained(ckpt_path).to(device)
        tokenizer = GPT2Tokenizer.from_pretrained(ckpt_path)
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
        model.config.pad_token_id = model.config.eos_token_id
        
        word_embedding = model.lm_head.weight.data  # shape [vocab, hidden_dim]
        
        all_out = []
        
        # We'll do a forward pass for each batch
        for start_i in range(0, len(dataset), BATCH_SIZE):
            batch = dataset[start_i:start_i+BATCH_SIZE]
            if len(batch) == 0:
                break
            inp_texts = [b["input_text"] for b in batch]
            
            # parse final gold token
            gold_tokens = []
            for b_item in batch:
                last_ent = parse_final_entity(b_item["target_text"])  # e.g. "t_6"
                gold_tokens.append(f"<{last_ent}>")
            gold_ids_list = []
            for g in gold_tokens:
                tok = tokenizer.encode(g, add_special_tokens=False)
                gold_ids_list.append(tok[0] if len(tok)>0 else tokenizer.unk_token_id)
            
            # standard forward
            tok_out = tokenizer(inp_texts, return_tensors="pt", padding=True)
            input_ids_ = tok_out["input_ids"].to(device)
            attention_mask_ = tok_out["attention_mask"].to(device)
            
            with torch.no_grad():
                outputs_ = model(input_ids=input_ids_, attention_mask=attention_mask_, output_hidden_states=True)
            hidden_states_all = outputs_.hidden_states  # tuple of length num_layer+1
            # rank before
            rank_before = return_rank(hidden_states_all[args.target_layer], word_embedding, gold_ids_list)
            
            # store partial
            partial_dict_list = []
            for i in range(len(batch)):
                partial_dict_list.append({"rank_before": rank_before[i]})
            
            # check if we do perturbation
            if ck in step_list:
                # parse each input's tokens
                parsed_tokens_list = [parse_input_tokens(x) for x in inp_texts]
                # for each position in 0..3
                for pos in range(args.max_positions):
                    # build CF inputs
                    # we pick a random token from pos2valid_tokens[pos] for each example
                    cf_inp_list = []
                    for i, ptoks in enumerate(parsed_tokens_list):
                        new_tok_list = ptoks[:]
                        while new_tok_list == ptoks:
                            alt_tok = random.choice(pos2valid_tokens[pos])  # something seen in training
                            new_tok_list[pos] = alt_tok  # replace the position
                        cf_inp_list.append("<" + "><".join(new_tok_list) + ">")
                        
                    
                    # forward pass on CF
                    tok_cf = tokenizer(cf_inp_list, return_tensors="pt", padding=True)
                    cf_inp_ids = tok_cf["input_ids"].to(device)
                    cf_attn = tok_cf["attention_mask"].to(device)
                    with torch.no_grad():
                        cf_outputs = model(cf_inp_ids, attention_mask=cf_attn, output_hidden_states=True)
                    cf_hstates_all = cf_outputs.hidden_states
                    
                    # now layerwise intervention from layer=1..target_layer
                    for layer_to_intervene in range(1, args.target_layer):
                        # original hidden states at that layer
                        orig_layer_h = hidden_states_all[layer_to_intervene].clone()
                        cf_layer_h    = cf_hstates_all[layer_to_intervene]
                        
                        # intervene on position=pos
                        orig_layer_h[:, pos, :] = cf_layer_h[:, pos, :]
                        
                        # forward the remainder
                        with torch.no_grad():
                            temp_hidden = orig_layer_h
                            for l in range(layer_to_intervene, args.target_layer):
                                f_layer = model.transformer.h[l]
                                # attn
                                residual = temp_hidden
                                temp_hidden = f_layer.ln_1(temp_hidden)
                                attn_out = f_layer.attn(temp_hidden)[0]
                                temp_hidden = attn_out + residual
                                # mlp
                                residual = temp_hidden
                                temp_hidden = f_layer.ln_2(temp_hidden)
                                ff = f_layer.mlp.c_proj(f_layer.mlp.act(f_layer.mlp.c_fc(temp_hidden)))
                                temp_hidden = residual + ff
                            temp_hidden = model.transformer.ln_f(temp_hidden)
                        
                        # measure rank
                        rank_after = return_rank(temp_hidden, word_embedding, gold_ids_list)
                        
                        # store
                        for i in range(len(batch)):
                            partial_dict_list[i][f"pos{pos}_layer{layer_to_intervene}"] = rank_after[i]
            
            # finalize store
            all_out.extend(partial_dict_list)
        
        results[ck] = all_out
    
    # save
    out_file_name = f"{os.path.basename(args.model_dir)}_causaltracing.json"
    with open(os.path.join(args.save_path, out_file_name), "w") as f:
        json.dump(results, f, indent=2)
        
    print(f"result saved to {out_file_name}")

if __name__=="__main__":
    main()