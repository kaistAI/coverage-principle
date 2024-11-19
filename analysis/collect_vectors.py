import argparse
import torch
import json
from tqdm import tqdm
import os
from transformers import GPT2Config, GPT2LMHeadModel, GPT2Tokenizer
import logging
import numpy as np
from collections import defaultdict
import random

def deduplicate_vectors(results):
    """
    Deduplicate vectors within each group and track removal statistics.
    
    Args:
        results (dict): Dictionary containing groups of results
        
    Returns:
        tuple: (deduplicated_results, dedup_stats)
    """
    dedup_stats = defaultdict(lambda: defaultdict(int))
    deduplicated_results = {
        "train_inferred": [],
        "test_inferred_iid": [],
        "test_inferred_ood": []
    }
    
    def vectors_equal(v1, v2):
        """Compare two vectors for equality with numerical tolerance."""
        if v1 is None or v2 is None:
            return v1 is None and v2 is None
        return np.allclose(np.array(v1), np.array(v2), rtol=1e-5, atol=1e-8)
    
    def get_vector_key(hidden_state):
        """Create a tuple of vectors from a hidden state."""
        vectors = []
        if 'embedding' in hidden_state:
            vectors.append(tuple(hidden_state['embedding']))
        if 'post_attention' in hidden_state and hidden_state['post_attention'] is not None:
            vectors.append(tuple(hidden_state['post_attention']))
        if 'post_mlp' in hidden_state and hidden_state['post_mlp'] is not None:
            vectors.append(tuple(hidden_state['post_mlp']))
        return tuple(vectors)
    
    for group_name, instances in results.items():
        seen_vectors = defaultdict(set)  # (layer, position) -> set of vector tuples
        logging.info(f"performing dedup for group {group_name}")
        
        for instance in tqdm(instances):
            is_duplicate = False
            
            # Track duplicates for each hidden state
            for hidden_state in instance['hidden_states']:
                layer = hidden_state['layer']
                pos = hidden_state['position']
                vector_key = get_vector_key(hidden_state)
                
                # Check if we've seen this vector before
                is_vec_duplicate = False
                for seen_vec in seen_vectors[(layer, pos)]:
                    if all(vectors_equal(v1, v2) for v1, v2 in zip(vector_key, seen_vec)):
                        is_vec_duplicate = True
                        dedup_stats[group_name][f"layer{layer}_pos{pos}"] += 1  # Convert tuple to string key
                        break
                
                if is_vec_duplicate:
                    is_duplicate = True
                    break
                else:
                    seen_vectors[(layer, pos)].add(vector_key)
            
            # If instance is not a duplicate, add it to deduplicated results
            if not is_duplicate:
                deduplicated_results[group_name].append(instance)
    
    # Convert defaultdict to regular dict with string keys
    final_stats = {}
    for group_name, stats in dedup_stats.items():
        final_stats[group_name] = dict(stats)  # Convert inner defaultdict to regular dict
    
    return deduplicated_results, final_stats

def setup_logging(debug_mode):
    level = logging.DEBUG if debug_mode else logging.ERROR
    logging.basicConfig(level=level, format='%(asctime)s - %(levelname)s - %(message)s')

def load_dataset(dataset_path, debug):
    with open(dataset_path, 'r') as f:
        dataset = json.load(f)
        if debug:
            dataset = random.sample(dataset, 200)
    logging.debug(f"Loaded {len(dataset)} instances from {dataset_path}")
    return dataset

def get_hidden_states(model, input_text, layer_pos_pairs, tokenizer, device):
    inputs = tokenizer(input_text, return_tensors="pt").to(device)
    
    activation = {}
    def get_activation(name):
        def hook(model, input, output):
            activation[name] = output
        return hook

    hooks = []
    # Register hooks for transformer layers (now indexed from 1 to 8)
    for layer, pos in layer_pos_pairs:
        if layer > 0:  # Skip layer 0 as it's handled separately
            hooks.append(model.transformer.h[layer-1].attn.register_forward_hook(
                get_activation(f'layer{layer}_attn')))
            hooks.append(model.transformer.h[layer-1].mlp.register_forward_hook(
                get_activation(f'layer{layer}_mlp')))

    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)

    for hook in hooks:
        hook.remove()

    hidden_states = []
    for layer, pos in layer_pos_pairs:
        try:
            if layer == 0:
                # Handle word embeddings (layer 0)
                word_embeddings = model.transformer.wte(inputs['input_ids'])
                hidden_state = word_embeddings[0, pos, :].detach().cpu().numpy()
                hidden_states.append({
                    'layer': layer,
                    'position': pos,
                    'embedding': hidden_state.tolist()
                })
            else:
                # Handle transformer layers (1 to 8)
                post_attention = activation[f'layer{layer}_attn']
                post_mlp = activation[f'layer{layer}_mlp']
                
                logging.debug(f"Layer {layer} attention output type: {type(post_attention)}")
                logging.debug(f"Layer {layer} MLP output type: {type(post_mlp)}")
                
                if isinstance(post_attention, tuple):
                    post_attention = post_attention[0]
                if isinstance(post_mlp, tuple):
                    post_mlp = post_mlp[0]

                logging.debug(f"Layer {layer} attention output shape: {post_attention.shape}")
                logging.debug(f"Layer {layer} MLP output shape: {post_mlp.shape}")

                if len(post_attention.shape) == 3:
                    post_attention = post_attention[0, pos, :].detach().cpu().numpy()
                elif len(post_attention.shape) == 2:
                    post_attention = post_attention[pos, :].detach().cpu().numpy()
                else:
                    logging.warning(f"Unexpected shape for attention output: {post_attention.shape}")
                    post_attention = None

                if len(post_mlp.shape) == 3:
                    post_mlp = post_mlp[0, pos, :].detach().cpu().numpy()
                elif len(post_mlp.shape) == 2:
                    post_mlp = post_mlp[pos, :].detach().cpu().numpy()
                else:
                    logging.warning(f"Unexpected shape for MLP output: {post_mlp.shape}")
                    post_mlp = None

                hidden_states.append({
                    'layer': layer,
                    'position': pos,
                    'post_attention': post_attention.tolist() if post_attention is not None else None,
                    'post_mlp': post_mlp.tolist() if post_mlp is not None else None
                })
        except Exception as e:
            logging.error(f"Error processing layer {layer}, position {pos}: {str(e)}")
            hidden_states.append({
                'layer': layer,
                'position': pos,
                'error': str(e)
            })

    return hidden_states

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True, help="Path to the model checkpoint")
    parser.add_argument("--dataset", required=True, help="Path to the evaluation dataset")
    parser.add_argument("--layer_pos_pairs", required=True, help="List of (layer, position) tuples to evaluate")
    parser.add_argument("--save_dir", required=True, help="Directory to save the analysis results")
    parser.add_argument("--save_fname", required=True, help="Filename to save the analysis results")
    parser.add_argument("--device", default="cuda:0", help="Device to run the model on")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode for verbose output")
    
    args = parser.parse_args()
    
    setup_logging(args.debug)
    
    logging.debug("Loading model and tokenizer...")
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model = GPT2LMHeadModel.from_pretrained(args.ckpt).to(device)
    model.eval()
    
    tokenizer = GPT2Tokenizer.from_pretrained(args.ckpt)
    tokenizer.padding_side = "left" 
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id
    model.config.pad_token_id = model.config.eos_token_id
    logging.debug("Model and tokenizer loaded successfully")
    
    dataset = load_dataset(args.dataset, args.debug)
    layer_pos_pairs = eval(args.layer_pos_pairs)
    logging.debug(f"Layer position pairs: {layer_pos_pairs}")
    
    results = {
        "train_inferred": [],
        "test_inferred_iid": [],
        "test_inferred_ood": []
    }
    
    skipped_count = 0
    for idx, instance in enumerate(tqdm(dataset)):
        logging.debug(f"Processing instance {idx}")
        logging.debug(f"Instance type: {instance.get('type', 'No type specified')}")
        
        if 'type' not in instance or 'input_text' not in instance:
            logging.error(f"Instance {idx} missing 'type' or 'input_text': {instance}")
            continue
        
        if instance['type'] not in results.keys():
            skipped_count += 1
            logging.debug(f"Skipping id_atomic instance {idx}")
            continue
        
        logging.debug(f"Input text: {instance.get('input_text', 'No input text')[:50]}...")
        
        input_text = instance["input_text"]
        hidden_states = get_hidden_states(model, input_text, layer_pos_pairs, tokenizer, device)
        
        result = {
            "input_text": input_text,
            "target_text": instance.get("target_text", "No target text"),
            "hidden_states": hidden_states
        }
        
        if instance["type"] in results:
            results[instance["type"]].append(result)
            logging.debug(f"Added result for {instance['type']}")
        else:
            logging.error(f"Unknown instance type: {instance['type']}")

    # After processing all instances but before saving:
    deduplicated_results, dedup_stats = deduplicate_vectors(results)
    logging.warning(f"dedup_stats: {dedup_stats}")
    
    # Save deduplicated results
    os.makedirs(args.save_dir, exist_ok=True)
    save_path = os.path.join(args.save_dir, args.save_fname)
    
    # Save main results
    with open(save_path, 'w') as f:
        json.dump(deduplicated_results, f)
    
    # Save deduplication statistics
    stats_save_path = os.path.join(args.save_dir, f"dedup_stats_{args.save_fname}")
    with open(stats_save_path, 'w') as f:
        json.dump(dedup_stats, f)
    
    # Log deduplication statistics
    logging.info(f"Deduplication statistics:")
    for group_name, stats in dedup_stats.items():
        logging.info(f"\nGroup: {group_name}")
        for key, count in sorted(stats.items()):
            layer, pos = key.replace('layer', '').split('_pos')
            logging.info(f"  Layer {layer}, Position {pos}: Removed {count} duplicates")
    
    logging.info(f"\nFinal counts after deduplication:")
    for key, value in deduplicated_results.items():
        logging.info(f"  {key}: {len(value)}")
    
    logging.info(f"\nResults saved to {save_path}")
    logging.info(f"Deduplication statistics saved to {stats_save_path}")

if __name__ == "__main__":
    main()
