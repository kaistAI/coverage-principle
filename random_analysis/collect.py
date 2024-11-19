import argparse
import torch
import json
from tqdm import tqdm
import os
from transformers import GPT2LMHeadModel, GPT2Tokenizer
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
    deduplicated_results = defaultdict(list)  # Changed to defaultdict

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
        if 'attn_output' in hidden_state and hidden_state['attn_output'] is not None:
            vectors.append(tuple(hidden_state['attn_output']))
        if 'mlp_output' in hidden_state and hidden_state['mlp_output'] is not None:
            vectors.append(tuple(hidden_state['mlp_output']))
        if 'residual' in hidden_state and hidden_state['residual'] is not None:
            vectors.append(tuple(hidden_state['residual']))
        return tuple(vectors)
    
    for group_name, instances in results.items():
        seen_vectors = defaultdict(set)  # (layer, position) -> set of vector tuples
        logging.info(f"Performing deduplication for group '{group_name}'")
        
        for instance in tqdm(instances, desc=f"Deduplicating {group_name}"):
            is_duplicate = False
            
            # Track duplicates for each hidden state
            for hidden_state in instance.get('hidden_states', []):
                layer = hidden_state.get('layer')
                pos = hidden_state.get('position')
                vector_key = get_vector_key(hidden_state)
                
                if layer is None or pos is None:
                    logging.warning(f"Missing layer or position in hidden_state: {hidden_state}")
                    continue
                
                # Check if we've seen this vector before
                is_vec_duplicate = False
                for seen_vec in seen_vectors[(layer, pos)]:
                    if all(vectors_equal(v1, v2) for v1, v2 in zip(vector_key, seen_vec)):
                        is_vec_duplicate = True
                        dedup_stats[group_name][f"layer{layer}_pos{pos}"] += 1
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

def setup_logging():
    logging.basicConfig(
        filename='parallel_collect_vectors.log',
        filemode='a',
        format='%(asctime)s - %(levelname)s - %(message)s',
        level=logging.INFO
    )
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    console.setFormatter(formatter)
    logging.getLogger('').addHandler(console)

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
    # Register hooks for transformer layers (now indexed from 1 to N)
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
                # Handle transformer layers (1 to N)
                attn_output = activation.get(f'layer{layer}_attn')
                mlp_output = activation.get(f'layer{layer}_mlp')
                
                logging.debug(f"Layer {layer} attention output type: {type(attn_output)}")
                logging.debug(f"Layer {layer} MLP output type: {type(mlp_output)}")
                
                if isinstance(attn_output, tuple):
                    attn_output = attn_output[0]
                if isinstance(mlp_output, tuple):
                    mlp_output = mlp_output[0]

                logging.debug(f"Layer {layer} attention output shape: {attn_output.shape if attn_output is not None else 'None'}")
                logging.debug(f"Layer {layer} MLP output shape: {mlp_output.shape if mlp_output is not None else 'None'}")

                # Extract vectors for the specified position
                attn_vector = None
                mlp_vector = None

                if attn_output is not None:
                    if len(attn_output.shape) == 3:
                        attn_vector = attn_output[0, pos, :].detach().cpu().numpy()
                    elif len(attn_output.shape) == 2:
                        attn_vector = attn_output[pos, :].detach().cpu().numpy()
                    else:
                        logging.warning(f"Unexpected shape for attention output: {attn_output.shape}")
                        attn_vector = None

                if mlp_output is not None:
                    if len(mlp_output.shape) == 3:
                        mlp_vector = mlp_output[0, pos, :].detach().cpu().numpy()
                    elif len(mlp_output.shape) == 2:
                        mlp_vector = mlp_output[pos, :].detach().cpu().numpy()
                    else:
                        logging.warning(f"Unexpected shape for MLP output: {mlp_output.shape}")
                        mlp_vector = None

                # Extract residual stream from hidden_states
                # hidden_states is a tuple where hidden_states[layer] corresponds to after layer `layer`
                residual = outputs.hidden_states[layer][0, pos, :].detach().cpu().numpy()

                hidden_state_dict = {
                    'layer': layer,
                    'position': pos,
                    'attn_output': attn_vector.tolist() if attn_vector is not None else None,
                    'mlp_output': mlp_vector.tolist() if mlp_vector is not None else None,
                    'residual': residual.tolist()
                }
                hidden_states.append(hidden_state_dict)
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
    
    setup_logging()
    
    logging.info("Loading model and tokenizer...")
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    try:
        model = GPT2LMHeadModel.from_pretrained(args.ckpt, use_safetensors=True).to(device)
    except OSError as e:
        logging.error(f"Failed to load model from {args.ckpt}: {e}")
        exit(1)
    model.eval()
    
    tokenizer = GPT2Tokenizer.from_pretrained(args.ckpt)
    tokenizer.padding_side = "left" 
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id
    model.config.pad_token_id = model.config.eos_token_id
    logging.info("Model and tokenizer loaded successfully")
    
    dataset = load_dataset(args.dataset, args.debug)
    layer_pos_pairs = eval(args.layer_pos_pairs)
    logging.info(f"Layer position pairs: {layer_pos_pairs}")
    
    results = defaultdict(list)  # Changed to defaultdict
    
    for idx, instance in enumerate(tqdm(dataset, desc="Processing dataset")):
        logging.debug(f"Processing instance {idx}")
        logging.debug(f"Instance type: {instance.get('type', 'No type specified')}")
        
        if 'type' not in instance or 'input_text' not in instance:
            logging.error(f"Instance {idx} missing 'type' or 'input_text': {instance}")
            continue
        
        # **New Condition:** Skip instances where 'type' includes 'atomic'
        instance_type = str(instance.get('type', '')).lower()
        if 'atomic' in instance_type:
            # logging.info(f"Skipping atomic instance {idx} with type '{instance.get('type')}'")
            continue
        
        # Proceed with processing for non-atomic instances
        group_type = instance['type']
        logging.debug(f"Input text: {instance.get('input_text', 'No input text')[:50]}...")
        
        input_text = instance["input_text"]
        hidden_states = get_hidden_states(model, input_text, layer_pos_pairs, tokenizer, device)
        
        result = {
            "input_text": input_text,
            "target_text": instance.get("target_text", "No target text"),
            "hidden_states": hidden_states
        }
        
        results[group_type].append(result)
        logging.debug(f"Added result for group '{group_type}'")
    
    # After processing all instances but before saving:
    deduplicated_results, dedup_stats = deduplicate_vectors(results)
    logging.warning(f"Deduplication statistics: {dedup_stats}")
    
    # Save deduplicated results
    os.makedirs(args.save_dir, exist_ok=True)
    save_path = os.path.join(args.save_dir, args.save_fname)
    
    # Save main results
    with open(save_path, 'w') as f:
        json.dump(deduplicated_results, f)
    
    # Save deduplication statistics
    # stats_save_path = os.path.join(args.save_dir, f"dedup_stats_{args.save_fname}")
    # with open(stats_save_path, 'w') as f:
    #     json.dump(dedup_stats, f)
    
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
    # logging.info(f"Deduplication statistics saved to {stats_save_path}")

if __name__ == "__main__":
    main()