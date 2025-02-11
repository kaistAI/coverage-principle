import argparse
import torch
import json
from tqdm import tqdm
import os
from transformers import GPT2Config, GPT2LMHeadModel, GPT2Tokenizer
import logging
from collections import defaultdict
import re

def deduplicate_vectors(results):
    """
    Deduplicate vectors within each target group and track removal statistics.
    
    Args:
        results (dict): Dictionary containing results grouped by targets
        
    Returns:
        tuple: (deduplicated_results, dedup_stats)
    """
    import numpy as np
    from collections import defaultdict
    
    dedup_stats = defaultdict(lambda: defaultdict(int))
    deduplicated_results = defaultdict(list)
    
    # ? 이거 믿을만한 threshold 인가?
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
    
    for target, instances in results.items():
        seen_vectors = defaultdict(set)  # (layer, position) -> set of vector tuples
        logging.info(f"Performing deduplication for target {target}")
        
        for instance in tqdm(instances, desc=f"Processing target {target}"):
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
                        dedup_stats[target][f"layer{layer}_pos{pos}"] += 1
                        break
                
                if is_vec_duplicate:
                    is_duplicate = True
                    break
                else:
                    seen_vectors[(layer, pos)].add(vector_key)
            
            # If instance is not a duplicate, add it to deduplicated results
            if not is_duplicate:
                deduplicated_results[target].append(instance)
    
    # Convert defaultdict to regular dict with string keys
    final_stats = {}
    for target, stats in dedup_stats.items():
        final_stats[target] = dict(stats)
    
    return dict(deduplicated_results), final_stats

def setup_logging(debug_mode):
    level = logging.DEBUG if debug_mode else logging.INFO
    logging.basicConfig(level=level, format='%(asctime)s - %(levelname)s - %(message)s')

def load_and_preprocess_data(train_path, test_path, first):
    with open(train_path, 'r') as f:
        train_data = json.load(f)
    
    with open(test_path, 'r') as f:
        test_data = json.load(f)

    # Filter train data to include only instances with two entities in input_text (atomic fact)
    filtered_train_data = [
        instance for instance in train_data
        if re.match(r'^<t_\d+><t_\d+>$', instance['input_text'])
    ]
    
    # Create a lookup dictionary for filtered train data
    train_lookup = {}
    for instance in filtered_train_data:
        input_text = instance['input_text']
        target_entities = re.findall(r'<t_\d+>', instance['target_text'])
        if len(target_entities) == 3:
            train_lookup[input_text] = target_entities[2]
        else:
            logging.warning(f"Unexpected target format: {instance['target_text']}")
    
    # Group test data based on identified targets
    grouped_id_train_data = defaultdict(list)
    grouped_id_test_data = defaultdict(list)
    grouped_ood_test_data = defaultdict(list)
    grouped_nonsense_test_data = defaultdict(list)
    
    for instance in test_data:
        # Extract <e_N1><r_r1> from test data
        input_prefix = '><'.join(instance['input_text'].split('><')[:2]) + '>' if first else '<' + '><'.join(instance['input_text'].split('><')[2:])
        if input_prefix.endswith('>>'):
            input_prefix=input_prefix[:-1]
        identified_bridge_entity = train_lookup.get(input_prefix, 'unknown')
        # Sanity Check 'unknown' is only for test_nonsenses
        if identified_bridge_entity == 'unknown':
            assert not re.match(r'^<t_\d+><t_\d+>$', input_prefix)
        else:
            assert re.match(r'^<t_\d+><t_\d+>$', input_prefix)
        
        if instance.get('type') == None:
            logging.warning(f"Unexpected data format - There is no type: {instance}")
        elif instance['type'] == 'train_inferred':
            grouped_id_train_data[identified_bridge_entity].append(instance)
        elif instance['type'] == 'type_0':
            grouped_id_test_data[identified_bridge_entity].append(instance)
        elif instance['type'] in ['type_1', 'type_2', 'type_3', 'type_4', 'type_5', 'type_6', 'type_7']:
            grouped_ood_test_data[identified_bridge_entity].append(instance)
    
    return filtered_train_data, grouped_id_train_data, grouped_id_test_data, grouped_ood_test_data, grouped_nonsense_test_data

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


def process_data_group(model, data_group, layer_pos_pairs, tokenizer, device):
    results = defaultdict(list)
    for target, instances in tqdm(data_group.items(), desc="Processing instances"):
        for instance in instances:
            logging.debug(f"Processing instance of type: {instance.get('type', 'test_inferred_iid')}")
            logging.debug(f"Input text: {instance['input_text'][:50]}...")
            
            hidden_states = get_hidden_states(model, instance['input_text'], layer_pos_pairs, tokenizer, device)
            
            result = {
                "input_text": instance['input_text'],
                "target_text": instance['target_text'],
                "identified_target": target,
                "type": instance.get('type', 'test_inferred_iid'),
                "hidden_states": hidden_states
            }
            
            results[target].append(result)
            logging.debug(f"Added result for target: {target}")
    return results

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True, help="Path to the model checkpoint")
    parser.add_argument("--layer_pos_pairs", required=True, help="List of (layer, position) tuples to evaluate")
    parser.add_argument("--save_dir", required=True, help="Directory to save the analysis results")
    parser.add_argument("--device", default="cuda:0", help="Device to run the model on")
    parser.add_argument("--merge_id_data", action="store_true", help="Combine train_inferred and test_inferred_id in grouping stage")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode for verbose output")
    parser.add_argument("--atomic_idx", required=True, type=int)
    parser.add_argument("--data_dir", required=True, type=str)
    
    args = parser.parse_args()
    
    setup_logging(args.debug)
    
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    if args.ckpt.split("/")[-1] == "":
        dataset, step = args.ckpt.split("/")[-3].split("_")[0], args.ckpt.split("/")[-2].split("-")[-1]
    else:
        dataset, step = args.ckpt.split("/")[-2].split("_")[0], args.ckpt.split("/")[-1].split("-")[-1]
    
    logging.info("Loading model and tokenizer...")
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model = GPT2LMHeadModel.from_pretrained(os.path.join(base_dir, args.ckpt)).to(device)
    model.eval()
    tokenizer = GPT2Tokenizer.from_pretrained(os.path.join(base_dir, args.ckpt))
    tokenizer.padding_side = "left" 
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id
    model.config.pad_token_id = model.config.eos_token_id
    logging.info("Model and tokenizer loaded successfully")
    
    data_dir = args.data_dir
    atomic_dir = os.path.join(data_dir, f"atomic_facts_{args.atomic_idx}.json")
    filtered_train_data, grouped_id_train_data, grouped_id_test_data, grouped_ood_test_data, grouped_nonsense_test_data = load_and_preprocess_data(atomic_dir, os.path.join(data_dir, "test.json"), first=args.atomic_idx==1)
    
    layer_pos_pairs = eval(args.layer_pos_pairs)
    logging.info(f"Layer position pairs: {layer_pos_pairs}")
    
    logging.info(f"Number of filtered train instances: {len(filtered_train_data)}")
    logging.info(f"Number of unique ID train targets: {len(grouped_id_train_data)}")
    logging.info(f"Number of unique ID test targets: {len(grouped_id_test_data)}")
    logging.info(f"Number of unique OOD test targets: {len(grouped_ood_test_data)}")
    logging.info(f"Number of unique nonsense test targets: {len(grouped_nonsense_test_data)}")
    
    logging.info("Processing in-distribution train data...")
    id_train_results = process_data_group(model, grouped_id_train_data, layer_pos_pairs, tokenizer, device)
    
    logging.info("Processing in-distribution test data...")
    id_test_results = process_data_group(model, grouped_id_test_data, layer_pos_pairs, tokenizer, device)
    
    logging.info("Processing out-of-distribution test data...")
    ood_test_results = process_data_group(model, grouped_ood_test_data, layer_pos_pairs, tokenizer, device)
    
    logging.info("Processing nonsense test data...")
    nonsense_results = process_data_group(model, grouped_nonsense_test_data, layer_pos_pairs, tokenizer, device)
    
    # Perform deduplication
    logging.info("Performing deduplication for in-distribution train data...")
    id_train_results_dedup, id_train_dedup_stats = deduplicate_vectors(id_train_results)
    
    logging.info("Performing deduplication for in-distribution test data...")
    id_test_results_dedup, id_test_dedup_stats = deduplicate_vectors(id_test_results)
    
    logging.info("Performing deduplication for out-of-distribution data...")
    ood_results_dedup, ood_dedup_stats = deduplicate_vectors(ood_test_results)
    
    logging.info("Performing deduplication for nonsense data...")
    nonsense_results_dedup, nonsense_dedup_stats = deduplicate_vectors(nonsense_results)
    
    # ToDo : If the code is changed to support multiple layer_pos_pairs, should change save_dir for each layer_pos_pair
    save_dir = os.path.join(args.save_dir, dataset, step)
    os.makedirs(save_dir, exist_ok=True)
    
    # Save deduplicated results
    id_train_save_path = os.path.join(save_dir, "id_train_dedup.json")
    with open(id_train_save_path, 'w') as f:
        json.dump(id_train_results_dedup, f)
    logging.info(f"Deduplicated in-distribution analysis results saved to {id_train_save_path}")
    
    id_test_save_path = os.path.join(save_dir, "id_test_dedup.json")
    with open(id_test_save_path, 'w') as f:
        json.dump(id_test_results_dedup, f)
    logging.info(f"Deduplicated in-distribution analysis results saved to {id_test_save_path}")
    
    ood_save_path = os.path.join(save_dir, "ood_dedup.json")
    with open(ood_save_path, 'w') as f:
        json.dump(ood_results_dedup, f)
    logging.info(f"Deduplicated out-of-distribution analysis results saved to {ood_save_path}")
    
    nonsense_save_path = os.path.join(save_dir, "nonsense_dedup.json")
    with open(nonsense_save_path, 'w') as f:
        json.dump(nonsense_results_dedup, f)
    logging.info(f"Deduplicated nonsense analysis results saved to {nonsense_save_path}")
    
    # Save deduplication statistics
    id_train_stats_save_path = os.path.join(save_dir, f"dedup_stats_id_train_dedup.json")
    with open(id_train_stats_save_path, 'w') as f:
        json.dump(id_train_dedup_stats, f)
        
    id_test_stats_save_path = os.path.join(save_dir, f"dedup_stats_id_test_dedup.json")
    with open(id_test_stats_save_path, 'w') as f:
        json.dump(id_test_dedup_stats, f)
    
    ood_stats_save_path = os.path.join(save_dir, f"dedup_stats_ood_dedup.json")
    with open(ood_stats_save_path, 'w') as f:
        json.dump(ood_dedup_stats, f)
        
    nonsense_stats_save_path = os.path.join(save_dir, f"dedup_stats_nonsense_dedup.json")
    with open(nonsense_stats_save_path, 'w') as f:
        json.dump(nonsense_dedup_stats, f)
    
    # Log deduplication statistics and final counts
    logging.info("\nDeduplication statistics:")
    logging.info("\nIn-distribution Train:")
    for target, stats in id_train_dedup_stats.items():
        logging.info(f"\nTarget: {target}")
        for key, count in sorted(stats.items()):
            layer, pos = key.replace('layer', '').split('_pos')
            logging.info(f"  Layer {layer}, Position {pos}: Removed {count} duplicates")
            
    logging.info("\nIn-distribution Test:")
    for target, stats in id_test_dedup_stats.items():
        logging.info(f"\nTarget: {target}")
        for key, count in sorted(stats.items()):
            layer, pos = key.replace('layer', '').split('_pos')
            logging.info(f"  Layer {layer}, Position {pos}: Removed {count} duplicates")
    
    logging.info("\nOut-of-distribution:")
    for target, stats in ood_dedup_stats.items():
        logging.info(f"\nTarget: {target}")
        for key, count in sorted(stats.items()):
            layer, pos = key.replace('layer', '').split('_pos')
            logging.info(f"  Layer {layer}, Position {pos}: Removed {count} duplicates")
            
    logging.info("\nNonsense:")
    for target, stats in nonsense_dedup_stats.items():
        logging.info(f"\nTarget: {target}")
        for key, count in sorted(stats.items()):
            layer, pos = key.replace('layer', '').split('_pos')
            logging.info(f"  Layer {layer}, Position {pos}: Removed {count} duplicates")
    
    logging.info("\nFinal counts after deduplication:")
    logging.info("In-distribution Train:")
    for target, instances in id_train_results_dedup.items():
        logging.info(f"  {target}: {len(instances)}")
    logging.info("In-distribution Test:")
    for target, instances in id_test_results_dedup.items():
        logging.info(f"  {target}: {len(instances)}")
    logging.info("Out-of-distribution:")
    for target, instances in ood_results_dedup.items():
        logging.info(f"  {target}: {len(instances)}")
    logging.info("Nonsense:")
    for target, instances in nonsense_results_dedup.items():
        logging.info(f"  {target}: {len(instances)}")    
        
if __name__ == "__main__":
    main()