import argparse
import torch
import json
from tqdm import tqdm
import os
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import logging
from collections import defaultdict
import re


def setup_logging(debug_mode):
    level = logging.DEBUG if debug_mode else logging.INFO
    logging.basicConfig(level=level, format='%(asctime)s - %(levelname)s - %(message)s')


def load_and_preprocess_data(train_path, test_path, merge_id_data=False):
    with open(train_path, 'r') as f:
        train_data = json.load(f)

    with open(test_path, 'r') as f:
        test_data = json.load(f)

    # Filter train data to include only atomic fact
    filtered_train_data = [
        instance for instance in train_data
        if re.match(r'^<e_\d+><r_\d+>$', instance['input_text'])
    ]

    # Create a lookup dictionary for filtered train data
    train_lookup = {}   # <h><r1> -> <t>
    for instance in filtered_train_data:
        input_text = instance['input_text']
        target_entities = re.findall(r'<e_\d+>', instance['target_text'])
        if len(target_entities) == 2:
            train_lookup[input_text] = target_entities[1]
        else:
            logging.warning(f"Unexpected target format: {instance['target_text']}")

    # Group test data based on bridge entity
    # <b> -> [{}, {}, ...]
    grouped_id_train_data = defaultdict(list)
    grouped_id_test_data = defaultdict(list)
    grouped_ood_test_data = defaultdict(list)
    grouped_nonsense_test_data = defaultdict(list)

    for instance in test_data:
        # Extract <h><r1> from test data
        input_prefix = '><'.join(instance['input_text'].split('><')[:2]) + '>'
        if input_prefix.endswith('>>'):
            input_prefix = input_prefix[:-1]
        identified_bridge_entity = train_lookup.get(input_prefix, 'unknown')
        # Sanity Check 'unknown' is only for test_nonsenses
        if identified_bridge_entity == 'unknown':
            assert not re.match(r'^<e_\d+><r_\d+>$', input_prefix)
        else:
            assert re.match(r'^<e_\d+><r_\d+>$', input_prefix)

        if instance.get('type') == None:
            logging.warning(f"Unexpected data format - There is no type: {instance}")
        elif instance['type'] == 'train_inferred':
            grouped_id_train_data[identified_bridge_entity].append(instance)
        elif instance['type'] == 'test_inferred_id':
            grouped_id_test_data[identified_bridge_entity].append(instance)
        elif instance['type'] == 'test_inferred_ood':
            grouped_ood_test_data[identified_bridge_entity].append(instance)
        elif instance['type'] == 'test_nonsenses':
            # ?. 얘는 bridge entity로 뷴류하는 것 없이 그냥 다 저장하는 게 맞나?
            grouped_nonsense_test_data[instance['target_text']].append(instance)

    if merge_id_data:
        merged_id_data = defaultdict(list)
        for key in set(grouped_id_train_data.keys()).union(grouped_id_test_data.keys()):
            merged_id_data[key].extend(grouped_id_train_data.get(key, []))
            merged_id_data[key].extend(grouped_id_test_data.get(key, []))
        return filtered_train_data, merged_id_data, grouped_ood_test_data, grouped_nonsense_test_data
    else:
        return filtered_train_data, (grouped_id_train_data, grouped_id_test_data), grouped_ood_test_data, grouped_nonsense_test_data


def get_hidden_states(model, input_text, layer_pos_pairs, tokenizer, device):
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
                logging.warning(f"Unexpected shape for residual stream output: {post_block.shape}")
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

def process_data_group(model, data_group, layer_pos_pairs, tokenizer, device):
    results = defaultdict(list)
    for bridge_entity, instances in tqdm(data_group.items(), desc="Processing instances"):
        for instance in instances:
            logging.debug(f"Processing instance of type: {instance.get('type')}")
            logging.debug(f"Input text: {instance['input_text'][:50]}...")
            
            hidden_states = get_hidden_states(model, instance['input_text'], layer_pos_pairs, tokenizer, device)
            
            result = {
                "input_text": instance['input_text'],
                "target_text": instance['target_text'],
                "identified_target": bridge_entity,
                "type": instance.get('type'),
                "hidden_states": hidden_states
            }
            
            results[bridge_entity].append(result)
            logging.debug(f"Added result for target: {bridge_entity}")
    return results

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

    for bridge_entity, instances in results.items():
        seen_vectors = defaultdict(set)  # (layer, position) -> set of vector tuples
        logging.info(f"Performing deduplication for target {bridge_entity}")

        for instance in tqdm(instances, desc=f"Processing target {bridge_entity}"):
            is_duplicate = False

            # Track duplicates for each hidden state
            for hidden_state in instance['hidden_states']:
                layer = hidden_state['layer']
                pos = hidden_state['position']
                vector_key = get_vector_key(hidden_state)
                
                # Check if we've seen this vector before
                is_vec_duplicate = False
                for seen_vec in seen_vectors[(layer, pos)]:
                    # post_mlp, post_attention 모두 equal하다고 나와야함
                    if all(vectors_equal(v1, v2) for v1, v2 in zip(vector_key, seen_vec)):
                        is_vec_duplicate = True
                        dedup_stats[bridge_entity][f"layer{layer}_pos{pos}"] += 1
                        break
                # 하나의 instance 당 여러 (layer, pos)에 대한 hidden representation을 저장하고 있을 경우, 하나만 동일해도 duplicate라고 인식
                if is_vec_duplicate:
                    is_duplicate = True
                    break
                else:
                    seen_vectors[(layer, pos)].add(vector_key)

            # If instance is not a duplicate, add it to deduplicated results
            if not is_duplicate:
                deduplicated_results[bridge_entity].append(instance)

    # Convert defaultdict to regular dict with string keys
    final_stats = {}
    for bridge_entity, stats in dedup_stats.items():
        final_stats[bridge_entity] = dict(stats)

    return dict(deduplicated_results), final_stats



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True, help="Path to the model checkpoint")
    parser.add_argument("--layer_pos_pairs", required=True, help="List of (layer, position) tuples to evaluate")
    parser.add_argument("--save_dir", required=True, help="Directory to save the analysis results")
    parser.add_argument("--device", default="cuda:0", help="Device to run the model on")
    parser.add_argument("--merge_id_data", action="store_true", help="Combine train_inferred and test_inferred_id in grouping stage")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode for verbose output")

    args = parser.parse_args()
    setup_logging(args.debug)

    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

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

    data_dir = os.path.join(base_dir, "data", dataset)
    if "inf" in dataset:
        filtered_train_data, id_data, grouped_ood_test_data, grouped_nonsense_test_data = load_and_preprocess_data(
            os.path.join(data_dir, "atomic_facts.json"),
            os.path.join(data_dir, "test.json"),
            merge_id_data=args.merge_id_data
        )
    else:
        filtered_train_data, id_data, grouped_ood_test_data, grouped_nonsense_test_data = load_and_preprocess_data(
            os.path.join(data_dir, "train.json"),
            os.path.join(data_dir, "test.json"),
            merge_id_data=args.merge_id_data
        )
        
    logging.info(f"Number of filtered train instances: {len(filtered_train_data)}")
    if args.merge_id_data:
        logging.info(f"Number of unique bridge entity in ID: {len(id_data)}")
    else:
        grouped_id_train_data, grouped_id_test_data = id_data
        logging.info(f"Number of unique ID train targets: {len(grouped_id_train_data)}")
        logging.info(f"Number of unique ID test targets: {len(grouped_id_test_data)}")
    logging.info(f"Number of unique OOD test targets: {len(grouped_ood_test_data)}")
    logging.info(f"Number of unique nonsense test targets: {len(grouped_nonsense_test_data)}")

    layer_pos_pairs = eval(args.layer_pos_pairs)
    logging.info(f"Layer position pairs: {layer_pos_pairs}")
    
    # ToDo : If the code is changed to support multiple layer_pos_pairs, should change save_dir for each layer_pos_pair
    if args.merge_id_data:
        save_dir = os.path.join(args.save_dir, "residual", dataset, "merged_id", str(layer_pos_pairs[0]).replace(" ", ""), step)
    else:
        save_dir = os.path.join(args.save_dir, "residual", dataset, str(layer_pos_pairs[0]).replace(" ", ""), step)
    if os.path.exists(save_dir):
        logging.info(f"{save_dir} already exist!!!")
        return
    else:
        os.makedirs(save_dir, exist_ok=True)

    # save and deduplicate hidden state 
    if args.merge_id_data:
        logging.info("Processing merged in-distribution data...")
        id_results = process_data_group(model, id_data, layer_pos_pairs, tokenizer, device)

        logging.info("Performing deduplication for merged in-distribution data...")
        id_results_dedup, id_dedup_stats = deduplicate_vectors(id_results)
    else:
        logging.info("Processing in-distribution train data...")
        id_train_results = process_data_group(model, grouped_id_train_data, layer_pos_pairs, tokenizer, device)
        logging.info("Processing in-distribution test data...")
        id_test_results = process_data_group(model, grouped_id_test_data, layer_pos_pairs, tokenizer, device)

        logging.info("Performing deduplication for in-distribution train data...")
        id_train_results_dedup, id_train_dedup_stats = deduplicate_vectors(id_train_results)
        logging.info("Performing deduplication for in-distribution test data...")
        id_test_results_dedup, id_test_dedup_stats = deduplicate_vectors(id_test_results)

    logging.info("Processing out-of-distribution test data...")
    ood_test_results = process_data_group(model, grouped_ood_test_data, layer_pos_pairs, tokenizer, device)
    logging.info("Processing nonsense test data...")
    nonsense_results = process_data_group(model, grouped_nonsense_test_data, layer_pos_pairs, tokenizer, device)

    logging.info("Performing deduplication for out-of-distribution data...")
    ood_results_dedup, ood_dedup_stats = deduplicate_vectors(ood_test_results)
    logging.info("Performing deduplication for nonsense data...")
    nonsense_results_dedup, nonsense_dedup_stats = deduplicate_vectors(nonsense_results)

    # Save deduplicated results
    if args.merge_id_data:
        id_save_path = os.path.join(save_dir, "id_merged_dedup.json")
        with open(id_save_path, 'w') as f:
            json.dump(id_results_dedup, f)
        logging.info(f"Deduplicated merged in-distribution analysis results saved to {id_save_path}")

        id_stats_save_path = os.path.join(save_dir, f"dedup_stats_id_merged_dedup.json")
        with open(id_stats_save_path, 'w') as f:
            json.dump(id_dedup_stats, f)
    else:
        id_train_save_path = os.path.join(save_dir, "id_train_dedup.json")
        with open(id_train_save_path, 'w') as f:
            json.dump(id_train_results_dedup, f)
        logging.info(f"Deduplicated in-distribution train analysis results saved to {id_train_save_path}")

        id_test_save_path = os.path.join(save_dir, "id_test_dedup.json")
        with open(id_test_save_path, 'w') as f:
            json.dump(id_test_results_dedup, f)
        logging.info(f"Deduplicated in-distribution test analysis results saved to {id_test_save_path}")

        id_train_stats_save_path = os.path.join(save_dir, f"dedup_stats_id_train_dedup.json")
        with open(id_train_stats_save_path, 'w') as f:
            json.dump(id_train_dedup_stats, f)

        id_test_stats_save_path = os.path.join(save_dir, f"dedup_stats_id_test_dedup.json")
        with open(id_test_stats_save_path, 'w') as f:
            json.dump(id_test_dedup_stats, f)

    ood_save_path = os.path.join(save_dir, "ood_dedup.json")
    with open(ood_save_path, 'w') as f:
        json.dump(ood_results_dedup, f)
    logging.info(f"Deduplicated out-of-distribution analysis results saved to {ood_save_path}")

    nonsense_save_path = os.path.join(save_dir, "nonsense_dedup.json")
    with open(nonsense_save_path, 'w') as f:
        json.dump(nonsense_results_dedup, f)
    logging.info(f"Deduplicated nonsense analysis results saved to {nonsense_save_path}")

    ood_stats_save_path = os.path.join(save_dir, f"dedup_stats_ood_dedup.json")
    with open(ood_stats_save_path, 'w') as f:
        json.dump(ood_dedup_stats, f)
        
    nonsense_stats_save_path = os.path.join(save_dir, f"dedup_stats_nonsense_dedup.json")
    with open(nonsense_stats_save_path, 'w') as f:
        json.dump(nonsense_dedup_stats, f)
        
    # Log deduplication statistics and final counts
    logging.info("\nDeduplication statistics:")
    if args.merge_id_data:
        logging.info("\nIn-distribution:")
        for target, stats in id_dedup_stats.items():
            logging.info(f"\nTarget: {target}")
            for key, count in sorted(stats.items()):
                layer, pos = key.replace('layer', '').split('_pos')
                logging.info(f"  Layer {layer}, Position {pos}: Removed {count} duplicates")
    else:
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
    
    # logging.info("\nFinal counts after deduplication:")
    # logging.info("In-distribution Train:")
    # for target, instances in id_train_results_dedup.items():
    #     logging.info(f"  {target}: {len(instances)}")
    # logging.info("In-distribution Test:")
    # for target, instances in id_test_results_dedup.items():
    #     logging.info(f"  {target}: {len(instances)}")
    # logging.info("Out-of-distribution:")
    # for target, instances in ood_results_dedup.items():
    #     logging.info(f"  {target}: {len(instances)}")
    # logging.info("Nonsense:")
    # for target, instances in nonsense_results_dedup.items():
    #     logging.info(f"  {target}: {len(instances)}")    

if __name__ == "__main__":
    main()