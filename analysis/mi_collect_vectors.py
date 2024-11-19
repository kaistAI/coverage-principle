import argparse
import torch
import json
from tqdm import tqdm
import os
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import logging
import numpy as np

def setup_logging(debug_mode):
    level = logging.DEBUG if debug_mode else logging.ERROR
    logging.basicConfig(level=level, format='%(asctime)s - %(levelname)s - %(message)s')

def load_dataset(dataset_path, debug):
    with open(dataset_path, 'r') as f:
        dataset = json.load(f)
        if debug:
            dataset = dataset[:20]
    logging.debug(f"Loaded {len(dataset)} instances from {dataset_path}")
    return dataset

def get_hidden_states(model, input_text, tokenizer, device):
    """
    Collect input embeddings (after positional embedding) and post_mlp outputs 
    for all positions and layers.
    """
    inputs = tokenizer(input_text, return_tensors="pt").to(device)
    
    # Dictionary to store outputs
    activation = {}
    
    def get_activation(name):
        def hook(model, input, output):
            activation[name] = output
        return hook

    # Register hook for input embeddings (after position embeddings)
    hooks = []
    hooks.append(model.transformer.drop.register_forward_hook(
        get_activation('input_embedding')))
    
    # Register hooks for MLP outputs
    for i in range(len(model.transformer.h)):
        hooks.append(model.transformer.h[i].mlp.register_forward_hook(
            get_activation(f'layer{i+1}_mlp')))

    with torch.no_grad():
        outputs = model(**inputs)

    # Get sequence length
    seq_length = inputs['input_ids'].shape[1]
    
    # Initialize list to store results for all positions
    all_hidden_states = []
    
    # Get complete input embeddings (after position embeddings are added)
    input_embeddings = activation['input_embedding']
    if isinstance(input_embeddings, tuple):
        input_embeddings = input_embeddings[0]
    
    # Process each position
    for pos in range(seq_length):
        hidden_states = []
        
        # Add complete input embedding (layer 0)
        embedding = input_embeddings[0, pos, :].detach().cpu().numpy()
        hidden_states.append({
            'layer': 0,
            'position': pos,
            'vector': embedding.tolist()
        })
        
        # Add post_mlp outputs for each transformer layer
        for layer in range(len(model.transformer.h)):
            mlp_output = activation[f'layer{layer+1}_mlp']
            if isinstance(mlp_output, tuple):
                mlp_output = mlp_output[0]
            
            post_mlp = mlp_output[0, pos, :].detach().cpu().numpy()
            hidden_states.append({
                'layer': layer + 1,
                'position': pos,
                'vector': post_mlp.tolist()
            })
        
        all_hidden_states.append(hidden_states)

    # Remove hooks
    for hook in hooks:
        hook.remove()

    return all_hidden_states

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True, help="Path to the model checkpoint")
    parser.add_argument("--dataset", required=True, help="Path to the evaluation dataset")
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
    
    # Initialize results dictionary to store instances by type
    results = {}
    
    for idx, instance in enumerate(tqdm(dataset)):
        logging.debug(f"Processing instance {idx}")
        
        if 'type' not in instance or 'input_text' not in instance:
            logging.error(f"Instance {idx} missing 'type' or 'input_text': {instance}")
            continue
        
        # Initialize group if not exists
        instance_type = instance['type']
        if instance_type not in results:
            results[instance_type] = []
        
        input_text = instance["input_text"]
        hidden_states = get_hidden_states(model, input_text, tokenizer, device)
        
        result = {
            "input_text": input_text,
            "target_text": instance.get("target_text", "No target text"),
            "hidden_states": hidden_states
        }
        
        results[instance_type].append(result)
        logging.debug(f"Added result for {instance_type}")
    
    # Save results
    os.makedirs(args.save_dir, exist_ok=True)
    save_path = os.path.join(args.save_dir, args.save_fname)
    
    with open(save_path, 'w') as f:
        json.dump(results, f)
    
    # Log final counts
    logging.info(f"\nFinal counts:")
    for key, value in results.items():
        logging.info(f"  {key}: {len(value)}")
    
    logging.info(f"\nResults saved to {save_path}")

if __name__ == "__main__":
    main()