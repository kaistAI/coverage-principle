import numpy as np
import torch
from latentmi import lmi
import json
from collections import defaultdict
from tqdm import tqdm
import logging
from typing import Dict, List, Tuple
import os
from transformers import GPT2LMHeadModel, GPT2Tokenizer

class GPTLMIAnalyzer:
    def __init__(self, model_path: str, debug: bool = False):
        """
        Initialize the analyzer for GPT representations.
        
        Args:
            model_path: Path to the model checkpoint (HuggingFace model ID or local path)
            debug: Enable debug logging
        """
        self.debug = debug
        self.setup_logging()
        
        # Load model and tokenizer
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.load_model_and_tokenizer(model_path)
        
    def setup_logging(self):
        level = logging.DEBUG if self.debug else logging.INFO
        logging.basicConfig(
            level=level,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
    
    def load_model_and_tokenizer(self, model_path: str):
        """Load GPT model and tokenizer"""
        self.model = GPT2LMHeadModel.from_pretrained(model_path).to(self.device)
        self.tokenizer = GPT2Tokenizer.from_pretrained(model_path)
        
        # Set padding configuration
        self.tokenizer.padding_side = "left"
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model.config.pad_token_id = self.model.config.eos_token_id
        
        # Get embedding matrices
        self.embedding = self.model.transformer.wte.weight.detach().cpu().numpy()
        self.unembedding = self.model.lm_head.weight.detach().cpu().numpy()
        
        logging.debug(f"Loaded model from {model_path}")
        logging.debug(f"Embedding shape: {self.embedding.shape}")
        logging.debug(f"Unembedding shape: {self.unembedding.shape}")
        
    def process_hidden_states(self, hidden_states_data: List) -> Dict[int, np.ndarray]:
        """
        Process hidden states from the saved format into numpy arrays.
        
        Args:
            hidden_states_data: List of hidden states from the saved results
            
        Returns:
            Dict mapping layer index to numpy array of shape (n_positions, hidden_dim)
        """
        layer_vectors = defaultdict(list)
        
        # Group vectors by layer
        for position_states in hidden_states_data:
            for state in position_states:
                layer = state['layer']
                vector = state['vector']
                layer_vectors[layer].append(vector)
                
        # Convert to numpy arrays
        return {layer: np.array(vectors) for layer, vectors in layer_vectors.items()}

    def analyze_layer_representations(
        self,
        results: Dict,
        save_dir: str,
        save_prefix: str,
        debug: bool
    ):
        """
        Perform comprehensive LMI analysis for each instance type and layer,
        using preprocessed embeddings that include positional information.
        """
        analysis_results = defaultdict(dict)
        
        for instance_type, instances in results.items():
            logging.info(f"Analyzing instance type: {instance_type}")
            
            # Collect all hidden states and embeddings for this instance type
            input_tokens = []
            target_tokens = []
            layer_states = defaultdict(list)
            
            for instance in instances:
                # Use preprocessed embeddings directly from the file
                input_text = instance['input_text']
                target_text = instance['target_text']
                
                # Retrieve input embeddings (already includes token + positional embedding)
                input_embeds = []
                for position_states in instance['hidden_states']:
                    for state in position_states:
                        if state['layer'] == 0:  # Only layer 0 contains the input embeddings
                            input_embeds.append(state['vector'])
                
                # Convert input embeddings list to numpy array
                input_embeds = np.array(input_embeds)
                input_tokens.append(input_embeds)
                
                # Retrieve target embeddings for mutual information estimation with output
                target_ids = self.tokenizer(target_text, return_tensors="pt")['input_ids'][0]
                target_embed = self.unembedding[target_ids.numpy()[-1]]  # Use last token as target
                target_tokens.append(target_embed)
                
                # Process hidden states by layer and position
                instance_states = self.process_hidden_states(instance['hidden_states'])
                for layer, states in instance_states.items():
                    layer_states[layer].append(states)
            
            # Convert input tokens and target tokens to numpy arrays for MI estimation
            input_tokens = np.stack(input_tokens)
            target_tokens = np.stack(target_tokens)
            
            # Analyze each layer
            for layer in tqdm(sorted(layer_states.keys())):
                layer_result = {}
                states = np.stack(layer_states[layer])
                
                n_samples = states.shape[0]
                n_positions = states.shape[1]  # Usually 3 for your case
                hidden_dim = states.shape[2]
                
                # 1. Position-wise MI with input (I(X;T_i))
                pos_input_mi = []
                for pos in range(n_positions):
                    mi, _ = self.estimate_mutual_information(
                        input_tokens.reshape(n_samples, -1),  # Flatten input embeddings
                        states[:, pos, :]
                    )
                    pos_input_mi.append({
                        'mi': float(mi)
                    })
                
                # 2. Position-wise MI with output (I(T_i;Y))
                pos_output_mi = []
                for pos in range(n_positions):
                    mi, _ = self.estimate_mutual_information(
                        states[:, pos, :],
                        target_tokens
                    )
                    pos_output_mi.append({
                        'mi': float(mi)
                    })
                
                # 3. Joint position MI (I(X;T_1,T_2,T_3))
                joint_input_mi, _ = self.estimate_mutual_information(
                    input_tokens.reshape(n_samples, -1),
                    states.reshape(n_samples, -1)  # Flatten all positions
                )
                
                # 4. Joint position MI (I(T_1,T_2,T_3;Y))
                joint_output_mi, _ = self.estimate_mutual_information(
                    states.reshape(n_samples, -1),
                    target_tokens
                )
                
                # Store results without pmis
                layer_result.update({
                    'position_wise_input_mi': pos_input_mi,
                    'position_wise_output_mi': pos_output_mi,
                    'joint_input_mi': float(joint_input_mi),
                    'joint_output_mi': float(joint_output_mi)
                })
                
                # Calculate interaction information
                total_pos_input_mi = sum(p['mi'] for p in pos_input_mi)
                total_pos_output_mi = sum(p['mi'] for p in pos_output_mi)
                
                layer_result.update({
                    'input_interaction': float(joint_input_mi - total_pos_input_mi),
                    'output_interaction': float(joint_output_mi - total_pos_output_mi)
                })
                
                analysis_results[instance_type][f'layer_{layer}'] = layer_result
                
                if debug:
                    break
            
            # Save results for this instance type
            save_path = os.path.join(
                save_dir,
                f"{save_prefix}_{instance_type}_analysis.json"
            )
            with open(save_path, 'w') as f:
                json.dump(analysis_results[instance_type], f, indent=2)
            
            logging.info(f"Saved analysis results for {instance_type} to {save_path}")
            
            if debug:
                break
        
        return analysis_results
    
    def estimate_mutual_information(
        self,
        X: np.ndarray,
        Y: np.ndarray,
        quiet: bool = False
    ) -> Tuple[float, np.ndarray]:
        """
        Estimate mutual information between X and Y using LMI.
        
        Returns:
            mi_estimate: Average MI value
            pmis: Pointwise mutual information values
        """
        pmis, _, _ = lmi.estimate(X, Y, quiet=quiet, n_dims=16, epochs=500, regularizer="models.AEMINE")
        mi_estimate = np.nanmean(pmis)
        return mi_estimate, pmis

def main():
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", required=True, help="Path to GPT model")
    parser.add_argument("--results_path", required=True, help="Path to hidden states results")
    parser.add_argument("--save_dir", required=True, help="Directory to save analysis")
    parser.add_argument("--save_prefix", default="lmi", help="Prefix for saved files")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    
    args = parser.parse_args()
    
    # Initialize analyzer
    analyzer = GPTLMIAnalyzer(args.model_path, args.debug)
    
    # Load results
    with open(args.results_path, 'r') as f:
        results = json.load(f)
    
    # Perform analysis
    analysis_results = analyzer.analyze_layer_representations(
        results,
        args.save_dir,
        args.save_prefix,
        args.debug
    )
    
    logging.info("Analysis complete!")

if __name__ == "__main__":
    main()