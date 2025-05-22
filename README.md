# The Coverage Principle: A Framework for Understanding Compositional Generalization

This repository contains code for **“The Coverage Principle: A Framework for Understanding Compositional Generalization.”**

## Abstract

Large language models excel at pattern matching, yet often fall short in systematic compositional generalization. We propose the coverage principle: a data-centric framework showing that models relying primarily on pattern matching for compositional tasks cannot reliably generalize beyond substituting fragments that yield identical results when used in the same contexts. We demonstrate that this framework has a strong predictive power for the generalization capabilities of Transformers. First, we derive and empirically confirm that the training data required for two-hop generalization grows at least quadratically with the token set size, and the training data efficiency does not improve with 20x parameter scaling. Second, for compositional tasks with path ambiguity where one variable affects the output through multiple computational paths, we show that Transformers learn context-dependent state representations that undermine both performance and interpretability. Third, Chain-of-Thought supervision improves training data efficiency for multi-hop tasks but still struggles with path ambiguity. Overall, the coverage principle provides a unified lens for understanding compositional reasoning, and underscores the need for fundamental architectural or training innovations to achieve truly systematic compositionality.

## File Structure
```
coverage-principle/
├── dataset\_generation/: scripts for training/evaluation data generation
├── data/: cached training/evaluation data
├── main.py: main script for model training
├── determine\_coverage.py: coverage determination algorithm
└── circuit\_analysis/: cosine similarity and causal tracing analysis

````

## Environmental Setup
```bash
conda create -n coverage-principle python=3.10
conda activate coverage-principle

pip install torch==1.13.1+cu116 torchvision==0.14.1+cu116 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu116

cd transformers
pip install -e .
cd ..

cd simpletransformers
pip install -e .
cd ..
````

## Data Preparation

### Generate Synthetic Dataset

Use the dataset-generation scripts to create synthetic compositional tasks. For example, to generate a 2-hop compositional dataset:

```bash
cd dataset_generation
python twohop.py --num_tokens 50 --max_train_data_num 10000 --default_seen_ratio 0.7 --test_size_for_type 2000 --seed 42
```

**Key arguments**

| Argument               | Description                                           |
| ---------------------- | ----------------------------------------------------- |
| `--num_tokens`         | Size of the token vocabulary                          |
| `--max_train_data_num` | Maximum number of training examples                   |
| `--default_seen_ratio` | Fraction of each function’s domain marked as **seen** |
| `--test_size_for_type` | Number of test samples for each coverage type         |
| `--cot`                | Enable Chain-of-Thought supervision                   |

This creates a dataset in `data/twohop.50.10000.diff-f12.inf/` containing:

* `train.json` – training data
* `test.json` – test data with coverage-type annotations
* `atomic_facts_f1.json`, `atomic_facts_f2.json` – primitive-function mappings
* `vocab.json` – vocabulary tokens

## Coverage Determination

To analyze which test examples fall within the coverage of your training data:

```bash
python determine_coverage.py --data_dir data/twohop.50.10000.diff-f12.inf/ --min_evidence 1 --k_sweep
```

**Key arguments**

| Argument         | Description                                               |
| ---------------- | --------------------------------------------------------- |
| `--data_dir`     | Path to dataset directory                                 |
| `--min_evidence` | Minimum evidence threshold *k* for functional equivalence |
| `--k_sweep`      | Run analysis for multiple *k* values                      |
| `--visualise`    | Generate graph visualization of coverage                  |
| `--ground_truth` | Use ground-truth functional equivalence (f1 only)         |

Outputs include:

* `k_sweep_results/` – coverage analysis results for different *k* values
* `test_annotated.json` – test data with coverage annotations
* Coverage visualization (if `--visualise` is used)

## Model Training

Train a GPT-2 model on the generated dataset:

```bash
bash script/train.sh twohop.50.10000.diff-f12.inf 0.1 8 12 0 42
```

**Script arguments**

1. Dataset name (e.g., `twohop.50.10000.diff-f12.inf`)
2. Weight decay (e.g., `0.1`)
3. Number of layers (e.g., `8`)
4. Number of attention heads (e.g., `12`)
5. GPU ID (e.g., `0`)
6. Random seed (e.g., `42`)

**Training configuration**

* Architecture: GPT-2 with specified layers/heads
* Learning rate: `1e-4`
* Batch size: `4096`
* Max steps: `62500`
* Distributed training on 4 GPUs

Trained models are saved in `CKPT_DIR/trained_checkpoints/`.

## Analysis

### Cosine Similarity Analysis

Analyze how models form clustered representations of functionally equivalent components:

```bash
cd circuit_analysis/hierarchical/2-hop
python collapse_analysis_2-hop.py \
    --ckpt CKPT_DIR/trained_checkpoints/twohop.50.10000.diff-f12.inf_wd-0.1_layer-8_head-12_seed-42/final_checkpoint \
    --layer_pos_pairs "[(3,1)]" \
    --save_dir ./results/cosine_analysis \
    --atomic_idx 1 \
    --mode residual
```

### Causal Tracing Analysis

Identify which representations are causally important:

```bash
cd circuit_analysis/hierarchical/2-hop
python causal_tracing_2-hop.py \
    --model_dir CKPT_DIR/trained_checkpoints/twohop.50.10000.diff-f12.inf_wd-0.1_layer-8_head-12_seed-42 \
    --step_list final_checkpoint \
    --data_dir ./data_fixed \
    --batch_size 1024 \
    --metric_type rank
```

## License

This project is licensed under the MIT License.
