## The Coverage Principle: A Framework for Understanding Compositional Generalization

>We study whether transformers can learn to implicitly reason over parametric knowledge, a skill that even the most capable language models struggle with. Focusing on two representative reasoning types, composition and comparison, we consistently find that transformers can learn implicit reasoning, but only through grokking, i.e., extended training far beyond overfitting. The levels of generalization also vary across reasoning types: when faced with out-of-distribution examples, transformers fail to systematically generalize for composition but succeed for comparison. We delve into the model's internals throughout training, conducting analytical experiments that reveal: 1) the mechanism behind grokking, such as the formation of the generalizing circuit and its relation to the relative efficiency of generalizing and memorizing circuits, and 2) the connection between systematicity and the configuration of the generalizing circuit. Our findings guide data and training setup to better induce implicit reasoning and suggest potential improvements to the transformer architecture, such as encouraging cross-layer knowledge sharing. Furthermore, we demonstrate that for a challenging reasoning task with a large search space, GPT-4-Turbo and Gemini-1.5-Pro based on non-parametric memory fail badly regardless of prompting styles or retrieval augmentation, while a fully grokked transformer can achieve near-perfect accuracy, showcasing the power of parametric memory for complex reasoning.


### File Structure
```
GrokkedTranformer/
├─  {composition/comparison/complex_reasoning}.ipynb: scripts for training/evaluation data generation
├─  data/: cached training/evaluation data
├─  main.py: main script for model training
├─  eval_qa.py: evaluation script for trained model
├─  causal_tracing_{composition/comparison}.py: causal tracing & logit lens
├─  LLM/: cached testing data & model outputs for LLMs based on non-parametric memory
    ├─ {prompt/retrieval}_{directna/cot}_*.txt: input for the setting {without/with} retrieval augmentation and {without/with} CoT
    ├─ answer_*.txt: ground truth answer
    ├─ {gemini/gpt4turbo}_*.txt: predictions of Gemini-Pro-1.5 and GPT-4-Turbo
├─  LLM.ipynb: evaluation script and cached evaluation results for LLMs
└── utils.py: other helper functions
```

### Environmental Setup
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
```

### Data Preparation
- Download from [link](https://buckeyemailosu-my.sharepoint.com/:f:/g/personal/wang_13930_buckeyemail_osu_edu/EghpRAb3V71FnQsi44nuAfsB47HZSmmWuxt5DML2hqtM7w?e=TWeYkW) and unzip into data/, or alternatively, run ```{composition/comparison/complex_reasoning}.ipynb``` to generate the data
- Download from [link](https://buckeyemailosu-my.sharepoint.com/:f:/g/personal/wang_13930_buckeyemail_osu_edu/EiTbt6SLSLhLrJd_kgJJBtIBPerEzHziFVsmn98pP8sSZQ?e=KUaI0d) and unzip into LLM/

### Model Training
```bash
MODEL_PATH=gpt2

DATASET=data/$1/
WEIGHT_DECAY=$2
N_LAYERS=$3
GPU=$4

OUTPUT_DIR=<your_dir>/$1_$2_$3

CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 --master_port 12345 main.py \
# CUDA_VISIBLE_DEVICES=$GPU python main.py \
--data_dir $DATASET \
--model_name_or_path ${MODEL_PATH} \
--weight_decay $WEIGHT_DECAY \
--output_dir $OUTPUT_DIR \
--max_seq_length 10 \
--max_length 10 \
--block_size 10 \
--train_batch_size 512 \
--eval_batch_size 512 \
--learning_rate 1e-4 \
--gradient_accumulation_steps 1 \
--save_step 50000 \
--save_step_dense 40000 \
--max_steps 1500000 \
--do_train \
--scheduler constant_schedule_with_warmup \
--fp16 \
--evaluate_during_training \
--predict_during_training \
--init_weights \
--add_tokens \
--n_layer $N_LAYERS
```


### Logit lens & Causal tracing
```bash
python causal_tracing_{comparison/composition}.py \
    --dataset <dataset_name> \
    --model_dir <your_dir> \
    --save_path <your_save_path> \
    --num_layer <number_layer_of_model> \
    --wd <weight_decay_used>
```
#### example
```bash
# example
python causal_tracing_{comparison/composition}.py --dataset composition.2000.200.9.0 --model_dir <dir_path> --save_path <save_dir_path> --num_layer 8 --wd 0.1
```
- this will load <dir_path>/{comparison/composition}.2000.200.9.0_0.1_8 model checkpoints, and save the all checkpoints result in <save_dir_path>/{comparison/composition}-2000.200.9.0_0.1_8.json file

