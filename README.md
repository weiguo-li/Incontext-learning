# Dual-Form Attention: Transformer Architecture Verification

This project reimplements and extends the methods from "Why Can GPT Learn In-Context? Language Models Implicitly Perform Gradient Descent as Meta-Optimizers" (Dai et al., 2022) using Qwen models. It implements relaxed linear attention mechanisms to verify the dual form of transformer architecture, focusing on in-context learning capabilities through attention analysis.

## Overview

The project investigates the relationship between fine-tuning and in-context learning (ICL) in transformer models by analyzing attention patterns and hidden state similarities. Building on the theoretical framework from Dai et al. (2022), this implementation adapts their methods to work with Qwen3 models, providing custom attention mechanisms and various metrics to understand how transformers process contextual information.

## Key Components

### Core Files

- **`customized_Qwen.py`**: Custom Qwen3 attention implementation with raw attention weight extraction capabilities
- **`metric.py`**: Implementation of similarity metrics (SimAOU, SimAM, Kendall's tau)
- **`utility.py`**: Utility functions for data processing, evaluation, and feature extraction
- **`main.py`**: Command-line interface for running experiments
- **`test.ipynb`**: Jupyter notebook for interactive testing and experimentation

### Key Features

1. **Custom Attention Module**: `Qwen3Attention_v1` with raw attention weight storage
2. **Attention Weight Collection**: `AttentionWeightsCollector` for extracting attention patterns
3. **Similarity Metrics**:
   - `SimAOU`: Similarity of Activations after Optimized Update
   - `SimAM`: Similarity of Attention Maps
   - `Kendall`: Kendall's tau correlation analysis

## Installation

This project uses `uv` for dependency management. Install dependencies with:

```bash
uv sync
```

Or install manually:
```bash
pip install torch transformers datasets tqdm scipy numpy
```

## Usage

### Command Line Interface

The project provides a comprehensive CLI through `main.py`:

#### Data Selection
```bash
uv run main.py data select --data-type sst2 --num-data-points 32 --seed-max 100
```

#### Fine-tuning Evaluation
```bash
uv run main.py finetune_eval --data-type sst2 --num-data-points 32 --seed 0
```

#### Similarity Metrics
```bash
# SimAOU (Similarity of Activations after Optimized Update)
uv run main.py SimAOU --data-type sst2 --seed 30

# SimAM (Similarity of Attention Maps)  
uv run main.py SimAM --data-type sst2 --seed 30

# Kendall's tau correlation
uv run main.py kendall --data-type sst2 --seed 30
```

### Programmatic Usage

```python
from utility import extract_attn_weights, extract_hiddenstates
from metric import SimAOU, SimAM, Kendall
from customized_Qwen import Qwen3Attention_v1, AttentionWeightsCollector

# Load model with custom attention
from transformers.models.qwen3 import modeling_qwen3
modeling_qwen3.Qwen3Attention = Qwen3Attention_v1

# Extract attention weights
attention_weights = extract_attn_weights(model, tokenizer, test_data)

# Calculate similarity metrics
sim_scores = SimAOU(model, tokenizer, train_data, test_data)
```

## Supported Datasets

- **SST-2**: Stanford Sentiment Treebank (binary classification)
- **SST-5**: Stanford Sentiment Treebank (5-class classification)

## Key Functions

### Attention Analysis
- `extract_attn_weights()`: Extract attention weights from model layers
- `extract_attentionweights()`: Extract raw attention scores before softmax
- `get_query_states()`: Extract query states from attention layers

### Evaluation
- `evaluate_demonstrations()`: Evaluate model performance with demonstrations
- `evaluate_zeroshot()`: Zero-shot evaluation
- `evaluate_finetuning()`: Fine-tuned model evaluation

### Training
- `enable_kv_only_training()`: Enable training only key and value parameters
- `finetune_model_eval()`: Fine-tune model and evaluate

## Metrics Explanation

### SimAOU (Similarity of Activations after Optimized Update)
Measures the similarity between hidden state changes in in-context learning vs. fine-tuning scenarios.

### SimAM (Similarity of Attention Maps)  
Compares attention patterns between different learning paradigms using cosine similarity.

### Kendall's Tau
Calculates rank correlation between inner products of query states and attention weights to demonstration tokens.

## Model Architecture

The project uses Qwen3-0.6B as the base model with custom attention mechanisms:
- Custom `Qwen3Attention_v1` class with raw weight storage
- `eager_attention_forward()` function for attention computation
- Support for grouped query attention via `repeat_kv()`

## Experimental Setup

The experiments typically involve:
1. Selecting optimal demonstration examples via `data_selection()`
2. Extracting features (attention weights, hidden states) for different conditions
3. Fine-tuning models on demonstration data
4. Computing similarity metrics between ICL and fine-tuning representations

## File Structure

```
dual-form-attention/
├── customized_Qwen.py     # Custom Qwen attention implementation
├── main.py                # CLI interface
├── metric.py             # Similarity metrics implementation  
├── utility.py            # Utility functions
├── test.ipynb           # Interactive testing notebook
├── tinytransformer.py   # Experimental linear attention (unused)
├── pyproject.toml       # Project configuration
└── README.md           # This file
```

## Citation

This work reimplements and extends the methods described in:

```bibtex
@Inproceedings{Dai2022WhyCG,
 author = {Damai Dai and Yutao Sun and Li Dong and Y. Hao and Shuming Ma and Zhifang Sui and Furu Wei},
 title = {Why Can GPT Learn In-Context? Language Models Implicitly Perform Gradient Descent as Meta-Optimizers},
 year = {2022}
}
```

## Contributing

This project is part of research on dual-form transformer architectures. For questions or contributions, please refer to the codebase documentation and inline comments.

## Notes

- The project requires GPU support for efficient model inference
- Memory usage is optimized through garbage collection and CUDA cache management
- Batch sizes may need adjustment based on available GPU memory
- The `AttentionWeightsCollector` enables extraction of attention weights before softmax normalization

