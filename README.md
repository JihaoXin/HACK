# HACK (Homomorphic Algebraic Compression Kernels)

HACK is a research project focused on accelerating LLM inference through HPC-aware optimizations. This repository contains the code and tools necessary to explore and evaluate various compression techniques for Large Language Models.

## Project Structure
- `run_llama3.1-8b.ipynb`: Jupyter notebook for running experiments with Llama 3.1-8B model, including perplexity evaluation and compression analysis
- `transformers/`: Modified Hugging Face Transformers library as a submodule
- `data/`: Storage for singular values and compression metricss
- `logs/`: Output logs for experiments and performance metrics

## Setup Instructions

### Prerequisites

- Python 3.12.4
- PyTorch 2.6.0+cu124'

### Clone the Repository

```bash
git clone https://github.com/JihaoXin/hack.git
cd hack
```

### Initialize Transformers Submodule

```bash
git submodule add git@github.com:JihaoXin/transformers.git
cd transformers
git checkout hack
pip install -e .
cd ..
```

### Download Model

Create a models directory and download Llama-3.1-8B from Hugging Face:

```bash
mkdir -p model
cd model

# Option 1: If you have Hugging Face CLI installed
huggingface-cli download meta-llama/Meta-Llama-3.1-8B --local-dir ./Meta-Llama-3.1-8B

# Option 2: Using Git LFS
git lfs install
git clone https://huggingface.co/meta-llama/Meta-Llama-3.1-8B
```

Note: You need to have access to the Meta-Llama-3.1-8B model on Hugging Face, which requires accepting the model's license terms on the Hugging Face website.

## Running Experiments

### Example Usage

Here's a simple example of how to run inference with the HACK-modified model:

```python
import torch
from transformers import LlamaForCausalLM, LlamaTokenizer, LlamaConfig

# Reset the MLP counter before inference
from transformers.models.llama.modeling_llama import reset_MLP_counter

# Initialize model with compression configuration
model_path = "./model/Meta-Llama-3.1-8B"
config = LlamaConfig.from_pretrained(model_path)
config.topk_svd = 0.5  # Set compression rate (retain 50% of singular values)

# Load model and tokenizer
tokenizer = LlamaTokenizer.from_pretrained(model_path)
model = LlamaForCausalLM.from_pretrained(
    model_path,
    config=config,
    torch_dtype=torch.float16,  # Use half precision for efficiency
    device_map="auto"
)

# Reset MLP counter before inference
reset_MLP_counter()

# Prepare input
prompt = "Explain how matrix decomposition works in machine learning:"
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

# Run inference (you might need to modify the LlamaMLP.forward mode value)
with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_length=200,
        temperature=0.7,
    )

# Decode output
generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(generated_text)
```

To change the compression mode, modify the `mode` variable in the `LlamaMLP.forward` method in `modeling_llama.py`:

```python
def forward(self, x):
    mode = 2  # 0: original, 1: full SVD, 2: tile-wise SVD
    # ... rest of the method ...
```


## Implementation Details

### Modified LlamaMLP

We've extended the standard LlamaMLP implementation in the Transformers library to support tile-wise SVD compression. The key modifications in `modeling_llama.py` include:

- Added tile-wise SVD functionality that processes the weight matrices in small tiles (128x128 by default)
- Implemented SVD-based compression with configurable rank parameter (topk)
- Added visualization capabilities for weight matrices through heatmap generation
- Reset counter functionality to track MLP layer processing

The implementation supports three operation modes:
1. Original execution (no compression) - `mode = 0`
2. Full SVD compression - `mode = 1`
3. Tile-wise SVD compression - `mode = 2` (current focus)

This approach allows us to maintain computational efficiency while achieving significant compression rates.

### Compression Approach

The tile-wise SVD technique works by:
1. Dividing large weight matrices into manageable tiles (default 128x128)
2. Performing SVD on each tile independently
3. Retaining only the top-k singular values (controlled by the `topk_svd` parameter)
4. Reconstructing the weight matrices from compressed representations

This approach enables:
- Better memory locality for hardware acceleration
- Parallel processing of independent tiles
- Adjustable compression rates for different accuracy targets

