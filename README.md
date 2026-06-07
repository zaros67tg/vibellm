# 🚀 Nano LLM from Scratch [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
A highly optimized, modern Nano Large Language Model (~30M-50M parameters) built entirely from scratch in PyTorch. 

This project goes beyond the standard GPT-2 tutorials by implementing the exact architectural enhancements used in frontier models like **Llama 3** and **Mistral**, scaled down to run efficiently on a local laptop GPU.

## ✨ Advanced Architecture Features

* **Rotary Positional Embeddings (RoPE):** Replaces absolute positional embeddings for better length generalization.
* **SwiGLU Feed-Forward Networks:** Replaces standard ReLU/GELU activations for higher parameter efficiency.
* **RMSNorm & Pre-LN Structure:** Ensures training stability in deep networks while omitting unnecessary mean subtraction.
* **FlashAttention:** Utilizes PyTorch's `scaled_dot_product_attention` for hardware-accelerated, memory-efficient attention.
* **High-Density Tokenization:** Uses `tiktoken` (`cl100k_base`) rather than a character-level tokenizer, allowing the nano model to dedicate its parameter budget to logic and context rather than spelling.
* **Memory-Mapped Datasets:** Implements `np.memmap` for the DataLoader, allowing the streaming of massive binary datasets without crashing system RAM.
* **Weight Tying:** Shares weights between the input embedding layer and the output projection layer to save ~25M parameters.

## 📂 Project Structure

* `train_nano_llm.py` - The complete, self-contained end-to-end pipeline (Model, DataLoader, Optimizer, Training Loop, and Generation).
* `data/` - Directory for datasets (generated automatically).
* `requirements.txt` - Python dependencies.

## 🛠️ Installation

1. Clone the repository:
   ```bash
   git clone [https://github.com/yourusername/nano-llm-scratch.git](https://github.com/yourusername/nano-llm-scratch.git)
   cd nano-llm-scratch

2. Install the required dependencies:

       '''Bash
       pip install torch numpy tiktoken

🚀 Quick Start Guide
The entire pipeline is handled through the single train_nano_llm.py script.

Step 1: Prepare the Data
Download the Tiny Shakespeare dataset, tokenize it using the cl100k_base vocabulary, and save it as highly optimized binary arrays (train.bin and val.bin).

    '''Bash
    python train_nano_llm.py --prepare-data

Step 2: Train the Model
Kick off the training loop. By default, this uses Mixed Precision (AMP Autocast) if a compatible CUDA GPU is detected.

    '''Bash
    python train_nano_llm.py --train

Optional: You can tweak hyperparameters directly from the command line:

    '''Bash
    python train_nano_llm.py --train --batch-size 16 --max-steps 2000 --learning-rate 6e-4

Step 3: Generate Text
Once trained (or even from a random initialization), you can prompt the model to generate text autoregressively.

    '''Bash
    python train_nano_llm.py --generate "To be, or not to be" --max-new-tokens 150

📊 Model Architecture (Default Config)
Vocabulary Size: 100,277

Hidden Dimension: 256

Attention Heads: 8

Layers: 6

Context Window: 256

Total Parameters: ~30.6M

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
