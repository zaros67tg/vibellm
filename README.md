# 🚀 Nano LLM from Scratch

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
