import os
import sys
import math
import time
import argparse
import urllib.request
from dataclasses import dataclass
from typing import Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# =============================================================================
# 1. MODEL ARCHITECTURE
# =============================================================================

class RMSNorm(nn.Module):
    """
    Root Mean Square Layer Normalization (RMSNorm).
    Replaces standard LayerNorm by omitting the mean subtraction and bias terms,
    providing better computational efficiency and comparable training stability.
    Reference: https://arxiv.org/abs/1910.07467
    """
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Calculate RMS along the last dimension
        variance = x.pow(2).mean(-1, keepdim=True)
        return x * torch.rsqrt(variance + self.eps) * self.weight


def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0) -> torch.Tensor:
    """
    Precompute complex rotary positional embedding (RoPE) frequencies.
    Args:
        dim: Head dimension (must be even).
        end: Maximum sequence length.
        theta: Scale factor for frequency base.
    """
    assert dim % 2 == 0, "RoPE dimension must be even."
    # Compute the theta_i values
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end, device=freqs.device, dtype=torch.float32)
    freqs = torch.outer(t, freqs)  # Shape: (end, dim // 2)
    # Convert polar coordinates (r=1, angle=freqs) to complex numbers
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # Shape: (end, dim // 2)
    return freqs_cis


def reshape_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """
    Reshape precomputed frequencies to broadcast with complex query/key tensors.
    """
    ndim = x.ndim
    assert 0 <= 1 < ndim
    assert freqs_cis.shape == (x.shape[1], x.shape[-1])
    shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
    return freqs_cis.view(*shape)


def apply_rotary_emb(
    xq: torch.Tensor, 
    xk: torch.Tensor, 
    freqs_cis: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Apply precomputed complex rotary embeddings to query and key tensors.
    xq: (B, T, n_heads, head_dim)
    xk: (B, T, n_kv_heads, head_dim)
    freqs_cis: (T, head_dim // 2) complex tensor
    """
    b, t, n_heads, head_dim = xq.shape
    _, _, n_kv_heads, _ = xk.shape
    
    # Reshape and view as complex numbers (grouping head_dim into pairs of real/imaginary parts)
    xq_ = torch.view_as_complex(xq.float().reshape(b, t, n_heads, -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(b, t, n_kv_heads, -1, 2))
    
    # Reshape freqs_cis for broadcasting: (1, T, 1, head_dim // 2)
    freqs_cis_xq = reshape_for_broadcast(freqs_cis, xq_)
    
    # Apply rotation via complex multiplication and unpack back to real values
    xq_out = torch.view_as_real(xq_ * freqs_cis_xq).flatten(3)
    xk_out = torch.view_as_real(xk_ * freqs_cis_xq).flatten(3)
    
    return xq_out.type_as(xq), xk_out.type_as(xk)


class FeedForward(nn.Module):
    """
    SwiGLU (Swish Gated Linear Unit) Feed-Forward Network.
    Outperforms standard MLP by multiplying a Swish-activated gate with a linear projection.
    Reference: https://arxiv.org/abs/2002.05202
    """
    def __init__(self, dim: int, hidden_dim: int):
        super().__init__()
        self.w1 = nn.Linear(dim, hidden_dim, bias=False)  # Gate projection
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)  # Up projection
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)  # Down projection

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # F.silu implements Swish (x * sigmoid(x)) in PyTorch
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class Attention(nn.Module):
    """
    Multi-Head Attention block utilizing Rotary Positional Embeddings (RoPE)
    and PyTorch's scaled_dot_product_attention (FlashAttention-enabled).
    """
    def __init__(self, dim: int, n_heads: int, head_dim: int, dropout: float = 0.0):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = head_dim
        
        # Projections
        self.wq = nn.Linear(dim, n_heads * head_dim, bias=False)
        self.wk = nn.Linear(dim, n_heads * head_dim, bias=False)
        self.wv = nn.Linear(dim, n_heads * head_dim, bias=False)
        self.wo = nn.Linear(n_heads * head_dim, dim, bias=False)
        
        self.dropout = dropout

    def forward(self, x: torch.Tensor, freqs_cis: torch.Tensor) -> torch.Tensor:
        b, t, _ = x.shape
        
        # Project queries, keys, and values
        xq = self.wq(x).view(b, t, self.n_heads, self.head_dim)
        xk = self.wk(x).view(b, t, self.n_heads, self.head_dim)
        xv = self.wv(x).view(b, t, self.n_heads, self.head_dim)
        
        # Apply Rotary Position Embeddings (RoPE)
        xq, xk = apply_rotary_emb(xq, xk, freqs_cis=freqs_cis)
        
        # Transpose to (B, n_heads, T, head_dim) for PyTorch SDPA
        xq = xq.transpose(1, 2)
        xk = xk.transpose(1, 2)
        xv = xv.transpose(1, 2)
        
        # Perform scaled dot-product attention with causal mask (is_causal=True)
        # This will automatically leverage FlashAttention or Memory-Efficient Attention on GPU
        output = F.scaled_dot_product_attention(
            xq, xk, xv,
            attn_mask=None,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=True
        )
        
        # Reshape back to (B, T, dim)
        output = output.transpose(1, 2).contiguous().view(b, t, -1)
        return self.wo(output)


class TransformerBlock(nn.Module):
    """
    Decoder-only Transformer Block utilizing Pre-LN design.
    """
    def __init__(self, dim: int, n_heads: int, head_dim: int, hidden_dim: int, dropout: float = 0.0):
        super().__init__()
        self.attention = Attention(dim, n_heads, head_dim, dropout)
        self.feed_forward = FeedForward(dim, hidden_dim)
        self.attention_norm = RMSNorm(dim)
        self.ffn_norm = RMSNorm(dim)

    def forward(self, x: torch.Tensor, freqs_cis: torch.Tensor) -> torch.Tensor:
        # Pre-LN structure: normalization is applied before block operations
        # and residual connections bypass the normalized block.
        h = x + self.attention(self.attention_norm(x), freqs_cis)
        out = h + self.feed_forward(self.ffn_norm(h))
        return out


@dataclass
class ModelArgs:
    vocab_size: int = 100277   # Default tiktoken cl100k_base vocabulary
    dim: int = 256             # Transformer hidden dimension
    n_layers: int = 6          # Number of Transformer blocks
    n_heads: int = 8           # Number of Attention heads (dim must be divisible by n_heads)
    max_seq_len: int = 512     # Maximum context window length
    dropout: float = 0.1       # Dropout rate


class NanoLLM(nn.Module):
    """
    The full NanoLLM model from scratch.
    """
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.vocab_size = args.vocab_size
        self.max_seq_len = args.max_seq_len
        self.n_layers = args.n_layers
        
        self.tok_embeddings = nn.Embedding(args.vocab_size, args.dim)
        self.dropout = nn.Dropout(args.dropout)
        
        # Stack of decoder blocks
        self.layers = nn.ModuleList([
            TransformerBlock(
                dim=args.dim,
                n_heads=args.n_heads,
                head_dim=args.dim // args.n_heads,
                hidden_dim=int(8 / 3 * args.dim),  # Llama standard SwiGLU sizing
                dropout=args.dropout
            ) for _ in range(args.n_layers)
        ])
        
        self.norm = RMSNorm(args.dim)
        self.output = nn.Linear(args.dim, args.vocab_size, bias=False)
        
        # Weight tying (embeddings and output projection)
        self.tok_embeddings.weight = self.output.weight
        
        # Precompute RoPE frequencies and register as buffer (non-persistent)
        freqs_cis = precompute_freqs_cis(args.dim // args.n_heads, args.max_seq_len)
        self.register_buffer("freqs_cis", freqs_cis, persistent=False)
        
        # Weight initialization
        self.apply(self._init_weights)
        
        # Special initialization scaling for residual projections to stabilize deep networks
        for name, param in self.named_parameters():
            if name.endswith('wo.weight') or name.endswith('w2.weight'):
                torch.nn.init.normal_(param, mean=0.0, std=0.02 / math.sqrt(2 * self.n_layers))

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, tokens: torch.Tensor, targets: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        b, t = tokens.shape
        assert t <= self.max_seq_len, f"Sequence length {t} exceeds max length {self.max_seq_len}"
        
        # Embed tokens and apply dropout
        h = self.tok_embeddings(tokens)
        h = self.dropout(h)
        
        # Slice precomputed RoPE frequencies for the sequence length
        freqs_cis = self.freqs_cis[:t]
        
        # Pass through layers
        for layer in self.layers:
            h = layer(h, freqs_cis)
            
        # Final norm
        h = self.norm(h)
        
        # Linear output layer
        logits = self.output(h) # (B, T, vocab_size)
        
        loss = None
        if targets is not None:
            # Flatten predictions and targets to calculate cross entropy
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
            
        return logits, loss


# =============================================================================
# 2. DATASET & DATALOADER
# =============================================================================

class MemMapDataset(Dataset):
    """
    High-performance PyTorch Dataset that loads tokenized data from a binary file
    using numpy memory-mapping (np.memmap). Avoids loading large files to RAM.
    """
    def __init__(self, bin_path: str, seq_len: int):
        self.bin_path = bin_path
        self.seq_len = seq_len
        self.dtype = np.uint32  # Using uint32 for cl100k_base token IDs (> 65535)
        
        if not os.path.exists(bin_path):
            raise FileNotFoundError(f"Binary file '{bin_path}' not found. Run with --prepare-data first.")
            
        file_size = os.path.getsize(bin_path)
        self.num_tokens = file_size // np.dtype(self.dtype).itemsize
        
        # Memory map the file
        self.data = np.memmap(bin_path, dtype=self.dtype, mode='r')
        
    def __len__(self) -> int:
        return max(0, self.num_tokens - self.seq_len)
        
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        # Yield sequence of length seq_len + 1 (inputs x and target labels y)
        chunk = torch.from_numpy(self.data[idx : idx + self.seq_len + 1].astype(np.int64))
        x = chunk[:-1]
        y = chunk[1:]
        return x, y


def download_data(input_path: str):
    """
    Download Tiny Shakespeare dataset.
    """
    url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
    print(f"Downloading Tiny Shakespeare from {url}...")
    try:
        urllib.request.urlretrieve(url, input_path)
        print(f"Successfully downloaded Tiny Shakespeare to {input_path}")
    except Exception as e:
        print(f"Error downloading dataset: {e}")
        sys.exit(1)


def prepare_data(data_dir: str):
    """
    Download, tokenize using tiktoken cl100k_base, and save split data to bin files.
    """
    try:
        import tiktoken
    except ImportError:
        print("Error: 'tiktoken' is required for tokenization. Please install it with 'pip install tiktoken'.")
        sys.exit(1)
        
    os.makedirs(data_dir, exist_ok=True)
    input_path = os.path.join(data_dir, "input.txt")
    train_bin_path = os.path.join(data_dir, "train.bin")
    val_bin_path = os.path.join(data_dir, "val.bin")
    
    if not os.path.exists(input_path):
        download_data(input_path)
        
    print("Reading dataset...")
    with open(input_path, "r", encoding="utf-8") as f:
        data = f.read()
        
    print(f"Dataset size: {len(data):,} characters")
    
    # 90/10 Train/Validation Split
    n = len(data)
    train_data = data[:int(n * 0.9)]
    val_data = data[int(n * 0.9):]
    
    # Tokenization
    print("Loading cl100k_base tiktoken encoder...")
    enc = tiktoken.get_encoding("cl100k_base")
    
    print("Tokenizing train split...")
    train_ids = enc.encode_ordinary(train_data)
    print("Tokenizing val split...")
    val_ids = enc.encode_ordinary(val_data)
    
    print(f"Train set: {len(train_ids):,} tokens")
    print(f"Val set: {len(val_ids):,} tokens")
    
    # Save as uint32 binary arrays
    train_ids = np.array(train_ids, dtype=np.uint32)
    val_ids = np.array(val_ids, dtype=np.uint32)
    
    train_ids.tofile(train_bin_path)
    val_ids.tofile(val_bin_path)
    print(f"Saved binary datasets: '{train_bin_path}' and '{val_bin_path}'")


# =============================================================================
# 3. OPTIMIZER, SCHEDULER & EVALUATION
# =============================================================================

def configure_optimizers(model: nn.Module, weight_decay: float, learning_rate: float, betas: Tuple[float, float], device_type: str) -> torch.optim.Optimizer:
    """
    Create AdamW optimizer, applying weight decay only to 2D tensors (weights in Linear/Embeddings),
    leaving 1D tensors (RMSNorm scaling weights, biases) un-decayed.
    """
    param_dict = {pn: p for pn, p in model.named_parameters() if p.requires_grad}
    
    # Decayed vs non-decayed
    decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
    nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
    
    optim_groups = [
        {'params': decay_params, 'weight_decay': weight_decay},
        {'params': nodecay_params, 'weight_decay': 0.0}
    ]
    
    num_decay_params = sum(p.numel() for p in decay_params)
    num_nodecay_params = sum(p.numel() for p in nodecay_params)
    print(f"Optimizer: {len(decay_params)} decayed tensors ({num_decay_params:,} params)")
    print(f"Optimizer: {len(nodecay_params)} nodecay tensors ({num_nodecay_params:,} params)")
    
    # Enable fused optimizer if running on CUDA for speed improvement
    use_fused = (device_type == 'cuda') and ('fused' in torch.optim.AdamW.__init__.__code__.co_varnames)
    extra_args = dict(fused=True) if use_fused else dict()
    optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, **extra_args)
    print(f"Using fused AdamW: {use_fused}")
    
    return optimizer


def get_lr(step: int, max_steps: int, warmup_steps: int, learning_rate: float, min_lr: float) -> float:
    """
    Calculates learning rate with linear warmup and cosine decay.
    """
    # 1) Linear Warmup
    if step < warmup_steps:
        return learning_rate * step / warmup_steps
    # 2) Post-Max Step Fallback
    if step > max_steps:
        return min_lr
    # 3) Cosine Decay
    decay_ratio = (step - warmup_steps) / (max_steps - warmup_steps)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (learning_rate - min_lr)


@torch.no_grad()
def evaluate(model: nn.Module, dataloader: DataLoader, device: torch.device, eval_iters: int) -> Tuple[float, float]:
    """
    Evaluation loop to calculate average loss and perplexity.
    """
    model.eval()
    losses = []
    
    # We use AMP auto-casting if supported by device
    device_type = device.type
    ptdtype = torch.bfloat16 if (device_type == 'cuda' and torch.cuda.is_bf16_supported()) else torch.float32
    
    for i, (X, Y) in enumerate(dataloader):
        if i >= eval_iters:
            break
        X, Y = X.to(device), Y.to(device)
        
        if ptdtype in (torch.float16, torch.bfloat16):
            with torch.amp.autocast(device_type=device_type, dtype=ptdtype):
                _, loss = model(X, Y)
        else:
            _, loss = model(X, Y)
        losses.append(loss.item())
        
    model.train()
    mean_loss = sum(losses) / len(losses)
    perplexity = math.exp(mean_loss)
    return mean_loss, perplexity


# =============================================================================
# 4. TEXT GENERATION
# =============================================================================

@torch.no_grad()
def generate_text(model: nn.Module, prompt: str, max_new_tokens: int, temperature: float, top_k: int, device: torch.device) -> str:
    """
    Autoregressive generation/sampling starting from prompt text.
    """
    try:
        import tiktoken
    except ImportError:
        return "Error: 'tiktoken' required to run text generation."
        
    model.eval()
    enc = tiktoken.get_encoding("cl100k_base")
    
    # Tokenize input prompt
    tokens = enc.encode(prompt)
    x = torch.tensor(tokens, dtype=torch.long, device=device).unsqueeze(0)  # (1, T)
    
    # Generate tokens one by one
    for _ in range(max_new_tokens):
        # Crop context sequence to fit model's max seq length
        x_cond = x if x.size(1) <= model.max_seq_len else x[:, -model.max_seq_len:]
        
        # Forward pass
        logits, _ = model(x_cond)  # (1, T, vocab_size)
        
        # Take logits for only the last step and scale by temperature
        logits = logits[:, -1, :] / temperature
        
        # Optionally crop logits to top-k choices
        if top_k is not None:
            v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            logits[logits < v[:, [-1]]] = -float('Inf')
            
        probs = F.softmax(logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)
        
        # Append token to sequence
        x = torch.cat((x, next_token), dim=1)
        
    decoded_text = enc.decode(x[0].tolist())
    model.train()
    return decoded_text


# =============================================================================
# 5. MAIN TRAINING & INFERENCE PIPELINE
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Train or evaluate the Nano LLM from scratch")
    parser.add_argument("--prepare-data", action="store_true", help="Download and tokenize the dataset")
    parser.add_argument("--train", action="store_true", help="Run model training loop")
    parser.add_argument("--generate", type=str, default=None, help="Prompt string for generating text")
    parser.add_argument("--max-new-tokens", type=int, default=100, help="Tokens to generate during text generation")
    
    # Hyperparameters
    parser.add_argument("--max-steps", type=int, default=1000, help="Maximum training steps")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size per training step")
    parser.add_argument("--seq-len", type=int, default=256, help="Context window sequence length")
    parser.add_argument("--dim", type=int, default=256, help="Transformer hidden layer size")
    parser.add_argument("--layers", type=int, default=6, help="Number of transformer decoder layers")
    parser.add_argument("--heads", type=int, default=8, help="Number of attention heads")
    parser.add_argument("--learning-rate", type=float, default=6e-4, help="Peak learning rate")
    parser.add_argument("--min-lr", type=float, default=6e-5, help="Minimum learning rate")
    parser.add_argument("--weight-decay", type=float, default=0.1, help="AdamW weight decay coefficient")
    parser.add_argument("--grad-clip", type=float, default=1.0, help="Gradient norm clipping threshold")
    parser.add_argument("--warmup-iters", type=int, default=200, help="Warmup iterations")
    
    # Configuration
    parser.add_argument("--data-dir", type=str, default="data", help="Directory where data is located")
    parser.add_argument("--checkpoint", type=str, default="nano_llm_checkpoint.pt", help="Filepath for checkpointing")
    parser.add_argument("--eval-interval", type=int, default=100, help="Iterations between evaluations")
    parser.add_argument("--eval-iters", type=int, default=20, help="Batches evaluated during validation")
    
    args = parser.parse_args()
    
    # Device configuration
    device_type = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device_type)
    print(f"Running on device: {device}")
    
    # 1. Action: Prepare Data
    if args.prepare_data:
        print("Starting data preparation...")
        prepare_data(args.data_dir)
        return
        
    # 2. Setup Model args
    # Verify that dimensions divide cleanly
    if args.dim % args.heads != 0:
        raise ValueError(f"Model dimension '--dim' ({args.dim}) must be cleanly divisible by '--heads' ({args.heads})")
        
    model_args = ModelArgs(
        vocab_size=100277,
        dim=args.dim,
        n_layers=args.layers,
        n_heads=args.heads,
        max_seq_len=args.seq_len,
        dropout=0.1
    )
    
    model = NanoLLM(model_args)
    model.to(device)
    
    # Print parameter count
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model Architecture: Dim={args.dim}, Layers={args.layers}, Heads={args.heads}, Context={args.seq_len}")
    print(f"Total Model Parameters: {num_params / 1e6:.2f}M (Shared Embeddings)")
    
    # 3. Action: Generate Text (Using checkpoint if exists, otherwise untrained model)
    if args.generate is not None:
        if os.path.exists(args.checkpoint):
            print(f"Loading checkpoint from '{args.checkpoint}' for generation...")
            checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
            state_dict = checkpoint['model_state_dict']
            if 'freqs_cis' in state_dict:
                state_dict.pop('freqs_cis')
            model.load_state_dict(state_dict)
        else:
            print("Warning: No checkpoint found. Generating text from randomly initialized model...")
            
        generated = generate_text(
            model=model,
            prompt=args.generate,
            max_new_tokens=args.max_new_tokens,
            temperature=0.8,
            top_k=40,
            device=device
        )
        print("\nGenerated Output:")
        print("-" * 50)
        sys.stdout.flush()
        try:
            print(generated)
        except UnicodeEncodeError:
            sys.stdout.buffer.write(generated.encode('utf-8', errors='replace'))
            sys.stdout.buffer.flush()
            sys.stdout.write('\n')
            sys.stdout.flush()
        print("-" * 50)
        return
        
    # 4. Action: Train Model
    if args.train:
        train_bin = os.path.join(args.data_dir, "train.bin")
        val_bin = os.path.join(args.data_dir, "val.bin")
        
        # Initialize datasets
        try:
            train_dataset = MemMapDataset(train_bin, args.seq_len)
            val_dataset = MemMapDataset(val_bin, args.seq_len)
        except FileNotFoundError as e:
            print(f"Error: {e}")
            print("Please run data preparation first: 'python train_nano_llm.py --prepare-data'")
            return
            
        # DataLoaders (using num_workers=0 to avoid serialization/IPC overhead on Windows)
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
        
        # Optimizer Setup
        optimizer = configure_optimizers(
            model=model,
            weight_decay=args.weight_decay,
            learning_rate=args.learning_rate,
            betas=(0.9, 0.95),
            device_type=device_type
        )
        
        # Setup AMP precision
        ptdtype = torch.bfloat16 if (device_type == 'cuda' and torch.cuda.is_bf16_supported()) else torch.float32
        print(f"Using Mixed Precision Autocast Dtype: {ptdtype}")
        
        # Load from checkpoint if exists
        start_step = 0
        best_val_loss = float('inf')
        if os.path.exists(args.checkpoint):
            print(f"Resuming training from checkpoint '{args.checkpoint}'...")
            checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
            state_dict = checkpoint['model_state_dict']
            if 'freqs_cis' in state_dict:
                state_dict.pop('freqs_cis')
            model.load_state_dict(state_dict)
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            start_step = checkpoint['step'] + 1
            best_val_loss = checkpoint.get('val_loss', float('inf'))
            print(f"Resuming at step {start_step}")
            
        # Convert DataLoader to cyclic iterators for training steps
        train_iter = iter(train_loader)
        
        model.train()
        t0 = time.time()
        
        print(f"Starting training loop from step {start_step} to {args.max_steps}...")
        for step in range(start_step, args.max_steps):
            # 1. Update Learning Rate
            lr = get_lr(step, args.max_steps, args.warmup_iters, args.learning_rate, args.min_lr)
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
                
            # 2. Periodically run validation evaluation
            if step % args.eval_interval == 0 or step == args.max_steps - 1:
                val_loss, val_perplexity = evaluate(model, val_loader, device, args.eval_iters)
                print(f"\n[Step {step}] VAL EVAL: loss={val_loss:.4f} | perplexity={val_perplexity:.4f}")
                
                # Save checkpoint if it's the best so far
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    print(f"Saving checkpoint to '{args.checkpoint}' (New Best Val Loss: {val_loss:.4f})...")
                    torch.save({
                        'step': step,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'val_loss': val_loss,
                        'args': model_args
                    }, args.checkpoint)
                    
            # 3. Fetch training batch
            try:
                X, Y = next(train_iter)
            except StopIteration:
                # Re-initialize iterator if it reaches end of dataset
                train_iter = iter(train_loader)
                X, Y = next(train_iter)
                
            X, Y = X.to(device), Y.to(device)
            
            # 4. Forward pass under AMP Autocast
            if ptdtype in (torch.float16, torch.bfloat16):
                with torch.amp.autocast(device_type=device_type, dtype=ptdtype):
                    _, loss = model(X, Y)
            else:
                _, loss = model(X, Y)
                
            # 5. Backward pass
            loss.backward()
            
            # 6. Gradient clipping
            if args.grad_clip > 0.0:
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            else:
                grad_norm = 0.0
                
            # 7. Optimizer step
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            
            # 8. Step timing and logging
            t1 = time.time()
            dt = t1 - t0
            t0 = t1
            
            # Calculate throughput: tokens processed per second
            tokens_per_sec = (args.batch_size * args.seq_len) / dt
            
            if step % 10 == 0:
                print(f"Step {step}/{args.max_steps} | loss: {loss.item():.4f} | lr: {lr:.2e} | grad_norm: {grad_norm:.4f} | step_time: {dt*1000:.1f}ms | {tokens_per_sec:.0f} tokens/s")
                
        print("Training completed successfully!")
        
    if not args.train and not args.prepare_data and args.generate is None:
        parser.print_help()


if __name__ == "__main__":
    main()
