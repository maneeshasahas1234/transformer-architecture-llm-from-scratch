from dataclasses import dataclass

@dataclass
class GPTConfig:
    # Model architecture
    vocab_size: int   = 50257   # GPT-2 vocab size (tiktoken)
    seq_len:    int   = 256     # max context length
    d_model:    int   = 384     # embedding dimension
    n_heads:    int   = 6       # attention heads
    n_layers:   int   = 6       # transformer blocks
    d_ff:       int   = 1536    # FFN inner dim (4 × d_model)
    dropout:    float = 0.1

    # Training
    batch_size:    int   = 32
    lr:            float = 3e-4
    max_iters:     int   = 5000
    eval_interval: int   = 50
    eval_iters:    int   = 50
    device: str = (
        'mps' if __import__('torch').backends.mps.is_available()
        else 'cuda' if __import__('torch').cuda.is_available()
        else 'cpu'
    )

# d_model must be divisible by n_heads
# 384 / 6 = 64 dims per head  ✓


