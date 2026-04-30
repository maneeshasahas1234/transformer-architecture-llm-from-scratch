import torch
import os
from config import GPTConfig
from model import GPT
from dataset import get_data
import torch.nn as nn
import time

cfg = GPTConfig(
    vocab_size = 50257,
    seq_len    = 128,
    d_model    = 128,
    n_heads    = 8,
    n_layers   = 4,
    d_ff       = 512,
    dropout    = 0.1,
    batch_size = 16,
    lr         = 3e-3,
    max_iters  = 10000,
)


train_loader,val_loader = get_data(cfg.seq_len,cfg.batch_size)
model = GPT(cfg).to(cfg.device)


optimizer = torch.optim.AdamW(
    model.parameters(),
    lr           = cfg.lr,
    weight_decay = 0.1,    # penalizes large weights
    betas        = (0.9, 0.95),
)

total_params = sum(p.numel() for p in model.parameters())
print(f"Total parameters: {total_params:,}")

trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Trainable parameters: {trainable_params:,}")

criterion = nn.CrossEntropyLoss()


def train():
    print(f"Training on:{cfg.device}" )
    model.train()
    step = 0
    start_time = time.time()
    for epoch in range(cfg.max_iters):
        for x, y in train_loader:
            x = x.to(cfg.device)  # [B, T]
            y = y.to(cfg.device)  # [B, T]
            logits = model(x)  # [B, T, vocab_size]
            loss = criterion(
                logits.view(-1, logits.size(-1)),
                y.view(-1)
            )
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            step += 1
            if step % 10 == 0:
                elapsed = time.time() - start_time
                print(f"Step {step} | Loss: {loss.item():.4f} | Time: {elapsed:.2f}s")
            if step % 500 == 0:
                save_checkpoint(step)
            if step >= cfg.max_iters:
                print("Training complete.")
                return

def save_checkpoint(step):
    os.makedirs("checkpoints", exist_ok=True)
    path = f"checkpoints/gpt_step_{step}.pt"
    torch.save({
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "step": step,
        "config": cfg.__dict__,
    }, path)
    print(f"Saved checkpoint: {path}")


if __name__ == "__main__":
    train()