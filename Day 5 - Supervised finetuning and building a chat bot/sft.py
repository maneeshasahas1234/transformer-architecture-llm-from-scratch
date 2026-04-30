import torch
import torch.nn.functional as F
import json
import os
import time

from torch.utils.data import Dataset, DataLoader

from config import GPTConfig
from model import GPT
from tokenizer import Tokenizer


class SFTDataset(Dataset):
    def __init__(self, path, tokenizer, seq_len):
        self.data = []
        self.tokenizer = tokenizer
        self.seq_len = seq_len

        with open(path, "r") as f:
            for line in f:
                self.data.append(json.loads(line))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        prompt_ids = self.tokenizer.encode(item["prompt"])
        response_ids = self.tokenizer.encode(item["response"])

        tokens = prompt_ids + response_ids

        tokens = tokens[:self.seq_len + 1]

        pad_id = 0
        pad_len = (self.seq_len + 1) - len(tokens)
        tokens = tokens + [pad_id] * pad_len

        x = tokens[:-1]
        y = tokens[1:]

        labels = y.copy()

        prompt_len = min(len(prompt_ids), self.seq_len)
        mask_len = max(0, prompt_len - 1)
        labels[:mask_len] = [-100] * mask_len

        real_len = min(len(prompt_ids) + len(response_ids), self.seq_len + 1) - 1
        labels[real_len:] = [-100] * (self.seq_len - real_len)

        return (
            torch.tensor(x, dtype=torch.long),
            torch.tensor(labels, dtype=torch.long)
        )


def get_loader(path, tokenizer, seq_len, batch_size):
    dataset = SFTDataset(path, tokenizer, seq_len)

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True
    )


cfg = GPTConfig(
    vocab_size = 50257,
    seq_len    = 128,
    d_model    = 128,
    n_heads    = 8,
    n_layers   = 4,
    d_ff       = 512,
    dropout    = 0.1,
    batch_size = 16,
    lr         = 8e-3,
    max_iters  = 10000,
)

device = cfg.device

tokenizer = Tokenizer()

model = GPT(cfg).to(device)

ckpt = torch.load("checkpoints/gpt_step_6000.pt", map_location=device)
model.load_state_dict(ckpt["model"])
print("Loaded pretrained model")

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=cfg.lr,
    weight_decay=0.1,
    betas=(0.9, 0.95),
)

loader = get_loader(
    "sft_data.jsonl",
    tokenizer,
    cfg.seq_len,
    cfg.batch_size
)


def train():
    model.train()
    step = 0
    start_time = time.time()

    for epoch in range(1000):
        for x, y in loader:

            x = x.to(device)
            y = y.to(device)

            logits = model(x)

            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                y.view(-1),
                ignore_index=-100
            )

            optimizer.zero_grad()
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            optimizer.step()

            step += 1

            if step % 10 == 0:
                elapsed = time.time() - start_time
                print(f"Step {step} | Loss: {loss.item():.4f} | Time: {elapsed:.2f}s")

            if step % 200 == 0:
                save_checkpoint(step)


def save_checkpoint(step):
    os.makedirs("checkpoints", exist_ok=True)

    torch.save({
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "step": step,
        "config": cfg.__dict__,
    }, f"checkpoints/sft_step_{step}.pt")


if __name__ == "__main__":
    train()