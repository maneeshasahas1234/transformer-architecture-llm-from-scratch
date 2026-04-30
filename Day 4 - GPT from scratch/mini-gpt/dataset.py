import torch
from torch.utils.data import Dataset, DataLoader
from tokenizer import Tokenizer


class TextDataset(Dataset):
    def __init__(self, text: str, seq_len: int):
        self.seq_len = seq_len
        tok = Tokenizer()

        # Encode entire text into one long list of token IDs
        self.tokens = torch.tensor(tok.encode(text), dtype=torch.long)
        print(f"Dataset: {len(self.tokens):,} tokens")

    def __len__(self):
        # Each sample is seq_len tokens
        # We need seq_len+1 tokens (input + target)
        return len(self.tokens) - self.seq_len

    def __getitem__(self, idx):
        # Input  : tokens[idx   : idx+seq_len]
        # Target : tokens[idx+1 : idx+seq_len+1]  ← shifted by 1
        x = self.tokens[idx: idx + self.seq_len]
        y = self.tokens[idx + 1: idx + self.seq_len + 1]
        return x, y


def get_data(seq_len: int, batch_size: int):
    with open("data/data.txt", "r") as f:
        text = f.read()

    # 90/10 train/val split
    n = int(0.9 * len(text))
    train_data = TextDataset(text[:n], seq_len)
    val_data = TextDataset(text[n:], seq_len)

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader
