import torch
import torch.nn.functional as F

from config import GPTConfig
from model import GPT
from tokenizer import Tokenizer


device = "cuda" if torch.cuda.is_available() else "cpu"
tokenizer = Tokenizer()

def load_model(checkpoint_path):
    ckpt = torch.load(checkpoint_path, map_location=device)

    cfg = GPTConfig(**ckpt["config"])
    model = GPT(cfg).to(device)
    model.load_state_dict(ckpt["model"])

    model.eval()
    return model, cfg


# -------------------
# Top-k filtering
# -------------------
def top_k_logits(logits, k):
    v, _ = torch.topk(logits, k)
    cutoff = v[:, -1].unsqueeze(-1)
    return torch.where(
        logits < cutoff,
        torch.full_like(logits, -float("inf")),
        logits
    )


# -------------------
# Generation
# -------------------
@torch.no_grad()
def generate(model, idx, cfg, max_new_tokens=100, temperature=1.0, top_k=50):

    for _ in range(max_new_tokens):

        idx_cond = idx[:, -cfg.seq_len:]  # crop context

        logits = model(idx_cond)          # [B, T, vocab]
        logits = logits[:, -1, :]         # last token

        logits = logits / temperature

        if top_k is not None:
            logits = top_k_logits(logits, top_k)

        probs = F.softmax(logits, dim=-1)

        next_token = torch.multinomial(probs, num_samples=1)

        idx = torch.cat((idx, next_token), dim=1)

    return idx

def main():
    checkpoint_path = "checkpoints/gpt_step_5000.pt"

    model, cfg = load_model(checkpoint_path)

    prompt = "hello world"

    # ✅ Use your tokenizer
    idx = torch.tensor(
        [tokenizer.encode(prompt)],
        dtype=torch.long,
        device=device
    )

    out = generate(
        model,
        idx,
        cfg,
        max_new_tokens=100,
        temperature=0.8,
        top_k=50
    )

    print("\n--- Generated Text ---\n")
    print(tokenizer.decode(out[0].tolist()))


if __name__ == "__main__":
    main()