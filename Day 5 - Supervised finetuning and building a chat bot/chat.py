import torch
import torch.nn.functional as F

from config import GPTConfig
from model import GPT
from tokenizer import Tokenizer

device = "cuda" if torch.cuda.is_available() else "cpu"
tokenizer = Tokenizer()


# -------------------
# Load model
# -------------------
def load_model(ckpt_path):
    ckpt = torch.load(ckpt_path, map_location=device)

    cfg = GPTConfig(**ckpt["config"])
    model = GPT(cfg).to(device)
    model.load_state_dict(ckpt["model"])

    model.eval()
    return model, cfg


# -------------------
# Generation
# -------------------
@torch.no_grad()
def generate(model, idx, cfg, max_new_tokens=100, temperature=0.8, top_k=50):

    for _ in range(max_new_tokens):

        idx_cond = idx[:, -cfg.seq_len:]

        logits = model(idx_cond)
        logits = logits[:, -1, :]

        logits = logits / temperature

        if top_k is not None:
            v, _ = torch.topk(logits, top_k)
            cutoff = v[:, -1].unsqueeze(-1)
            logits = torch.where(logits < cutoff, torch.full_like(logits, -float("inf")), logits)

        probs = F.softmax(logits, dim=-1)

        next_token = torch.multinomial(probs, num_samples=1)

        idx = torch.cat((idx, next_token), dim=1)

    return idx


# -------------------
# Chat loop
# -------------------
def chat():
    model, cfg = load_model("checkpoints/sft_step_600.pt")

    print("Chatbot ready! Type 'exit' to stop.\n")

    while True:
        user_input = input("Ask anything: ")

        if user_input.lower() == "exit":
            break

        # format prompt (VERY IMPORTANT)
        prompt = f"User: {user_input}\nAssistant:"

        idx = torch.tensor(
            [tokenizer.encode(prompt)],
            dtype=torch.long,
            device=device
        )

        out = generate(model, idx, cfg, max_new_tokens=100, temperature=0.8)

        text = tokenizer.decode(out[0].tolist())

        # extract only assistant response
        response = text.split("Assistant:")[-1].strip()

        print(f"Bot: {response}\n")


if __name__ == "__main__":
    chat()