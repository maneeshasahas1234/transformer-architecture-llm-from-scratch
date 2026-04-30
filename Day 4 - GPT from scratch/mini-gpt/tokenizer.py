import tiktoken

class Tokenizer:
    def __init__(self):
        # Load GPT-2's BPE tokenizer
        # vocab_size = 50,257 tokens
        self.enc = tiktoken.get_encoding("gpt2")
        self.vocab_size = self.enc.n_vocab

    def encode(self, text):
        """Text → list of integer token IDs"""
        return self.enc.encode(text)

    def decode(self, ids):
        """List of token IDs → text"""
        return self.enc.decode(ids)

