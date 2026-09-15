from transformers import AutoTokenizer

class Tokenizer:
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained("gpt2")

    