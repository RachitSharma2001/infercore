import numpy as np 
from models.tiny_gpt2.tinygpt2 import GPT2Model
import torch

class Scheduler:
    def __init__(self, tokenizer):
        self.requests = []
        self.model = GPT2Model(model_name="gpt2")
        self.tokenizer = tokenizer
        self.max_response_tokens = 15

    def add(self, token_ids):
        req = SchedulerRequest(token_ids)
        self.requests.append(req)

    def step(self):
        req = self.requests[0]
        print("input: ", req.current_token_ids)
        logits = self.model.forward(req.current_token_ids.unsqueeze(0))
        logits = logits[:, -1, :]
        # todo: enable configuraiton of next token selection. right now its always greedy
        probs = torch.softmax(logits, dim=-1)
        next_token = torch.argmax(probs, dim=-1)
        if next_token.item() == self.tokenizer.eos_token_id or req.num_response_tokens == self.max_response_tokens:
            print(self.tokenizer.batch_decode(req.current_token_ids))
            self.requests = self.requests[1:]
            return 
        req.current_token_ids = torch.cat((req.current_token_ids, next_token))
        req.num_response_tokens += 1

class SchedulerRequest:
    def __init__(self, prompt_token_ids):
        self.current_token_ids = prompt_token_ids
        self.num_response_tokens = 0
