import numpy as np 
from models.tiny_gpt2.tinygpt2 import GPT2Model
import torch
import time

class Scheduler:
    def __init__(self, tokenizer):
        self.requests = []
        self.model = GPT2Model(model_name="gpt2")
        self.tokenizer = tokenizer
        self.max_response_tokens = 15

    def add(self, request):
        req = SchedulerRequest(request)
        self.requests.append(req)

    def step(self):
        req = self.requests[0]
        logits = self.model.forward(req.current_token_ids.unsqueeze(0))
        logits = logits[:, -1, :]
        # todo: enable configuraiton of next token selection. right now its always greedy
        probs = torch.softmax(logits, dim=-1)
        next_token = torch.argmax(probs, dim=-1)
        if next_token.item() == self.tokenizer.eos_token_id or req.num_response_tokens == self.max_response_tokens:
            # req.request.future.set_result(self.tokenizer.batch_decode(req.current_token_ids))
            req.request.future.set_result(req.current_token_ids)
            self.requests = self.requests[1:]
            return 
        req.current_token_ids = torch.cat((req.current_token_ids, next_token))
        req.num_response_tokens += 1

    def process(self):
        while True:
            if not self.requests:
                time.sleep(0.01)
                continue
            self.step()
            time.sleep(0.5)

class SchedulerRequest:
    def __init__(self, request):
        self.request = request 
        self.current_token_ids = request.token_ids
        self.num_response_tokens = 0
