import numpy as np 
from models.tiny_gpt2.tinygpt2 import GPT2Model
import torch
import time
from torch.nn.utils.rnn import pad_sequence


class Scheduler:
    def __init__(self, tokenizer, batch_size = 3):
        self.requests = []
        self.model = GPT2Model(model_name="gpt2")
        self.tokenizer = tokenizer
        self.max_response_tokens = 15
        self.batch_size = batch_size

    def add(self, request):
        req = SchedulerRequest(request)
        self.requests.append(req)

    def step(self):
        requests = self.requests[:self.batch_size]
        print("requests:", len(requests))
        token_ids = pad_sequence(
            [req.current_token_ids for req in requests],
            batch_first = True,
            padding_value = 0,
        )
        logits = self.model.forward(token_ids)
        probs = torch.softmax(logits, dim=-1)
        next_tokens = torch.argmax(probs, dim=-1)
        requests_to_del = []
        for i, token_seq in enumerate(next_tokens):
            req = self.requests[i]
            next_token = token_seq[len(req.current_token_ids)-1]
            if next_token == self.tokenizer.eos_token_id or req.num_response_tokens == self.max_response_tokens:
                req.request.future.set_result(req.current_token_ids)
                requests_to_del.append(req)
            else:
                req.current_token_ids = torch.cat((req.current_token_ids, token_seq[len(req.current_token_ids)-1:len(req.current_token_ids)]))
                req.num_response_tokens += 1
        for req in requests_to_del:
            self.requests.remove(req)

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
