import numpy as np 
from models.tiny_gpt2.tinygpt2 import GPT2Model
import torch
import time


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
        # print("requests:", len(requests))
        token_ids = torch.cat(
            [req.current_token_ids for req in requests],
            dim=0,
        )
        position_ids = torch.cat(
            [
                torch.arange(len(req.current_token_ids), device=req.current_token_ids.device)
                for req in requests
            ],
            dim=0,
        )
        request_ids = torch.cat(
            [
                torch.full(
                    (len(req.current_token_ids),),
                    i,
                    device=req.current_token_ids.device,
                    dtype=torch.long,
                )
                for i, req in enumerate(requests)
            ],
            dim=0,
        )
        # print(token_ids, position_ids, request_ids)
        # logits are packed: [total_tokens, vocab], one row per input token
        # across every request concatenated together (no batch dimension).
        logits = self.model.forward(token_ids, request_ids, position_ids)
        probs = torch.softmax(logits, dim=-1)
        next_tokens = torch.argmax(probs, dim=-1)
        requests_to_del = []
        offset = 0
        for req in requests:
            request_len = len(req.current_token_ids)
            # last packed row belonging to this request predicts its next token
            next_token = next_tokens[offset + request_len - 1]
            if next_token == self.tokenizer.eos_token_id or req.num_response_tokens == self.max_response_tokens:
                req.request.future.set_result(req.current_token_ids)
                requests_to_del.append(req)
            else:
                req.current_token_ids = torch.cat((req.current_token_ids, next_token.view(1)))
                req.num_response_tokens += 1
            offset += request_len
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
