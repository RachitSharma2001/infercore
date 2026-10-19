import numpy as np 
from models.tiny_gpt2.tinygpt2 import GPT2Model
import torch
import time

import random
import sys

from kvcache2 import Pool

class Scheduler:
    def __init__(self, tokenizer, batch_size = 3):
        self.requests = []

        self.gpt2_kpool = Pool(12, 110, 12, 64)
        self.gpt2_vpool = Pool(12, 110, 12, 64)

        self.model = GPT2Model(self.gpt2_kpool, self.gpt2_vpool, model_name="gpt2")
        self.tokenizer = tokenizer
        self.max_response_tokens = 15
        self.batch_size = batch_size
        self.iter = 0

    def add(self, request):
        req = SchedulerRequest(request, self.gpt2_kpool, self.gpt2_vpool)
        self.requests.append(req)

    def step(self):
        requests = self.requests[:self.batch_size]
        # print("requests:", len(requests))
        token_ids = torch.cat(
            [req.get_token_ids() for req in requests],
            dim=0,
        )
        position_ids = torch.cat(
            [
                req.get_position_ids()
                for req in requests
            ],
            dim=0,
        )
        request_ids = torch.cat(
            [
                torch.full(
                    (len(req.get_token_ids()),),
                    req.request_id,
                    device=req.get_token_ids().device,
                    dtype=torch.long,
                )
                for i, req in enumerate(requests)
            ],
            dim=0,
        )
        read_slots = torch.cat(
            [
                torch.tensor(req.k_read_slots)
                for i, req in enumerate(requests)
            ], 
            dim =0,
        )
        read_offsets = torch.tensor(
            [
                req.k_read_offsets
                for i, req in enumerate(requests)
            ]
        )
        write_slots = torch.cat(
            [
                torch.tensor(req.k_write_slots)
                for i, req in enumerate(requests)
            ],
            dim=0,
        )
        print("========================================")
        print("read slots", read_slots)
        print("read offsets", read_offsets)
        print("write slots", write_slots)

        self.iter += 1

        # if self.iter == 3:
        #     print("no doing")
        #     time.sleep(1000)
        # print(token_ids)
        # print(position_ids)
        # print(request_ids)
        # logits are packed: [total_tokens, vocab], one row per input token
        # across every request concatenated together (no batch dimension).
        logits = self.model.forward(token_ids, request_ids, position_ids, read_slots, read_offsets, write_slots)
        probs = torch.softmax(logits, dim=-1)
        next_tokens = torch.argmax(probs, dim=-1)
        requests_to_del = []
        offset = 0
        for req in requests:
            request_len = len(req.get_token_ids())
            # last packed row belonging to this request predicts its next token
            next_token = next_tokens[offset + request_len - 1]
            if next_token == self.tokenizer.eos_token_id or req.num_response_tokens == self.max_response_tokens:
                req.request.future.set_result(req.current_token_ids)
                requests_to_del.append(req)
            else:
                req.current_token_ids = torch.cat((req.current_token_ids, next_token.view(1)))
                req.num_response_tokens += 1
            offset += request_len
            req.set_to_decode()
            req.set_slots(self.gpt2_kpool, self.gpt2_vpool)
        for req in requests_to_del:
            req.free_slots(self.gpt2_kpool, self.gpt2_vpool)
            self.requests.remove(req)

    def process(self):
        while True:
            if not self.requests:
                time.sleep(0.01)
                continue
            self.step()
            time.sleep(0.5)

class SchedulerRequest:
    def __init__(self, request, kpool, vpool):
        self.request = request 
        self.current_token_ids = request.token_ids
        self.num_response_tokens = 0
        self.is_prefill = True 
        self.request_id = random.randint(1, sys.maxsize)
        self.k_read_slots = []
        self.k_read_offsets = 0
        self.k_write_slots = []
        self.v_read_slots = []
        self.v_read_offsets = 0
        self.v_write_slots = []
        self.set_slots(kpool, vpool)

    def free_slots(self, kpool, vpool):
        kpool.free_these_slots(self.k_read_slots + self.k_write_slots)
        vpool.free_these_slots(self.v_read_slots + self.v_write_slots)
        self.k_read_slots = []
        self.k_write_slots = []
        self.v_read_slots = []
        self.v_write_slots = []

    def set_slots(self, kpool, vpool):
        if self.is_prefill:
            self.k_read_slots = []
            self.k_read_offsets = 0
            self.k_write_slots = kpool.request_slots(len(self.current_token_ids))
            self.v_read_slots = []
            self.v_read_offsets = 0
            self.v_write_slots = vpool.request_slots(len(self.current_token_ids))
        else:
            self.k_read_slots.extend(self.k_write_slots)
            self.k_read_offsets = len(self.k_read_slots)
            self.v_read_slots.extend(self.v_write_slots)
            self.v_read_offsets = len(self.v_read_slots)
            self.k_write_slots = kpool.request_slots(1)
            self.v_write_slots = vpool.request_slots(1)            

    def get_request_id(self):
        return self.request_id

    def set_to_decode(self):
        self.is_prefill = False

    def get_token_ids(self):
        if self.is_prefill:
            return self.current_token_ids
        return self.current_token_ids[-1:]

    def get_position_ids(self):
        if self.is_prefill:
            return  torch.arange(len(self.get_token_ids()), device=self.get_token_ids().device)
        return torch.tensor([len(self.current_token_ids)-1])

# from request import Request 
# kpool = Pool(3, 5, 10, 60)
# vpool = Pool(3, 5, 10, 60)
# from transformers import AutoTokenizer
# tokenizer = AutoTokenizer.from_pretrained("gpt2")
# req = Request(tokenizer("Where am i", return_tensors="pt")['input_ids'][0])
# sReq = SchedulerRequest(req)
# sReq.set_slots(kpool, vpool)
# print(sReq.k_read_slots)
# print(sReq.k_write_slots)
# print(sReq.v_read_slots)
# print(sReq.v_write_slots)
# sReq.set_to_decode()
# sReq.set_slots(kpool, vpool)
# print("-------------------------")
# print(sReq.k_read_slots)
# print(sReq.k_write_slots)
# print(sReq.v_read_slots)
# print(sReq.v_write_slots)
# sReq.free_slots(kpool, vpool)
# print(sReq.k_read_slots)
# print(sReq.k_write_slots)
# print(sReq.v_read_slots)
# print(sReq.v_write_slots)
# print(kpool.free_slots)
# print(vpool.free_slots)