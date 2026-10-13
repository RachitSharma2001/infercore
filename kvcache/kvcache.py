
import torch

class KVCache:

    def __init__(self):
        self.k_cache = {}
        self.v_cache = {}

    def set_k_cache(self, k_cache):
        self.k_cache = k_cache

    def set_v_cache(self, v_cache):
        self.v_cache = v_cache

    def prepend_k(self, request_ids, k):
        return self.prepend_generic(request_ids, self.k_cache, k)

    def prepend_v(self, request_ids, v):
        return self.prepend_generic(request_ids, self.v_cache, v)

    def prepend_generic(self, request_ids, cache, vec):
        unique_req_ids, counts = torch.unique_consecutive(request_ids, return_counts=True)
        ksplit = torch.split(vec, counts.tolist())
        new_k = []
        block_nums = []
        req_len = []

        for k_piece, unique_req_id in zip(ksplit, unique_req_ids):
            if unique_req_id.item() not in cache:
                new_k.append(k_piece)
                cache[unique_req_id.item()] = k_piece
                block_nums.extend([1 for _ in range(len(k_piece))])
                req_len.append(len(k_piece))
                continue 
            block_nums.append(len(cache[unique_req_id.item()]) + 1)
            req_len.append(len(cache[unique_req_id.item()]) + 1)
            new_k.append(cache[unique_req_id.item()])
            cache[unique_req_id.item()] = torch.cat((cache[unique_req_id.item()], k_piece))
            new_k.append(k_piece)

        p = torch.cumsum(torch.tensor(block_nums), dim=0) - 1
        new_req_ids = torch.repeat_interleave(unique_req_ids, repeats=torch.tensor(req_len))
        return torch.cat(new_k), p, new_req_ids