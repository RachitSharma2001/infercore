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
        for k_piece, unique_req_id in zip(ksplit, unique_req_ids):
            if unique_req_id.item() not in cache:
                new_k.append(k_piece)
                cache[unique_req_id.item()] = k_piece
                continue 
            new_k.append(cache[unique_req_id.item()])
            cache[unique_req_id.item()] = torch.cat((cache[unique_req_id.item()], k_piece))
            new_k.append(k_piece)
        return torch.cat(new_k)

request_ids = torch.tensor([0, 1, 1, 1, 1, 2])

h = 4

k = torch.randn(len(request_ids), h)

k_cache = {
    0: torch.ones(3, h),
    2: torch.ones(5, h)
}

v_cache = {
    0: torch.ones(3, h),
    2: torch.ones(5, h)
}

v = torch.randn(len(request_ids), h)


kv_cache = KVCache()
kv_cache.set_k_cache(k_cache)
kv_cache.set_v_cache(v_cache)

print(kv_cache.prepend_k(request_ids, k))
# print(kv_cache.prepend_v(request_ids, v))

print(kv_cache.k_cache)
# print(kv_cache.v_cache)

print("------------------------------------------")

request_ids = torch.tensor([0, 1, 2])
k = torch.randn(len(request_ids), h)
v = torch.randn(len(request_ids), h)

print(kv_cache.prepend_k(request_ids, k))
# print(kv_cache.prepend_v(request_ids, v))

print(kv_cache.k_cache)
# print(kv_cache.v_cache)