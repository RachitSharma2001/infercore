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
        ind = 0
        p = torch.tensor([])
        new_req_ids = torch.tensor([])
        for k_piece, unique_req_id in zip(ksplit, unique_req_ids):
            if unique_req_id.item() not in cache:
                p = torch.cat((p, torch.arange(len(k_piece))+ind))
                ind += len(k_piece)
                new_k.append(k_piece)
                cache[unique_req_id.item()] = k_piece
                new_req_ids = torch.cat((new_req_ids, torch.tensor([unique_req_id for _ in range(len(k_piece))])))
                continue 
            new_req_ids = torch.cat((new_req_ids, torch.tensor([unique_req_id for _ in range(len(cache[unique_req_id.item()])+1)])))
            ind += len(cache[unique_req_id.item()]) + 1
            p = torch.cat((p, torch.tensor([ind-1])))
            new_k.append(cache[unique_req_id.item()])
            cache[unique_req_id.item()] = torch.cat((cache[unique_req_id.item()], k_piece))
            new_k.append(k_piece)
        return torch.cat(new_k), p, new_req_ids


class KVCache2:

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


############# Test Case one ############# 
    
request_ids = torch.tensor([0, 1, 2])

h = 4

k = torch.randn(len(request_ids), h)

k_cache = {
    0: torch.ones(3, h),
    1: torch.zeros(5, h),
    2: torch.ones(2, h),
}

v_cache = {
    0: torch.ones(3, h),
    1: torch.zeros(5, h),
    2: torch.ones(2, h),
}

k_cache2 = {
    0: k_cache[0],
    1: k_cache[1],
    2: k_cache[2],
}

v_cache2 = {
    0: v_cache[0],
    1: v_cache[1],
    2: v_cache[2],
}

v = torch.randn(len(request_ids), h)


kv_cache = KVCache()
kv_cache.set_k_cache(k_cache)
kv_cache.set_v_cache(v_cache)

kv_cache2 = KVCache2()
kv_cache2.set_k_cache(k_cache2)
kv_cache2.set_v_cache(v_cache2)

new_k, p, new_req_ids = kv_cache.prepend_k(request_ids, k)
new_k2, p2, new_req_ids2 = kv_cache2.prepend_k(request_ids, k)

print(new_k == new_k2)
print(p == p2)
print(new_req_ids == new_req_ids2)
print(p)


############# Test Case two ############# 
'''
request_ids = torch.tensor([0, 0, 1, 1, 1, 1, 2, 2, 2])

h = 4

k = torch.randn(len(request_ids), h)

k_cache = {
    # 0: torch.ones(3, h),
    # 1: torch.zeros(5, h),
    # 2: torch.ones(2, h),
}

v_cache = {
    # 0: torch.ones(3, h),
    # 1: torch.zeros(5, h),
    # 2: torch.ones(2, h),
}

k_cache2 = {
    # 0: k_cache[0],
    # 1: k_cache[1],
    # 2: k_cache[2],
}

v_cache2 = {
    # 0: v_cache[0],
    # 1: v_cache[1],
    # 2: v_cache[2],
}

v = torch.randn(len(request_ids), h)


kv_cache = KVCache()
kv_cache.set_k_cache(k_cache)
kv_cache.set_v_cache(v_cache)

kv_cache2 = KVCache2()
kv_cache2.set_k_cache(k_cache2)
kv_cache2.set_v_cache(v_cache2)

new_k, p, new_req_ids = kv_cache.prepend_k(request_ids, k)
new_k2, p2, new_req_ids2 = kv_cache2.prepend_k(request_ids, k)

print(new_k == new_k2)
print(p == p2)
print(new_req_ids == new_req_ids2)
print(p)
'''

# new_v, _, _ = kv_cache.prepend_v(request_ids, v)

# key_ind = torch.arange(len(new_k))
# causal_mask = key_ind < p.reshape(len(p), 1)

# # print(causal_mask)

# same_req_mask = request_ids.reshape(len(request_ids), 1) == new_req_ids


# print(new_k)
# attention_mask = causal_mask & same_req_mask

# print("p: ", p)
# print("request id: ", new_req_ids)
# print(attention_mask)

################################################################################

'''
request_ids = torch.tensor([0, 0, 0, 1, 1, 2, 2, 2])

h = 4

k = torch.randn(len(request_ids), h)

k_cache = {
    # 0: torch.ones(3, h),
    # 2: torch.ones(5, h),
    # 3: torch.zeros(2, h),
    # 5: torch.ones(3, h)
}

v_cache = {
    # 0: torch.ones(3, h),
    # 2: torch.ones(5, h),
    # 3: torch.zeros(2, h),
    # 5: torch.ones(3, h)
}

v = torch.randn(len(request_ids), h)


kv_cache = KVCache()
kv_cache.set_k_cache(k_cache)
kv_cache.set_v_cache(v_cache)

new_k, p, new_req_ids = kv_cache.prepend_k(request_ids, k)

new_v, _, _ = kv_cache.prepend_v(request_ids, v)

key_ind = torch.arange(len(new_k))
causal_mask = key_ind < p.reshape(len(p), 1)

# print(causal_mask)

same_req_mask = request_ids.reshape(len(request_ids), 1) == new_req_ids


print(new_k)
attention_mask = causal_mask & same_req_mask

print("p: ", p)
print("request id: ", new_req_ids)
print(attention_mask)
'''
################################################################################
'''
request_ids = torch.tensor([0, 1, 1, 1, 1, 2, 3, 4, 4, 4, 5])

h = 4

k = torch.randn(len(request_ids), h)

k_cache = {
    0: torch.ones(3, h),
    2: torch.ones(5, h),
    3: torch.zeros(2, h),
    5: torch.ones(3, h)
}

v_cache = {
    0: torch.ones(3, h),
    2: torch.ones(5, h),
    3: torch.zeros(2, h),
    5: torch.ones(3, h)
}

v = torch.randn(len(request_ids), h)


kv_cache = KVCache()
kv_cache.set_k_cache(k_cache)
kv_cache.set_v_cache(v_cache)

new_k, p, new_req_ids = kv_cache.prepend_k(request_ids, k)

new_v, _, _ = kv_cache.prepend_v(request_ids, v)

key_ind = torch.arange(len(new_k))
causal_mask = key_ind < p.reshape(len(p), 1)

# print(causal_mask)

same_req_mask = request_ids.reshape(len(request_ids), 1) == new_req_ids


print(new_k)
attention_mask = causal_mask & same_req_mask

print("p: ", p)
print("request id: ", new_req_ids)
print(attention_mask)
'''

################################################################################
'''
request_ids = torch.tensor([0, 1, 2])

h = 4

k = torch.randn(len(request_ids), h)

k_cache = {
    0: torch.ones(3, h),
    1: torch.zeros(5, h),
    2: torch.ones(2, h),
}

v_cache = {
    0: torch.ones(3, h),
    1: torch.zeros(5, h),
    2: torch.ones(2, h),
}

v = torch.randn(len(request_ids), h)


kv_cache = KVCache()
kv_cache.set_k_cache(k_cache)
kv_cache.set_v_cache(v_cache)

new_k, p, new_req_ids = kv_cache.prepend_k(request_ids, k)

new_v, _, _ = kv_cache.prepend_v(request_ids, v)

key_ind = torch.arange(len(new_k))
causal_mask = key_ind < p.reshape(len(p), 1)

# print(causal_mask)

same_req_mask = request_ids.reshape(len(request_ids), 1) == new_req_ids


# print(new_k)
# print(new_v)

# print(new_req_ids)


attention_mask = causal_mask & same_req_mask

print("p: ", p)
print(attention_mask)
'''

################################################################################

# print(causal_mask)

# print(same_req_mask)

# print(new_v)

# print(new_v.shape)


# print(kv_cache.prepend_v(request_ids, v))

# print(kv_cache.k_cache)
# # print(kv_cache.v_cache)

# print("------------------------------------------")

# request_ids = torch.tensor([0, 1, 2])
# k = torch.randn(len(request_ids), h)
# v = torch.randn(len(request_ids), h)

# print(kv_cache.prepend_k(request_ids, k))
# print(kv_cache.prepend_v(request_ids, v))

# print(kv_cache.k_cache)
# print(kv_cache.v_cache)