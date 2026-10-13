# import torch


# x = torch.tensor([10, 20, 30, 40])
# counts = torch.tensor([2, 0, 3, 1])

# print(torch.repeat_interleave(x, repeats=counts))

import torch

lengths = torch.tensor([3, 1, 4])
values  = torch.tensor([10, 20, 30])


# flat_values = torch.repeat_interleave(values,repeats=lengths)
# print(flat_values)
# splits = torch.split(flat_values, lengths.tolist())

flat_values = torch.repeat_interleave(values, lengths)
starts = torch.cumsum(lengths, 0) - lengths          # start index of each segment
# [0, 3, 4]
positions = torch.arange(lengths.sum()) - torch.repeat_interleave(starts, lengths)
# [0 1 2 3 4 5 6 7] - [0 0 0 3 4 4 4 4] = [0 1 2 0 0 1 2 3]

print(flat_values)
print(positions)

'''
flat_values: [10, 10, 10, 20, 30, 30, 30, 30]
positions: [0, 1, 2, 0, 0, 1, 2, 3]
'''


# # import torch
# # import torch.nn as nn
# # from transformers import GPT2LMHeadModel 
# # import math

# # # class KVCache:
# # #     def __init__(self):
# # #         self.k_cache = {}
# # #         self.v_cache = {}

# # #     def prepend()



# # class Attention(nn.Module):
# #     def __init__(self, config, hf_attention):
# #         super().__init__()

# #         self.n_head = config.n_head
# #         self.n_embd = config.n_embd
# #         self.head_dim = config.n_embd // config.n_head

# #         # GPT-2 uses one linear layer to produce Q, K, and V
# #         self.c_attn = nn.Linear(
# #             self.n_embd,
# #             3 * self.n_embd
# #         )

# #         # Projects concatenated attention heads back to hidden dimension
# #         self.c_proj = nn.Linear(
# #             self.n_embd,
# #             self.n_embd
# #         )

# #         # Load pretrained Hugging Face weights
# #         # HF's GPT-2 uses Conv1D internally, which stores its weight as
# #         # [in_features, out_features] -- the transpose of nn.Linear's
# #         # [out_features, in_features] -- so it must be transposed on copy.
# #         with torch.no_grad():
# #             self.c_attn.weight.copy_(hf_attention.c_attn.weight.t())
# #             self.c_attn.bias.copy_(hf_attention.c_attn.bias)

# #             self.c_proj.weight.copy_(hf_attention.c_proj.weight.t())
# #             self.c_proj.bias.copy_(hf_attention.c_proj.bias)

# #         self.k_cache = {1: [torch.rand_like((1,768))]}
# #         self.v_cache = {}

# #     def forward(self, x, request_ids):
# #         # x is packed: [total_tokens, C], no batch dimension. total_tokens
# #         # is every in-flight request's tokens concatenated back-to-back.
# #         total_tokens, _ = x.shape

# # #         # [T, C] -> [T, 3C]
# # #         qkv = self.c_attn(x)

# # #         # [T, 3C] -> three [T, C] tensors
# # #         q, k, v = qkv.split(self.n_embd, dim=-1)

# # #         unique_req_ids = 

# # import torch


# # class KVCache:

# #     def __init__(self):
# #         self.k_cache = {}
# #         self.v_cache = {}

# #     def set_k_cache(self, k_cache):
# #         self.k_cache = k_cache

# #     def set_v_cache(self, v_cache):
# #         self.v_cache = v_cache

# #     def prepend_k(self, request_ids, k):
# #         return self.prepend_generic(request_ids, self.k_cache, k)

# #     def prepend_v(self, request_ids, v):
# #         return self.prepend_generic(request_ids, self.v_cache, v)

# #     def prepend_generic(self, request_ids, cache, vec):
# #         unique_req_ids, counts = torch.unique_consecutive(request_ids, return_counts=True)
# #         new = torch.tensor([])
# #         vec_ind = 0
# #         for i, unique_req_id in enumerate(unique_req_ids):
# #             count = counts[i]

# #             if unique_req_id.item() not in cache:
# #                 new = torch.cat((new, vec[vec_ind:vec_ind+count]))
# #                 vec_ind += count
# #                 continue

# #             new = torch.cat((new, cache[unique_req_id.item()]))
# #             new = torch.cat((new, vec[vec_ind:vec_ind+count]))
# #             vec_ind += count 

# #         return new


# # request_ids = torch.tensor([0, 1, 1, 1, 1, 2])

# # h = 4

# # k = torch.randn(len(request_ids), h)

# # k_cache = {
# #     0: torch.ones(3, h),
# #     2: torch.ones(5, h)
# # }

# # v_cache = {
# #     0: torch.ones(3, h),
# #     2: torch.ones(5, h)
# # }

# # v = torch.randn(len(request_ids), h)


# # kv_cache = KVCache()
# # kv_cache.set_k_cache(k_cache)
# # kv_cache.set_v_cache(v_cache)

# # print(kv_cache.prepend_k(request_ids, k))
# # print(kv_cache.prepend_v(request_ids, v))


# import torch
# request_ids = torch.tensor([0, 1, 1, 1, 1, 2])
# h = 4
# k = torch.randn(len(request_ids), h)
# k_cache = {
#     0: torch.ones(3, h),
#     2: torch.ones(5, h)
# }

# new_k = torch.tensor([])

# # print("request ids", request_ids)

# # print("unique req ids and counts", unique_req_ids, counts)

# # print("original k: ", k)

# unique_req_ids, counts = torch.unique_consecutive(request_ids, return_counts=True)
# ksplit = torch.split(k, counts.tolist())
# new_k = []
# for k_piece, unique_req_id in zip(ksplit, unique_req_ids):
#     if unique_req_id.item() not in k_cache:
#         new_k.append(k_piece)
#         continue 
#     cached_vec = k_cache[unique_req_id.item()]
#     new_k.append(cached_vec)
#     k_cache[unique_req_id.item()] = torch.cat((cached_vec, k_piece))
#     new_k.append(k_piece)

# new_k_tensor = torch.cat(new_k)

# print(new_k_tensor)
# print(new_k_tensor.shape)

# print(k_cache)

# # print(new_k)
# # print(new_k.shape)
    
