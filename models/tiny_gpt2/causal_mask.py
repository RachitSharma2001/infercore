import torch
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


# num_heads = 2
# new_k_tensor = new_k_tensor.view(num_heads, 14, 2)

# print(new_k_tensor.shape)


# matrix: 5x6x14

m = torch.zeros((5,6,14))
request_ids = torch.tensor([0, 1, 1, 1, 1, 2])
new_request_ids = torch.tensor([0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 2])
pos_in_new_request_ids = {
    0: 2,
    1: 3,
    2: 4,
    3: 5,
    4: 6,
    5: 11,
}

'''
[
[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
]
[0, 0, 0, 1, 2, 3, 4, 5, 5, 5, 5]




'''


# original_num_tokens = 6
# new_num_tokens = 11


# indices = torch.tensor([int(i / new_num_tokens) for i in range(original_num_tokens*new_num_tokens)]).reshape(original_num_tokens, new_num_tokens)

# pos = torch.tensor([])

# pos = torch.tensor([0, 0, 0, 1, 2, 3, 4, 5, 5, 5, 5])

# res = indices >= pos

# print(res)




### Figure out: how does this work???


# indices = torch.cat([torch.arange(6).reshape(6,1) for _ in range(11)], dim=1)

# key_ind = torch.arange(11)
# p = torch.tensor([2, 3, 4, 5, 6, 11]).reshape(6, 1)

# print(key_ind <= p)


old_request_ids = torch.tensor([0, 1, 1, 1, 1, 2])
new_request_ids = torch.tensor([0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 2])

print(old_request_ids.reshape(6, 1) == new_request_ids)




# print(torch.arange(6).reshape(6,1))
# col = torch.stack([torch.tensor([])])



# indices = torch.tensor([int(i / 11) for i in range(6*11)]).reshape(6, 11)

# print("indices: ", indices)

# pos = torch.tensor([0, 0, 0, 1, 2, 3, 4, 5, 5, 5, 5])

# res = indices >= pos

# print(res)


# cm = torch.tensor([])
# for i in range(6):
#     curr_mask = []
#     for j in range(11):
#         if j <= pos_in_new_request_ids[i]:
#             curr_mask.append(1)
#         else:
#             curr_mask.append(0)
#     cm = torch.cat((cm, torch.tensor(curr_mask)))
# cm = cm.reshape(6, 11)

# # cm = [torch.tensor([1 if j <= pos_in_new_request_ids[i] else 0 for j in range(14)]) for i in range(6)]

# print(cm)