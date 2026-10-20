import torch 

class Pool:
    def __init__(self, max_layers, max_slots, num_heads, head_dim):
        m = {}
        for layer in range(max_layers):
            m[layer] = torch.empty((max_slots, num_heads, head_dim))
        self.pool_from_layer = m
        self.free_slots = [i for i in range(max_slots)]
        self.num_heads = num_heads
        self.head_dim = head_dim

    def set_vector(self, layer_num, slot_num, vec):
        self.pool_from_layer[layer_num][slot_num] = vec
        # self.free_slots.remove(slot_num)

    def free_these_slots(self, slot_nums):
        for _, n in enumerate(slot_nums):
            self.free_slots.append(n)

    def request_slots(self, num_slots):
        slots_to_return = self.free_slots[:num_slots]
        self.free_slots = self.free_slots[num_slots:]
        return slots_to_return

    # def prepend(self, vec):

    '''
    Two jobs:
    - return the new k vector
    - modify current pool from layer 

    - what is p again?
    
    ([], [70, 80], [], [90], [110, 150, 160], [])
    ([12, 13, 14], [17], [15, 16], [20], [21], [35, 56])
    (vec1, vec2, vec3, vec4, vec5, vec6)


    '''

    def prepend(self, layer_num, request_ids, read_slots, read_offsets, write_slots, vec):
        unique_req_ids, counts = torch.unique_consecutive(request_ids, return_counts=True)
        read_slots_split = torch.split(read_slots, read_offsets.tolist())
        write_slots_split = torch.split(write_slots, counts.tolist())

        tot_slots = []
        req_len = counts + read_offsets
        block_nums = torch.ones((torch.sum(counts), )) + torch.repeat_interleave(read_offsets, repeats=counts)
        self.pool_from_layer[layer_num][write_slots] = vec.reshape(len(write_slots), self.num_heads, self.head_dim)
        for read_slots_piece, write_slots in zip(read_slots_split, write_slots_split):
            tot_slots.extend(read_slots_piece)
            tot_slots.extend(write_slots)
  
        p = torch.cumsum(torch.tensor(block_nums), dim=0) - 1
        new_req_ids = torch.repeat_interleave(unique_req_ids, repeats=torch.tensor(req_len))
        return self.pool_from_layer[layer_num][torch.tensor(tot_slots)], p, new_req_ids



# p = Pool(12, 1000, 1, 1)
# k = torch.rand((10,60))
# print(p.free_slots)
# p.set_vector(3, 1, k)
# print(p.free_slots)
# p.free_these_slots([1])
# print(p.free_slots)

# request_ids = torch.tensor([1, 1, 2, 3, 3, 3])
# read_slots = torch.tensor([3, 9])
# read_offsets = torch.tensor([0, 2, 0])
# write_slots = torch.tensor([12, 13, 14, 15, 16, 17])

# k = torch.ones((6, 12))

# newk, p, new_req_ids = p.prepend(1, request_ids, read_slots, read_offsets, write_slots, k)
# print(newk)
# print(p)
# print(new_req_ids)

'''
0 0 1 0 0 0 1 1 1 0 1 1 1 1
'''

# request_ids = torch.tensor([1, 2, 3, 3, 4, 5, 5, 5])
# read_slots = torch.tensor([500, 200, 100, 0, 400, 300])
# read_offsets = torch.tensor([2, 3, 0, 1, 0])
# write_slots = torch.tensor([12, 13, 14, 17, 15, 16, 20, 21])

# # 

# k = torch.ones((8, 1))

# newk, p, new_req_ids = p.prepend(1, request_ids, read_slots, read_offsets, write_slots, k)
# print(newk)
# print(p)
# print(new_req_ids)

###############################
# 1 1 1 0 0 1 1 1 0 1 0 0 0 1 1 1 
p = Pool(12, 1000, 1, 1)
request_ids = torch.tensor([5, 5, 5, 2, 1, 1, 3, 4, 6, 6])
read_slots = torch.tensor([70, 80, 90, 110, 150, 160])
read_offsets = torch.tensor([0, 2, 0, 1, 3, 0])
write_slots = torch.tensor([12, 13, 14, 17, 15, 16, 20, 21, 35, 56])

# 

k = torch.ones((10, 1))

newk, p, new_req_ids = p.prepend(1, request_ids, read_slots, read_offsets, write_slots, k)
print(newk)
print(p)
print(new_req_ids)

# p = Pool(12, 1000, 3, 1)
# request_ids = torch.tensor([1, 1, 1, 1])
# read_slots = torch.tensor([])
# read_offsets = torch.tensor([0])
# write_slots = torch.tensor([0, 1, 2, 3])
# k = torch.ones((4, 3))

# newk, p, new_req_ids = p.prepend(1, request_ids, read_slots, read_offsets, write_slots, k)
# print(newk)
# print(p)
# print(new_req_ids)

# print(k.shape)
# print(newk.shape)