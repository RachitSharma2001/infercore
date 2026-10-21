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

    def free_these_slots(self, slot_nums):
        for _, n in enumerate(slot_nums):
            self.free_slots.append(n)

    def request_slots(self, num_slots):
        slots_to_return = self.free_slots[:num_slots]
        self.free_slots = self.free_slots[num_slots:]
        return slots_to_return

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
