import torch
import torch.nn as nn
from transformers import GPT2LMHeadModel 
import math


class GPT2Model:
    def __init__(self, model_name="gpt2"):
        hf_model = GPT2LMHeadModel.from_pretrained(model_name)
        # initialize the token embedding layer, which is [vocab_size] x [hidden dimension size]
        self.token_embedding_layer = nn.Embedding(
            hf_model.config.vocab_size,
            hf_model.config.n_embd
        )
        # initialize the position embedding layer, which is [max # of tokens possible to generate] x [hidden dimension size]
        self.position_embedding_layer = nn.Embedding(
            hf_model.config.n_positions,
            hf_model.config.n_embd,
        ) 
        # prevent py torch from computing gradients for this operation; this is inference, not training
        with torch.no_grad():
            self.token_embedding_layer.weight.copy_(hf_model.transformer.wte.weight)
            self.position_embedding_layer.weight.copy_(hf_model.transformer.wpe.weight)

        self.transformer_blocks = [TransformerBlock(hf_model, i) for i in range(hf_model.config.n_layer)]

        self.lm_head = nn.Linear(hf_model.config.n_embd, hf_model.config.vocab_size, bias=False)
        with torch.no_grad():
            self.lm_head.weight.copy_(hf_model.lm_head.weight)
        
        self.final_layer_norm = nn.LayerNorm(hf_model.config.n_embd)
        with torch.no_grad():
            self.final_layer_norm.weight.copy_(hf_model.transformer.ln_f.weight)
            self.final_layer_norm.bias.copy_(hf_model.transformer.ln_f.bias)


    # performs forward pass. input is token ids, and output is logits 
    # token ids is array of array, with N arrays (N = batch size) and each array containing tokens corresponding to a prompt
    def forward(self, token_ids, return_intermediates=False):
        # lets say token ids = [ [3, 121, 9] ] (so batch of 1). This produces [ [vector 3 in token embedding layer], [vector 121 in token embedding layer], [vector 9 in token embedding layer]]
        token_embedding = self.token_embedding_layer(token_ids)
        # lets say token ids are [ [3, 121, 9] ]. arange(len([3, 121, 9])) = arange(3) = [0, 1, 2]. This is basically generating positions 0, 1, ... N-1 where N is number of tokens in prompt
        position_embedding_indices = torch.arange(token_ids.shape[1])
        # grabs the appropriate position vectors. So vectors 0, 1, and 2
        positional_embedding = self.position_embedding_layer(position_embedding_indices)
        # adds token and positional embedding results
        x = token_embedding + positional_embedding
        embeddings = x

        hidden_states = []
        for transformer_block in self.transformer_blocks:
            x = transformer_block(x)
            hidden_states.append(x)

        final_hidden_states = self.final_layer_norm(x)
        logits = self.lm_head(final_hidden_states)

        if return_intermediates:
            return {
                "embeddings": embeddings,
                "hidden_states": hidden_states,
                "final_hidden_states": final_hidden_states,
                "logits": logits,
            }
        return logits

class TransformerBlock(nn.Module):
    def __init__(self, hf_model, block_index_number):
        super().__init__()
        self.layer_norm_1 = nn.LayerNorm(hf_model.config.n_embd)
        self.layer_norm_2 = nn.LayerNorm(hf_model.config.n_embd)
        with torch.no_grad():
            self.layer_norm_1.weight.copy_(hf_model.transformer.h[block_index_number].ln_1.weight)    
            self.layer_norm_1.bias.copy_(hf_model.transformer.h[block_index_number].ln_1.bias)  
        with torch.no_grad():
            self.layer_norm_2.weight.copy_(hf_model.transformer.h[block_index_number].ln_2.weight)    
            self.layer_norm_2.bias.copy_(hf_model.transformer.h[block_index_number].ln_2.bias)     
        self.attention = Attention(hf_model.config, hf_model.transformer.h[block_index_number].attn)
        self.mlp = MLP(hf_model.config, hf_model.transformer.h[block_index_number].mlp)

    def forward(self, x):
        x = x + self.attention(self.layer_norm_1(x))
        x = x + self.mlp(self.layer_norm_2(x))
        return x

#### This Attention Class is vibe coded ####
class Attention(nn.Module):
    def __init__(self, config, hf_attention):
        super().__init__()

        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.head_dim = config.n_embd // config.n_head

        # GPT-2 uses one linear layer to produce Q, K, and V
        self.c_attn = nn.Linear(
            self.n_embd,
            3 * self.n_embd
        )

        # Projects concatenated attention heads back to hidden dimension
        self.c_proj = nn.Linear(
            self.n_embd,
            self.n_embd
        )

        # Load pretrained Hugging Face weights
        # HF's GPT-2 uses Conv1D internally, which stores its weight as
        # [in_features, out_features] -- the transpose of nn.Linear's
        # [out_features, in_features] -- so it must be transposed on copy.
        with torch.no_grad():
            self.c_attn.weight.copy_(hf_attention.c_attn.weight.t())
            self.c_attn.bias.copy_(hf_attention.c_attn.bias)

            self.c_proj.weight.copy_(hf_attention.c_proj.weight.t())
            self.c_proj.bias.copy_(hf_attention.c_proj.bias)

    def forward(self, x):
        batch_size, seq_len, _ = x.shape

        # [B, T, C] -> [B, T, 3C]
        qkv = self.c_attn(x)

        # [B, T, 3C] -> three [B, T, C] tensors
        q, k, v = qkv.split(self.n_embd, dim=-1)

        # [B, T, C] -> [B, num_heads, T, head_dim]
        q = q.view(
            batch_size,
            seq_len,
            self.n_head,
            self.head_dim
        ).transpose(1, 2)

        k = k.view(
            batch_size,
            seq_len,
            self.n_head,
            self.head_dim
        ).transpose(1, 2)

        v = v.view(
            batch_size,
            seq_len,
            self.n_head,
            self.head_dim
        ).transpose(1, 2)

        # Q @ K^T
        # [B, H, T, D] @ [B, H, D, T]
        # -> [B, H, T, T]
        attention_scores = q @ k.transpose(-2, -1)

        # Scale by sqrt(head_dim)
        attention_scores = attention_scores / math.sqrt(self.head_dim)

        # Causal mask: token i cannot attend to tokens after i
        causal_mask = torch.tril(
            torch.ones(
                seq_len,
                seq_len,
                device=x.device,
                dtype=torch.bool
            )
        )

        attention_scores = attention_scores.masked_fill(
            ~causal_mask,
            float("-inf")
        )

        # Convert scores into probabilities
        attention_probs = torch.softmax(
            attention_scores,
            dim=-1
        )

        # Weighted sum of values
        # [B, H, T, T] @ [B, H, T, D]
        # -> [B, H, T, D]
        y = attention_probs @ v

        # Combine attention heads
        # [B, H, T, D] -> [B, T, H, D] -> [B, T, C]
        y = y.transpose(1, 2).contiguous()
        y = y.view(batch_size, seq_len, self.n_embd)

        # Final output projection
        y = self.c_proj(y)

        return y

#### This MLP Class is vibe coded #### 
class MLP(nn.Module):
    def __init__(self, config, hf_mlp):
        super().__init__()

        self.fc = nn.Linear(
            config.n_embd,
            4 * config.n_embd
        )

        self.proj = nn.Linear(
            4 * config.n_embd,
            config.n_embd
        )

        # Load pretrained Hugging Face weights (see note on Conv1D transpose
        # in Attention.__init__)
        with torch.no_grad():
            self.fc.weight.copy_(hf_mlp.c_fc.weight.t())
            self.fc.bias.copy_(hf_mlp.c_fc.bias)

            self.proj.weight.copy_(hf_mlp.c_proj.weight.t())
            self.proj.bias.copy_(hf_mlp.c_proj.bias)

    def forward(self, x):
        x = self.fc(x)
        x = torch.nn.functional.gelu(x, approximate="tanh")
        x = self.proj(x)

        return x

