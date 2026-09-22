# tests/test2_gpt2.py
#
# Verifies continuous-batching correctness: two different requests packed
# together into one flat call must produce exactly the same logits as
# running each request through HF individually. This is the property that
# makes packing safe -- request_ids must fully isolate attention between
# requests, with no cross-contamination and no padding needed.

import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer

from tinygpt2 import GPT2Model


MODEL_NAME = "sshleifer/tiny-gpt2"


def test_gpt2_packed_batch_matches_individual_requests():

    hf_model = GPT2LMHeadModel.from_pretrained(MODEL_NAME)
    hf_model.eval()

    tokenizer = GPT2Tokenizer.from_pretrained(MODEL_NAME)

    model = GPT2Model(model_name=MODEL_NAME)

    prompts = [
        "Hello world",
        "The quick brown fox jumps",
    ]

    # Tokenize each prompt on its own -- no padding. Packing needs no
    # padding since requests aren't stacked into a rectangular batch.
    token_id_list = [
        tokenizer(prompt, return_tensors="pt")["input_ids"][0]
        for prompt in prompts
    ]

    print("\nPer-request lengths:", [len(t) for t in token_id_list])

    # Pack: concatenate every request's tokens back-to-back.
    token_ids = torch.cat(token_id_list, dim=0)
    position_ids = torch.cat(
        [torch.arange(len(t)) for t in token_id_list],
        dim=0,
    )
    request_ids = torch.cat(
        [
            torch.full((len(t),), i, dtype=torch.long)
            for i, t in enumerate(token_id_list)
        ],
        dim=0,
    )

    print("Packed token_ids:", token_ids)
    print("Packed position_ids:", position_ids)
    print("Packed request_ids:", request_ids)

    with torch.no_grad():
        packed_logits = model.forward(token_ids, request_ids, position_ids)

    assert packed_logits.shape == (len(token_ids), model.lm_head.out_features), (
        f"unexpected logits shape: {tuple(packed_logits.shape)}"
    )
    assert torch.isfinite(packed_logits).all(), "logits contain NaN/Inf values"

    # Each request's slice of the packed logits must match running that
    # request through HF completely on its own -- proof that no request
    # attended to another request's tokens.
    offset = 0
    for i, request_token_ids in enumerate(token_id_list):
        request_len = len(request_token_ids)

        with torch.no_grad():
            hf_logits = hf_model(request_token_ids.unsqueeze(0)).logits[0]

        my_logits = packed_logits[offset:offset + request_len]

        torch.testing.assert_close(
            my_logits,
            hf_logits,
            rtol=1e-4,
            atol=1e-5,
            msg=f"request {i} logits diverged from its standalone HF run -- "
                f"packed attention is leaking across requests",
        )

        offset += request_len

    print("\n===================================")
    print("ALL TESTS PASSED")
    print("===================================")


if __name__ == "__main__":
    test_gpt2_packed_batch_matches_individual_requests()
