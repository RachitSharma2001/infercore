# import torch

# state_dict = torch.load(
#     "../../models/tiny-gpt2/pytorch_model.bin",
#     map_location="cpu",
# )

# for name, tensor in state_dict.items():
#     print(name, tensor.shape)

# tests/test_gpt2.py

import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer

# from models.gpt2 import GPT2Model


MODEL_NAME = "sshleifer/tiny-gpt2"


def assert_same(name, actual, expected):
    """Compare two tensors and print useful debugging information."""

    print(f"\nChecking: {name}")
    print(f"  actual shape:   {tuple(actual.shape)}")
    print(f"  expected shape: {tuple(expected.shape)}")

    assert actual.shape == expected.shape, (
        f"{name}: shape mismatch: "
        f"{tuple(actual.shape)} != {tuple(expected.shape)}"
    )

    torch.testing.assert_close(
        actual,
        expected,
        rtol=1e-4,
        atol=1e-5,
    )

    print("  PASS")


def test_gpt2_intermediates():

    # =========================================================
    # 1. Load Hugging Face reference model
    # =========================================================

    hf_model = GPT2LMHeadModel.from_pretrained(MODEL_NAME)
    hf_model.eval()

    tokenizer = GPT2Tokenizer.from_pretrained(MODEL_NAME)

    # =========================================================
    # 2. Load YOUR model
    # =========================================================

    # my_model = GPT2Model.from_pretrained(
    #     "models/tiny-gpt2"
    # )
    # my_model.eval()

    # =========================================================
    # 3. Create input
    # =========================================================

    inputs = tokenizer(
        "Hello world",
        return_tensors="pt",
    )

    input_ids = inputs["input_ids"]

    print("\nInput:")
    print(input_ids)
    print("Input shape:", tuple(input_ids.shape))

    # =========================================================
    # 4. Run Hugging Face
    # =========================================================

    with torch.no_grad():
        hf_output = hf_model(
            input_ids,
            output_hidden_states=True,
        )

    hf_logits = hf_output.logits
    hf_hidden_states = hf_output.hidden_states


    # # =========================================================
    # # 5. Run YOUR model
    # # =========================================================

    # with torch.no_grad():
    #     my_output = my_model(
    #         input_ids,
    #         return_intermediates=True,
    #     )

    # my_logits = my_output["logits"]

    # # =========================================================
    # # 6. Compare embeddings
    # # =========================================================
    # #
    # # HF:
    # #
    # #   hidden_states[0]
    # #
    # # is the output of:
    # #
    # #   token embedding + position embedding
    # #
    # # before the first Transformer block.
    # # =========================================================

    # my_embeddings = my_output["embeddings"]
    # hf_embeddings = hf_hidden_states[0]

    # assert_same(
    #     "embeddings",
    #     my_embeddings,
    #     hf_embeddings,
    # )

    # # =========================================================
    # # 7. Compare Transformer blocks
    # # =========================================================
    # #
    # # HF:
    # #
    # #   hidden_states[0] = embeddings
    # #   hidden_states[1] = block 0 output
    # #   hidden_states[2] = block 1 output
    # #
    # # tiny-gpt2 has 2 layers.
    # # =========================================================

    # for layer_idx in range(2):

    #     my_block_output = my_output["hidden_states"][layer_idx]

    #     hf_block_output = hf_hidden_states[layer_idx + 1]

    #     assert_same(
    #         f"Transformer block {layer_idx}",
    #         my_block_output,
    #         hf_block_output,
    #     )

    # # =========================================================
    # # 8. Compare final LayerNorm
    # # =========================================================

    # my_final_hidden = my_output["final_hidden_states"]

    # hf_final_hidden = hf_model.transformer.ln_f(
    #     hf_hidden_states[-1]
    # )

    # assert_same(
    #     "final hidden state",
    #     my_final_hidden,
    #     hf_final_hidden,
    # )

    # # =========================================================
    # # 9. Compare logits
    # # =========================================================

    # assert_same(
    #     "logits",
    #     my_logits,
    #     hf_logits,
    # )

    # # =========================================================
    # # 10. Print success
    # # =========================================================

    # print("\n===================================")
    # print("ALL TESTS PASSED")
    # print("===================================")


if __name__ == "__main__":
    test_gpt2_intermediates()