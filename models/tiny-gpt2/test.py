# tests/test_gpt2.py

import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer

from main import GPT2Model


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

    # Capture the raw output of each HF transformer block via forward hooks.
    # NOTE: `output_hidden_states` is NOT equivalent to this -- its last
    # entry has the final LayerNorm (ln_f) already applied to it, while
    # every other entry is a block's *input*. Hooking each block directly
    # gives us the true raw per-block output for every layer, including
    # the last one.
    hf_block_outputs = {}

    def make_hook(layer_idx):
        def hook(module, inputs, output):
            # GPT2Block returns a bare Tensor when output_attentions /
            # use_cache are off, or a tuple with the hidden states first
            # otherwise -- handle both.
            hf_block_outputs[layer_idx] = output[0] if isinstance(output, tuple) else output
        return hook

    for layer_idx, block in enumerate(hf_model.transformer.h):
        block.register_forward_hook(make_hook(layer_idx))

    tokenizer = GPT2Tokenizer.from_pretrained(MODEL_NAME)

    # =========================================================
    # 2. Load YOUR model
    # =========================================================

    my_model = GPT2Model(model_name=MODEL_NAME)

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

    # =========================================================
    # 5. Run YOUR model
    # =========================================================

    with torch.no_grad():
        my_output = my_model.forward(
            input_ids,
            return_intermediates=True,
        )

    my_logits = my_output["logits"]

    # =========================================================
    # 6. Compare embeddings
    # =========================================================
    #
    # HF:
    #
    #   hidden_states[0]
    #
    # is the output of:
    #
    #   token embedding + position embedding
    #
    # before the first Transformer block.
    # =========================================================

    my_embeddings = my_output["embeddings"]
    hf_embeddings = hf_hidden_states[0]

    assert_same(
        "embeddings",
        my_embeddings,
        hf_embeddings,
    )

    # =========================================================
    # 7. Compare Transformer blocks (raw output, via hooks -- see above)
    # =========================================================

    for layer_idx in range(hf_model.config.n_layer):

        my_block_output = my_output["hidden_states"][layer_idx]

        hf_block_output = hf_block_outputs[layer_idx]

        assert_same(
            f"Transformer block {layer_idx}",
            my_block_output,
            hf_block_output,
        )

    # =========================================================
    # 8. Compare final LayerNorm
    # =========================================================

    my_final_hidden = my_output["final_hidden_states"]

    hf_final_hidden = hf_model.transformer.ln_f(
        hf_block_outputs[hf_model.config.n_layer - 1]
    )

    assert_same(
        "final hidden state",
        my_final_hidden,
        hf_final_hidden,
    )

    # =========================================================
    # 9. Compare logits
    # =========================================================

    assert_same(
        "logits",
        my_logits,
        hf_logits,
    )

    # =========================================================
    # 10. Print success
    # =========================================================

    print("\n===================================")
    print("ALL TESTS PASSED")
    print("===================================")


if __name__ == "__main__":
    test_gpt2_intermediates()
