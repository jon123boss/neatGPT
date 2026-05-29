"""
Integration test for neatGPT model + dataloader.

Creates a dummy .bin shard, instantiates FastDataLoader + DecoderOnlyTransformer,
and runs both the varlen (cu_seqlens) and autoregressive (kv_cache) paths.
Runs on CUDA with bf16 and torch.compile.
"""

import struct
import tempfile
from pathlib import Path

import torch
import torch.nn as nn

from dataloader import FastDataLoader
from model import DecoderOnlyTransformer


def create_dummy_bin(path: Path, num_tokens: int, vocab_size: int = 1000, bos_id: int = 50256):
    """Write a .bin shard in the expected format."""
    # 256 int32 header
    header = [0] * 256
    header[0] = 20240520  # magic
    header[1] = 1         # version
    header[2] = num_tokens

    with open(path, "wb") as f:
        # Write header
        for val in header:
            f.write(struct.pack("<i", val))
        # Write tokens as uint16
        tokens = torch.randint(0, vocab_size, (num_tokens,), dtype=torch.int32)
        # Sprinkle BOS tokens every ~256 tokens so the dataloader can find docs
        for i in range(0, num_tokens, 256):
            tokens[i] = bos_id
        tokens_np = tokens.numpy().astype("uint16")
        f.write(tokens_np.tobytes())


def main():
    device = "cuda"
    dtype = torch.bfloat16
    batch_size = 1024
    max_seq_len = 512
    bos_id = 50256
    vocab_size = 1000

    # --- 1. Create dummy .bin shard ---
    tmpdir = tempfile.mkdtemp()
    shard_path = Path(tmpdir) / "dummy_0000.bin"
    # Need enough tokens for a few batches
    create_dummy_bin(shard_path, num_tokens=500_000, vocab_size=vocab_size, bos_id=bos_id)
    print(f"Created dummy shard: {shard_path}")

    # --- 2. Build tiny model ---
    config = {
        "vocab_size": vocab_size,
        "d_model": 256,
        "n_heads": 4,
        "kv_heads": 2,
        "n_layers": 2,
        "ffn_ratio": 4,
        "max_seq_len": max_seq_len,
        "rope_base": 10000,
        "norm_eps": 1e-6,
        "weight_tied": True,
    }

    model = DecoderOnlyTransformer(config).to(device=device, dtype=dtype)
    model = torch.compile(model)
    print(f"Model params: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")

    # --- 3. Build dataloader ---
    loader = FastDataLoader(
        filename_pattern=str(Path(tmpdir) / "*.bin"),
        batch_size=batch_size,
        max_seq_len=max_seq_len,
        align_to_bos=True,
        bos_id=bos_id,
        device=device,
    )

    # --- 4. Test varlen path (cu_seqlens) ---
    print("\n--- Test 1: Varlen forward (cu_seqlens) ---")
    inputs, targets, cu_seqlens, n_docs, max_seqlen = next(loader)
    print(f"  inputs shape: {inputs.shape}, dtype: {inputs.dtype}")
    print(f"  targets shape: {targets.shape}, dtype: {targets.dtype}")
    print(f"  cu_seqlens shape: {cu_seqlens.shape}, n_docs: {n_docs}, max_seqlen: {max_seqlen}")

    with torch.no_grad():
        logits, kv_cache = model(
            inputs,
            kv_cache=None,
            cu_seqlens=cu_seqlens,
            n_docs=n_docs,
            max_seqlen=max_seqlen,
        )
    print(f"  logits shape: {logits.shape}, dtype: {logits.dtype}")
    assert logits.shape == (1, batch_size, vocab_size), f"Unexpected logits shape: {logits.shape}"
    assert logits.dtype == dtype
    print("  Varlen forward: OK")

    # --- 5. Test autoregressive path (kv_cache) ---
    print("\n--- Test 2: Autoregressive forward (kv_cache) ---")
    prompt_len = 64
    gen_len = 32
    prompt = torch.randint(0, vocab_size, (1, prompt_len), device=device, dtype=torch.int64)

    with torch.no_grad():
        logits, kv_cache = model(prompt, kv_cache=None)
    print(f"  Prompt forward logits: {logits.shape}")
    assert logits.shape == (1, prompt_len, vocab_size)

    for step in range(gen_len):
        next_token = logits[:, -1:].argmax(dim=-1)
        with torch.no_grad():
            logits, kv_cache = model(next_token, kv_cache=kv_cache)
        assert logits.shape == (1, 1, vocab_size)
        assert len(kv_cache) == config["n_layers"]
        assert kv_cache[0] is not None
    print(f"  Generated {gen_len} tokens autoregressively")
    print("  KV-cache forward: OK")

    # --- 6. Loss sanity check ---
    print("\n--- Test 3: Loss computation ---")
    logits, _ = model(inputs, cu_seqlens=cu_seqlens, n_docs=n_docs, max_seqlen=max_seqlen)
    loss = nn.functional.cross_entropy(
        logits.view(-1, vocab_size).float(),
        targets.view(-1),
    )
    print(f"  loss: {loss.item():.4f}")
    assert loss.isfinite()
    print("  Loss computation: OK")

    print("\nAll tests passed!")


if __name__ == "__main__":
    main()
