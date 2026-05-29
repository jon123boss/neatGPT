import glob
import math
import threading
from pathlib import Path

import numpy as np
import torch

DEFAULT_BOS_ID = 50256

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_data_shard(file: Path):
    """
    Load a .bin data shard with zero-copy reads.

    File format:
      - 256 int32 header
      - uint16 tokens

    Header[0] = magic (20240520)
    Header[1] = version (1)
    Header[2] = num_tokens
    """
    header = torch.from_file(str(file), False, 256, dtype=torch.int32)
    assert header[0] == 20240520, "magic number mismatch in the data .bin file"
    assert header[1] == 1, "unsupported version"
    num_tokens = int(header[2])
    with file.open("rb", buffering=0) as f:
        # Avoid pin_memory copy by allocating pinned directly
        tokens = torch.empty(num_tokens, dtype=torch.uint16, pin_memory=True)
        f.seek(256 * 4)
        nbytes = f.readinto(tokens.numpy())
        assert nbytes == 2 * num_tokens, "number of tokens read does not match header"
    return tokens


def next_multiple_of_n(v: float | int, *, n: int):
    return math.ceil(v / n) * n


# ---------------------------------------------------------------------------
# Shard management
# ---------------------------------------------------------------------------

class Shard:
    """
    Manages a single shard of token data with fast BOS indexing.

    Uses a two-phase indexing strategy for instant availability:
      1. Partial index over the first 6M tokens is built synchronously.
      2. Full index is built in a background thread and swapped in when ready.
    """

    def __init__(self, tokens: torch.Tensor, bos_id: int = DEFAULT_BOS_ID):
        self.tokens = tokens
        self.size = tokens.numel()
        self.bos_id = bos_id
        self.i = 0

        # Partial index now, full index async
        self.bos_idx = (
            (tokens[:6_000_000] == self.bos_id)
            .nonzero(as_tuple=True)[0]
            .to(torch.int64)
            .cpu()
            .numpy()
        )
        self._full_idx = None
        self._loader_thread = None
        self._ready = threading.Event()
        self._loader_thread = threading.Thread(target=self._scan)
        self._loader_thread.start()

    def _scan(self):
        self._full_idx = (
            (self.tokens == self.bos_id)
            .nonzero(as_tuple=True)[0]
            .to(torch.int64)
            .cpu()
            .numpy()
        )
        self._ready.set()

    def _maybe_switch(self):
        # Switch to full index as soon as async scan completes
        if self.bos_idx is not self._full_idx and self._ready.is_set():
            self._loader_thread.join()
            self.bos_idx = self._full_idx

    def next_batch(self, num_tokens: int, max_seq_len: int):
        """
        Extract the next batch of ``num_tokens`` tokens, aligned to BOS boundaries.

        Documents are truncated to ``max_seq_len``.  The returned ``inputs`` and
        ``targets`` satisfy ``targets[i] == inputs[i+1]`` (standard LM objective).

        Returns
        -------
        inputs : torch.Tensor
            CPU pinned uint16 tensor of shape ``(num_tokens,)``.
        targets : torch.Tensor
            CPU pinned uint16 tensor of shape ``(num_tokens,)``.
        cum_lengths : torch.Tensor
            CPU int64 tensor of cumulative document lengths.
        """
        self._maybe_switch()
        n = len(self.bos_idx)
        starts = []
        ends = []

        idx = self.i
        cur_len = 0
        while cur_len <= num_tokens:
            self._maybe_switch()
            n = len(self.bos_idx)
            if idx >= n:
                raise StopIteration("Insufficient BOS ahead; hit tail of shard.")
            cur = self.bos_idx[idx]
            starts.append(cur)
            idx += 1
            end = min(
                self.bos_idx[idx] if idx < n else self.size,
                cur + max_seq_len,
                cur + num_tokens - cur_len + 1,
            )
            ends.append(end)
            cur_len += end - cur

        assert cur_len == num_tokens + 1
        self.i = idx

        start_idxs = torch.tensor(starts)
        end_idxs = torch.tensor(ends)

        buf = torch.empty(cur_len, dtype=self.tokens.dtype)
        pos = 0
        for s, e in zip(starts, ends):
            doc_len = e - s
            buf[pos : pos + doc_len] = self.tokens[s:e]
            pos += doc_len

        inputs = buf[:-1]
        targets = buf[1:]
        # Last document was too long to account for the targets offset
        end_idxs[-1] -= 1
        cum_lengths = (end_idxs - start_idxs).cumsum(0)

        return inputs, targets, cum_lengths

    @staticmethod
    def load_async(file: Path, bos_id: int = DEFAULT_BOS_ID):
        """Returns a getter function that blocks until the shard is loaded."""
        result = {}
        ready = threading.Event()

        def load():
            tokens = _load_data_shard(file)
            result["shard"] = Shard(tokens, bos_id=bos_id)
            ready.set()

        thread = threading.Thread(target=load)
        thread.start()

        def get():
            ready.wait()
            thread.join()
            return result["shard"]

        return get


# ---------------------------------------------------------------------------
# FastDataLoader
# ---------------------------------------------------------------------------

class FastDataLoader:
    """
    High-performance single-GPU dataloader for .bin token files.

    Parameters
    ----------
    filename_pattern : str
        Glob pattern matching one or more ``.bin`` shards.
    batch_size : int
        Number of tokens per batch.  The caller is responsible for dividing by
        ``grad_accum_steps`` if desired.
    max_seq_len : int
        Maximum sequence length (in tokens) for BOS-aligned mode.  Ignored when
        ``align_to_bos=False``.
    align_to_bos : bool, default True
        If True, each sequence starts on a BOS token and documents are never
        concatenated across batch boundaries.  If False, contiguous chunks of
        ``batch_size`` tokens are returned.
    bos_id : int, default 50256
        Token ID that marks the beginning of a sequence.
    device : str, default "cuda"
        CUDA device to transfer tensors to.

    Yields
    ------
    inputs : torch.Tensor
        ``(batch_size,)`` int64 CUDA tensor.
    targets : torch.Tensor
        ``(batch_size,)`` int64 CUDA tensor.
    cu_seqlens : torch.Tensor
        ``(max_num_docs,)`` int32 CUDA tensor.  Padded cumulative sequence
        lengths for Flash Attention varlen.  Static shape avoids graph breaks.
    n_docs : int
        Actual number of documents in this batch.  Use to slice ``cu_seqlens``
        before passing to ``flash_attn_varlen_func``.
    max_seqlen : int
        Maximum sequence length among documents in this batch.  Precomputed
        on CPU to avoid a GPU→CPU sync / ``torch.compile`` graph break.
    """

    def __init__(
        self,
        filename_pattern: str,
        batch_size: int,
        max_seq_len: int,
        align_to_bos: bool = True,
        bos_id: int = DEFAULT_BOS_ID,
        device: str = "cuda",
    ):
        self.files = [Path(f) for f in sorted(glob.glob(filename_pattern))]
        if not self.files:
            raise FileNotFoundError(f"No files found for pattern: {filename_pattern}")

        self.batch_size = batch_size
        self.max_seq_len = max_seq_len
        self.align_to_bos = align_to_bos
        self.bos_id = bos_id
        self.device = device

        # Precompute static cu_seqlens shape to avoid torch.compile recompiles
        self._update_max_num_docs()

        # Cache total tokens for __len__
        self._total_tokens = 0
        for f in self.files:
            header = torch.from_file(str(f), False, 256, dtype=torch.int32)
            self._total_tokens += int(header[2])

        self.file_iter = iter(self.files)
        self.current_shard = None
        self.next_shard_getter = None
        self.tokens = None
        self.pos = 0

        if self.align_to_bos:
            self._load_first_shard()
        else:
            self.tokens = _load_data_shard(next(self.file_iter))

        self._generator = self._gen()

    # -----------------------------------------------------------------------
    # Internal helpers
    # -----------------------------------------------------------------------

    def _update_max_num_docs(self):
        """Compute padded ``max_num_docs`` for static-shaped ``cu_seqlens``."""
        defaults = {16384: 64, 32768: 96, 49152: 128}
        self.max_num_docs = defaults.get(
            self.batch_size, next_multiple_of_n(self.batch_size // 300, n=128)
        )

    def _load_first_shard(self):
        self.current_shard = Shard(_load_data_shard(next(self.file_iter)), bos_id=self.bos_id)
        self._preload_next_shard()

    def _preload_next_shard(self):
        try:
            next_file = next(self.file_iter)
            self.next_shard_getter = Shard.load_async(next_file, bos_id=self.bos_id)
        except StopIteration:
            self.next_shard_getter = None

    def _switch_shard(self):
        if self.next_shard_getter is not None:
            self.current_shard = self.next_shard_getter()
            self._preload_next_shard()
            return True
        return False

    def _build_cu_seqlens(self, cum_lengths: torch.Tensor) -> torch.Tensor:
        """
        Build static-shaped ``cu_seqlens`` for Flash Attention.

        Padded to ``max_num_docs`` so the tensor shape is identical across
        every training step.  This prevents ``torch.compile`` graph breaks.
        """
        n_docs = len(cum_lengths)
        cu_seqlens = torch.full((self.max_num_docs,), 0, dtype=torch.int32)
        cu_seqlens[0] = 0
        if n_docs > 0:
            cu_seqlens[1 : n_docs + 1] = cum_lengths
        return cu_seqlens

    # -----------------------------------------------------------------------
    # Generator core
    # -----------------------------------------------------------------------

    def _gen(self):
        while True:
            if self.align_to_bos:
                try:
                    inputs, targets, cum_lengths = self.current_shard.next_batch(
                        self.batch_size, self.max_seq_len
                    )
                except StopIteration:
                    if not self._switch_shard():
                        return
                    inputs, targets, cum_lengths = self.current_shard.next_batch(
                        self.batch_size, self.max_seq_len
                    )
            else:
                if self.pos + self.batch_size + 1 >= len(self.tokens):
                    try:
                        self.tokens = _load_data_shard(next(self.file_iter))
                        self.pos = 0
                    except StopIteration:
                        return

                buf = self.tokens[self.pos : self.pos + self.batch_size + 1]
                inputs = buf[:-1]
                targets = buf[1:]
                # Document boundaries within this contiguous chunk
                bos_positions = torch.nonzero(inputs == self.bos_id, as_tuple=False)[:, 0]
                if len(bos_positions) == 0:
                    doc_lengths = torch.tensor([len(inputs)], dtype=torch.int64)
                else:
                    starts = torch.cat([torch.tensor([0], dtype=torch.int64), bos_positions])
                    ends = torch.cat([bos_positions, torch.tensor([len(inputs)], dtype=torch.int64)])
                    doc_lengths = ends - starts
                cum_lengths = doc_lengths.cumsum(0)
                self.pos += self.batch_size

            # Build padded cu_seqlens for flash attention varlen
            n_docs = len(cum_lengths)
            cu_seqlens = self._build_cu_seqlens(cum_lengths)

            # Cast to the exact dtypes the model expects (do this on CPU)
            inputs = inputs.to(dtype=torch.int32)
            targets = targets.to(dtype=torch.int64)

            # Precompute max_seqlen on CPU to avoid .item() GPU sync in model
            actual_max = int(cum_lengths.max().item()) if n_docs > 0 else self.batch_size

            # Non-blocking H2D transfer — GPU kernel queue can overlap with next batch IO
            batch = (
                inputs.to(device=self.device, non_blocking=True),
                targets.to(device=self.device, non_blocking=True),
                cu_seqlens.to(device=self.device, non_blocking=True),
                n_docs,
                actual_max,
            )

            new_params = yield batch
            if new_params is not None:
                # Accept either (batch_size, max_seq_len) or (batch_size, max_seq_len, _)
                self.batch_size = int(new_params[0])
                self.max_seq_len = int(new_params[1])
                self._update_max_num_docs()

    # -----------------------------------------------------------------------
    # Public API
    # -----------------------------------------------------------------------

    def __iter__(self):
        return self

    def __next__(self):
        return next(self._generator)

    def send(self, params):
        """
        Send new hyper-parameters into the generator.

        ``params`` should be ``(batch_size, max_seq_len)`` or
        ``(batch_size, max_seq_len, grad_accum_steps)`` (the third element is
        ignored).  Passing ``None`` is equivalent to ``next(loader)``.
        """
        return self._generator.send(params)

    def __len__(self):
        """Approximate number of batches (for progress bars)."""
        return self._total_tokens // self.batch_size

    def reset(self):
        """Reset the loader so it can be iterated again (new epoch)."""
        torch.cuda.synchronize()
        self.file_iter = iter(self.files)
        self.current_shard = None
        self.next_shard_getter = None
        self.pos = 0
        if self.align_to_bos:
            self._load_first_shard()
        else:
            self.tokens = _load_data_shard(next(self.file_iter))
        self._generator = self._gen()

    def update(self, batch_size: int | None = None, max_seq_len: int | None = None):
        """
        Update batch size and/or max sequence length in-place.
        Takes effect on the *next* batch yielded.
        """
        if batch_size is not None:
            self.batch_size = batch_size
        if max_seq_len is not None:
            self.max_seq_len = max_seq_len
        self._update_max_num_docs()
