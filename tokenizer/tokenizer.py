from __future__ import annotations
import torch


class ByteTokenizer:
    """Simple UTF-8 byte-level tokenizer with special tokens."""

    PAD = 256
    BOS = 257
    EOS = 258

    @property
    def vocab_size(self) -> int:
        return 259

    def encode(self, s: str, add_bos=False, add_eos=False) -> torch.Tensor:
        ids = list(s.encode("utf-8"))

        if add_bos:
            ids.insert(0, self.BOS)

        if add_eos:
            ids.append(self.EOS)

        return torch.tensor(ids, dtype=torch.long)

    def decode(self, ids) -> str:
        if isinstance(ids, torch.Tensor):
            ids = ids.tolist()

        # Remove special tokens before decoding bytes
        ids = [
            i for i in ids
            if i not in (self.PAD, self.BOS, self.EOS)
        ]

        return bytes(ids).decode("utf-8", errors="ignore")