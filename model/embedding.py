import math
import torch
import torch.nn as nn


class TokenEmbedding(nn.Module):
    def __init__(self, vocab_size: int, d_model: int):
        super().__init__()

        self.embedding = nn.Embedding(
            vocab_size,
            d_model
        )

        self.d_model = d_model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T]
        return self.embedding(x) * math.sqrt(self.d_model)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_seq_len: int = 512):
        super().__init__()

        pe = torch.zeros(max_seq_len, d_model)

        position = torch.arange(
            0,
            max_seq_len,
            dtype=torch.float
        ).unsqueeze(1)

        div_term = torch.exp(
            torch.arange(0, d_model, 2).float()
            * (-math.log(10000.0) / d_model)
        )

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        # [1, max_seq_len, d_model]
        pe = pe.unsqueeze(0)

        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, d_model]

        T = x.size(1)

        return x + self.pe[:, :T, :]