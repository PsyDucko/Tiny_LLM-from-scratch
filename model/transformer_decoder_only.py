import torch
import torch.nn as nn
import model.decoder_only as DL
import model.embedding as EMB
class TransformerDecoderOnly(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        num_heads: int,
        d_ff: int = 2048,
        num_layers: int = 6,
        max_seq_len: int = 512
    ):
        super().__init__()

        self.embedding = EMB.TokenEmbedding(
            vocab_size,
            d_model
        )

        self.positional_encoding = EMB.PositionalEncoding(
            d_model,
            max_seq_len
        )

        self.decoder = DL.decoder(
            d_model,
            num_heads,
            d_ff,
            num_layers
        )

        self.output_projection = nn.Linear(
            d_model,
            vocab_size
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T]

        x = self.embedding(x)  # [B, T, d_model]
        x = self.positional_encoding(x)  # [B, T, d_model]

        x = self.decoder(x, context=None)  # [B, T, d_model]

        logits = self.output_projection(x)  # [B, T, vocab_size]

        return logits