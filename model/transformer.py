import torch
import torch.nn as nn
import model.encoder as E
import model.decoder as D
import model.embedding as EMB
class Transformer(nn.Module):
    def __init__(
        self,
        vocab_size,
        d_model,
        num_heads,
        d_ff=2048,
        num_layers=6,
        max_seq_len=512
    ):
        super().__init__()

        self.src_embedding = EMB.TokenEmbedding(
            vocab_size,
            d_model
        )

        self.tgt_embedding = EMB.TokenEmbedding(
            vocab_size,
            d_model
        )

        self.positional_encoding = EMB.PositionalEncoding(
            d_model,
            max_seq_len
        )

        self.encoder = E.encoder(
            d_model,
            num_heads,
            d_ff,
            num_layers
        )

        self.decoder = D.decoder(
            d_model,
            num_heads,
            d_ff,
            num_layers
        )

        self.output_projection = nn.Linear(
            d_model,
            vocab_size
        )# Assuming vocab_size is defined elsewhere
    def forward(self, src, tgt):

        src = self.src_embedding(src)
        src = self.positional_encoding(src)

        tgt = self.tgt_embedding(tgt)
        tgt = self.positional_encoding(tgt)

        context = self.encoder(src)

        output = self.decoder(
            tgt,
            context
        )

        logits = self.output_projection(output)

        return logits