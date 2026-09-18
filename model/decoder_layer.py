import torch
import torch.nn as nn

import model.cross_attention as CA
import model.feed_forward as FF
import model.multihead_attention as MHA


class DecoderLayer(nn.Module):

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int = 2048
    ):
        super().__init__()

        # Masked self-attention
        self.multihead_attn = MHA.MultiHeadAttention(
            d_model,
            num_heads,
            mask=True
        )

        # Cross-attention
        self.cross_attn = CA.cross_attention(
            d_model,
            num_heads
        )

        self.ff = FF.FeedForward(
            d_model,
            d_ff
        )

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)

    def forward(
        self,
        x: torch.Tensor,
        context: torch.Tensor
    ) -> torch.Tensor:

        # x: [B,T,d_model]
        # context: [B,S,d_model]

        # 1. Masked self-attention
        attn_output1, _ = self.multihead_attn(x)
        # attn_output1: [B,T,d_model]

        x = self.norm1(x + attn_output1)

        # 2. Cross-attention
        attn_output2, _ = self.cross_attn(x, context)
        # attn_output2: [B,T,d_model]

        x = self.norm2(x + attn_output2)

        # 3. Feed-forward
        ff_output = self.ff(x)

        x = self.norm3(x + ff_output)

        return x