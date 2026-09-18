import torch
import torch.nn as nn

class cross_attention(nn.Module):
    def __init__(self, d_model: int, num_heads: int, mask: bool=False):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        
        self.num_heads = num_heads
        self.d_model = d_model
        self.depth = d_model // num_heads
        
        self.wq = nn.Linear(d_model, d_model)
        self.wk = nn.Linear(d_model, d_model)
        self.wv = nn.Linear(d_model, d_model)

        self.mask=mask
        self.dense = nn.Linear(d_model, d_model)
        
    def split_heads(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, d_model]
        B, T, _ = x.size()
        x = x.view(B, T, self.num_heads, self.depth)  # [B, T, num_heads, depth]
        return x.permute(0, 2, 1, 3)  # [B, num_heads, T, depth]    
    def forward(self, x: torch.Tensor, context: torch.Tensor, mask=None) -> torch.Tensor:
        # x: [B, T, d_model]
        # context: [B, S, d_model]
        Q = self.split_heads(self.wq(x))  # [B, num_heads, T, depth]
        K = self.split_heads(self.wk(context))  # [B, num_heads, S, depth]
        V = self.split_heads(self.wv(context))  # [B, num_heads, S, depth]
        
        # Compute attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) / torch.sqrt(torch.tensor(self.depth).float())  # [B, num_heads, T, S]
        
        if self.mask:
            mask = torch.triu(
                torch.ones(
                    scores.size(-2), scores.size(-1),
                    device=x.device,
                    dtype=torch.bool
                ),
                diagonal=1
            )

            scores = scores.masked_fill(
                mask,
                float("-inf")
            )

        attn_weights = torch.softmax(scores, dim=-1)  # [B, num_heads, T, S]
        
        # Compute the output as a weighted sum of values
        output = torch.matmul(attn_weights, V)  # [B, num_heads, T, depth]
        
        # Concatenate heads and pass through final linear layer
        output = output.permute(0, 2, 1, 3).contiguous()  # [B, T, num_heads, depth]
        B, T, _, _ = output.shape
        output = output.view(B, T, self.d_model)  # [B, T, d_model]
        
        return self.dense(output),attn_weights  # [B, T, d_model], [B, num_heads, T, S]