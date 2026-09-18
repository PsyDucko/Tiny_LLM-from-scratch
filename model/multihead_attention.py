import torch
import torch.nn as nn

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int, mask: bool=False):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        
        self.d_model = d_model
        self.num_heads = num_heads
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
    
    def forward(self, x: torch.Tensor, mask=None) -> torch.Tensor:
        # x: [B, T, d_model]
        Q = self.split_heads(self.wq(x))  # [B, num_heads, T, depth]
        K = self.split_heads(self.wk(x))  # [B, num_heads, T, depth]
        V = self.split_heads(self.wv(x))  # [B, num_heads, T, depth]
        
        # Compute attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) / torch.sqrt(torch.tensor(self.depth).float())  # [B, num_heads, T, T]
        
        # Apply softmax to get attention weights
        #attn_weights = torch.softmax(scores, dim=-1)  # [B, num_heads, T, T]
        T=x.size(1)
        if self.mask:
            mask = torch.triu(
                torch.ones(
                    T, T,
                    device=x.device,
                    dtype=torch.bool
                ),
                diagonal=1
            )

            scores = scores.masked_fill(
                mask,
                float("-inf")
            )

        attn_weights = torch.softmax(scores, dim=-1)
        # Compute the output as a weighted sum of values
        output = torch.matmul(attn_weights, V)  # [B, num_heads, T, depth]
        
        # Concatenate heads and pass through final linear layer
        output = output.permute(0, 2, 1, 3).contiguous()
        # [B, T, num_heads, depth]

        B, T, _, _ = output.shape

        output = output.view(B, T, self.d_model)
        # [B, T, d_model]

        output = self.dense(output)
        # [B, T, d_model]
        return output,attn_weights