import torch
import torch.nn as nn  

class Attention(nn.Module):
    def __init__(self,d_model:int):
        super().__init__()
        self.d_model = d_model
        self.scale = torch.sqrt(torch.tensor(d_model, dtype=torch.float32))
        self.wq=nn.Linear(d_model, d_model)  
        self.wk=nn.Linear(d_model, d_model)  
        self.wv=nn.Linear(d_model, d_model)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, d_model]
        Q = self.wq(x)  # [B, T, d_model]
        K = self.wk(x)  # [B, T, d_model]
        V = self.wv(x)  # [B, T, d_model]

        # Compute attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale  # [B, T, T]

        # Apply softmax to get attention weights
        attn_weights = torch.softmax(scores, dim=-1)  # [B, T, T]

        # Compute the output as a weighted sum of values
        output = torch.matmul(attn_weights, V)  # [B, T, d_model]

        return output
