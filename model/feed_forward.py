import torch
import torch.nn as nn

class FeedForward(nn.Module):
    def __init__ (self, d_model: int, d_ff: int = 2048):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(d_ff, d_model)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, d_model]
        x = self.linear1(x)  # [B, T, d_ff]
        x = self.relu(x)     # [B, T, d_ff]
        x = self.linear2(x)  # [B, T, d_model]
        return x