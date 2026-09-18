import torch
import torch.nn as nn
import model.decoder_layer as DL
class decoder(nn.Module):
    def __init__(self,d_model:int, num_heads:int, d_ff:int= 2048, num_layers:int=6):
        super().__init__()
        self.layers= nn.ModuleList([DL.DecoderLayer(d_model,num_heads,d_ff) for _ in range(num_layers)])
    def forward(self,x:torch.Tensor, context: torch.Tensor)->torch.Tensor:
        # x: [B, T, d_model]
        # context: [B, S, d_model]
        for layer in self.layers:
            x= layer(x, context)  # [B, T, d_model]
        return x