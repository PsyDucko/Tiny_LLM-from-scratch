import torch
import torch.nn as nn
import model.encoder_layer as EL
class encoder(nn.Module):
    def __init__(self,d_model:int, num_heads:int, d_ff:int= 2048, num_layers:int=6):
        super().__init__()
        self.layers= nn.ModuleList([EL.encoder_layer(d_model,num_heads,d_ff) for _ in range(num_layers)])
    def forward(self,x:torch.Tensor)->torch.Tensor:
        # x: [B, T, d_model]
        for layer in self.layers:
            x= layer(x)  # [B, T, d_model]
        return x