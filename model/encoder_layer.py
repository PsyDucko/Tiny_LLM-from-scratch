import torch
import torch.nn as nn
import model.multihead_attention as MHA 
import model.feed_forward as FF
class encoder_layer(nn.Module):
    def __init__(self,d_model:int, num_heads:int, d_ff:int= 2048):
        super().__init__()
        self.mha= MHA.MultiHeadAttention(d_model,num_heads)
        self.ff= FF.FeedForward(d_model,d_ff)
        self.norm1= nn.LayerNorm(d_model)
        self.norm2= nn.LayerNorm(d_model)
    def forward(self,x:torch.Tensor)->torch.Tensor:
        # x: [B, T, d_model]
        attn_output,_= self.mha(x)  # [B, T, d_model]
        x= self.norm1(x+attn_output)  # [B, T, d_model]
        ff_output= self.ff(x)  # [B, T, d_model]
        x= self.norm2(x+ff_output)  # [B, T, d_model]
        return x
