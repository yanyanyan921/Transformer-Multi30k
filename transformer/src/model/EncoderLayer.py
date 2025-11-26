from torch import nn
from src.model.MultiHeadSelfAttention import MultiHeadSelfAttention
from src.model.PositionWiseFFN import PositionWiseFFN
from src.model.LayerNorm import LayerNorm

class EncoderLayer(nn.Module):
    def __init__(self,d_model,d_ff,num_heads, dropout=0.1):
        super().__init__()
        self.multiheadattention = MultiHeadSelfAttention(d_model, num_heads)  # 创建多头自注意力实例
        self.ffn = PositionWiseFFN(d_model, d_ff)  # 创建前馈神经网络实例
        self.layernorm1=LayerNorm(d_model)  #创建层归一化实例
        self.layernorm2=LayerNorm(d_model)  #创建层归一化实例
        self.dropout = nn.Dropout(dropout)


    def forward(self,x,src_mask=None):
        # X的形状为（batch, seq_len, d_model）
        # 子层1：层归一化→ 多头自注意力 → 残差连接
        x_norm=self.layernorm1(x)  #层归一化
        x_attentioned=self.multiheadattention(x_norm,x_norm,x_norm,src_mask)  #计算多头注意力
        output1=x_norm+self.dropout(x_attentioned)  #残差连接,得到注意力子层最终输出


        # 子层2：层归一化→ 前馈网络 → 残差连接
        output1_norm=self.layernorm2(output1)  # 层归一化
        output1_ffn=self.ffn(output1_norm)  # 前馈神经网络计算
        output2=output1_norm+self.dropout(output1_ffn)  #残差连接,得到前馈子层最终输出
        return output2  # 输出形状保持为（batch, seq_len, d_model）


