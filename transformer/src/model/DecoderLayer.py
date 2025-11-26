from torch import nn
from src.model.MultiHeadSelfAttention import MultiHeadSelfAttention
from src.model.PositionWiseFFN import PositionWiseFFN
from src.model.LayerNorm import LayerNorm


class DecoderLayer(nn.Module):
    def __init__(self,d_model,d_ff,num_heads,dropout=0.1):
        super().__init__()
        self.selfattention= MultiHeadSelfAttention(d_model, num_heads)  #创建掩码多头自注意力实例,自注意力使用因果掩码
        self.crossattention= MultiHeadSelfAttention(d_model, num_heads)  #创建交叉注意力实例
        self.ffn = PositionWiseFFN(d_model, d_ff)  # 创建前馈神经网络实例
        self.layernorm1=LayerNorm(d_model)  #创建层归一化实例
        self.layernorm2=LayerNorm(d_model)  #创建层归一化实例
        self.layernorm3=LayerNorm(d_model)  #创建层归一化实例
        self.dropout = nn.Dropout(dropout)

    def forward(self,x,encoder_output, tgt_mask=None, src_tgt_mask=None):
        # x: 解码器输入,形状为（batch, tgt_len, d_model）
        # encoder_output:编码器输出,形状为（batch, src_len, d_model）
        #tgt_mask: 目标序列掩码 (batch, tgt_len, tgt_len)
        #src_tgt_mask: 源目标掩码 (batch, tgt_len, src_len)
        #子层1：掩码多头自注意力
        x_norm=self.layernorm1(x)  #层归一化
        x_selfatten=self.selfattention(x_norm,x_norm,x_norm,tgt_mask)  #掩码注意力
        output1=x_norm+self.dropout(x_selfatten)  #残差连接后是子层1的输出结果

        #子层2：编码器-解码器注意力
        output1_norm=self.layernorm2(output1)  #层归一化
        output1_crossatten=self.crossattention(output1_norm,encoder_output,encoder_output,src_tgt_mask)  #用目标序列与编码器输出进行交叉注意力计算
        output2=output1_norm+self.dropout(output1_crossatten)  #残差连接后是子层2的输出结果

        #子层3：前馈网络
        output2_norm=self.layernorm3(output2)  #层归一化
        output2_ffn=self.ffn(output2_norm)  #前馈网络
        output3=output2_norm+self.dropout(output2_ffn)  #残差连接后是子层3的输出结果

        return output3  #output3的形状为（batch, seq_len, d_model）




