from torch import nn
import torch


class LayerNorm(nn.Module):
    def __init__(self,d_model):
        super().__init__()
        self.eps = 1e-5
        self.weight = nn.Parameter(torch.ones(d_model))  #初始化随机可学习的缩放参数
        self.bias = nn.Parameter(torch.zeros(d_model))  #初始化随机可学习的偏移参数

    def forward(self,input):
        #input的形状为（batch, seq_len, d_model）
        # 对最后一个维度，即词向量维度计算该词向量的均值，得到结果的形状为（batch, seq_len, 1）
        mean=torch.mean(input, dim=-1, keepdim=True)
        # 对最后一个维度，即词向量维度计算该词向量的方差，得到结果的形状为（batch, seq_len, 1）
        var=torch.var(input, dim=-1, keepdim=True, unbiased=False)  # 在LayerNorm中通常使用总体方差(除以n)
        #自动广播，可以直接广播计算，对每个样本（词向量）的所有特征进行归一化
        # 最终输出结果的形状为（batch, seq_len, d_model）
        input_normalize=self.weight*((input-mean)/torch.sqrt(var+self.eps))+self.bias
        return input_normalize

