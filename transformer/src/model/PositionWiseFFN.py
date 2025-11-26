from torch import nn, softmax, transpose
import math
import torch


class PositionWiseFFN(nn.Module):
    def __init__(self,d_model,d_ff,dropout=0.1):
        super().__init__()
        self.d_model=d_model  #输入词向量维度
        self.d_ff=d_ff  #扩张后的词向量维度
        self.linear1=torch.nn.Linear(d_model,d_ff)  # 扩展维度
        self.linear2=torch.nn.Linear(d_ff,d_model)  # 压缩回原维度
        self.relu=nn.ReLU()  #激活函数
        self.dropout = nn.Dropout(dropout)
        # 添加正确的权重初始化
        self._init_weights()  #创建实例时会自动初始化

    def _init_weights(self):
        """初始化FFN权重"""
        # He初始化
        nn.init.kaiming_normal_(self.linear1.weight,nonlinearity='relu')
        nn.init.zeros_(self.linear1.bias)

        # 缩小输出层的初始化范围
        nn.init.xavier_normal_(self.linear2.weight, gain=0.02)  #gain=0.02：缩放因子，控制初始化的范围
        nn.init.zeros_(self.linear2.bias)

    def forward(self,input):
        # 将词向量维度从d_module变为d_ff，输入input的形状从(batch,seq_len,d_model)变为(batch,seq_len,d_ff)
        input=self.linear1(input)
        # 输入input_1形状(batch, seq_len, d_ff)不变
        input=self.dropout(self.relu(input)) #将维度变化后的输入进行激活函数处理，再正则化
        # 将词向量维度从d_ff变为d_module，输入input_1的形状从(batch,seq_len,d_ff)变为(batch,seq_len,d_model)
        input = self.linear2(input)
        return input




