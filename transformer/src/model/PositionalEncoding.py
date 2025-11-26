from torch import nn
import torch
import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_seq_len=5000,dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        assert d_model%2==0  #保证词向量维度是偶数
        # 初始化最大长度道的位置编码矩阵position，形状为（max_seq_len,d_model）,相同位置在所有句子中位置编码相同
        position=torch.zeros(max_seq_len,d_model)
        for pos in range(max_seq_len):  # 遍历一个句子中每个词的位置，pos是位置索引
            for i in range(0, d_model, 2):  #遍历每个词向量的偶数维度,i是词向量中的维度索引
                # 计算频率项：10000^(2i/d_model),为了数值稳定性，将其改为exp(log(10000)*(2 * i) / d_model)的形式
                #本代码中i就为偶数维度，不需要*2；原论文中的位置编码公式i是维度对的索引，2i 表示的是偶数维度的索引。
                freq = math.exp((i) * math.log(10000) / d_model)
                position[pos, i] = math.sin(pos / freq)  #根据奇偶交替编码，偶数维度：sin
                if i + 1 < d_model:  # 奇数维度：cos（如果存在）
                    position[pos,i + 1] = math.cos(pos/freq)
        position= position.unsqueeze(0)  #增加批次维度，形状为[1, max_seq_len, d_model]
        self.register_buffer('pe', position, persistent=False)  #位置编码矩阵设为不可学习参数
    def forward(self,input):
        #input的形状为（batch, seq_len, d_model）
        batch, seq_len, d_model=input.shape
        #最终向量=词向量+位置编码向量，批次处理中，position广播到整个批次，每个句子独立添加位置编码
        return self.dropout(input+self.pe[:,:seq_len,:]) #根据句子实际长度截取需要的位置编码

