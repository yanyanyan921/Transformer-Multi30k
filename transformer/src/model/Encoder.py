from torch import nn
from src.model.EncoderLayer import EncoderLayer
from src.model.Embedding import TransformEmbedding


class Encoder(nn.Module):
    def __init__(self,enc_voc_size,max_len,num_layers,num_heads,d_model,d_ff,dropout):
        #enc_voc_size: 源语言词汇数量，决定嵌入矩阵大小
        super().__init__()
        self.emb=TransformEmbedding(vocab_size=enc_voc_size,
                           d_model=d_model,  # 嵌入层：将token IDs转换为向量表示 + 位置编码
                           max_len=max_len,  #输入: (batch, seq_len) -> 输出: (batch, seq_len, d_model)
                           dropout=dropout)
        # 创建一个PyTorch模块列表容器,快速生成多个相同结构的编码子层,最终得到包含num_layers个独立编码器层的列表
        self.encoderlayers=nn.ModuleList([EncoderLayer(d_model=d_model,d_ff=d_ff,num_heads=num_heads,dropout=dropout) for _ in range(num_layers)])

    def forward(self,x,src_mask=None):
        # x的形状为（batch, seq_len）;src_mask:
        # 源序列掩码 (batch, 1, seq_len, seq_len),用于处理padding位置，防止关注到无效位置
        x = self.emb(x)  #嵌入层处理：将离散的token IDs转换为连续的向量表示
        for encoderlayer in self.encoderlayers:  #将x顺序输入num_layers个编码子层，每层学习不同的特征变换
            x=encoderlayer(x,src_mask)
        return x  # 输出结果的形状为（batch, seq_len, d_model）

