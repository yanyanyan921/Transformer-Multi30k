from torch import nn
from src.model.DecoderLayer import DecoderLayer
from src.model.Embedding import TransformEmbedding

class Decoder(nn.Module):
    """
       Transformer解码器：基于编码器输出和已生成序列自回归地预测下一个token
       通过掩码自注意力和编码器-解码器注意力生成目标序列
    """
    def __init__(self, dec_voc_size, max_len, num_layers, num_heads,d_model, d_ff, dropout):
        #dec_voc_size: 解码器词汇表大小，目标语言词汇数量
        super().__init__()
        self.emb = TransformEmbedding(vocab_size=dec_voc_size,
                                      d_model=d_model,  # 嵌入层：将token IDs转换为向量表示 + 位置编码
                                      max_len=max_len,
                                      dropout=dropout)
        # 创建一个PyTorch模块列表容器,快速生成多个相同结构的解码子层,最终得到包含num_layers个独立解码器层的列表
        self.decoderlayers=nn.ModuleList([DecoderLayer(d_model=d_model, d_ff=d_ff, num_heads=num_heads,dropout=dropout) for _ in range(num_layers)])
        self.linear=nn.Linear(d_model,dec_voc_size)  #输出线性层,将解码器的隐藏表示映射回词汇表空间
    def forward(self,trg,encoder_output, tgt_mask=None, src_tgt_mask=None):
        # trg:目标序列，形状 (batch, trg_len);在训练时通常是完整的目标序列，在推理时是已生成的部分序列
        #编码器输入 = 特征 (Features);目标序列 = 标签 (Labels)向右移一位
        # encoder_output: 编码器输出，形状 (batch, src_len, d_model)
        # tgt_mask: 目标序列掩码（因果掩码），形状(batch, trg_len, trg_len)
        # src_tgt_mask: 源 - 目标掩码，形状(batch, trg_len, src_len)
        trg=self.emb(trg)  #输入:(batch, trg_len)->输出: (batch, trg_len, d_model)
        for decoderlayer in self.decoderlayers:  #将trg,encoder_output顺序输入num_layers个解码子层
            trg=decoderlayer(trg,encoder_output,tgt_mask, src_tgt_mask)
        output=self.linear(trg)  #输入: (batch, trg_len, d_model) -> 输出: (batch, trg_len, dec_voc_size)
        return output  # # 输出形状为(batch, trg_len, dec_voc_size) - 每个位置对词汇表的概率分布


