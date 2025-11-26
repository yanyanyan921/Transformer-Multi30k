import torch
import torch.nn as nn
from src.model.Encoder import Encoder
from src.model.Decoder import Decoder


class Transformer(nn.Module):
    def __init__(self, src_pad_idx, trg_pad_idx, trg_sos_idx, enc_voc_size, dec_voc_size, max_len, d_model,num_heads,
                 d_ff,num_layers, dropout, device):
        super(Transformer, self).__init__()
        self.src_pad_idx = src_pad_idx  #源序列padding的token索引
        self.trg_pad_idx = trg_pad_idx  #目标序列padding的token索引
        self.trg_sos_idx = trg_sos_idx  #目标序列起始符<start>的token索引
        self.device = device
        #enc_voc_size: 编码器词汇表大小（源语言词汇量）
        #dec_voc_size: 解码器词汇表大小（目标语言词汇量）
        # 编码器：将源序列编码为上下文相关的表示
        self.encoder = Encoder(enc_voc_size=enc_voc_size,
                               max_len=max_len,
                               num_layers=num_layers,
                               num_heads=num_heads,
                               d_model=d_model,
                               d_ff=d_ff,
                               dropout=dropout)
        # 解码器：基于编码器输出自回归地生成目标序列
        self.decoder = Decoder(dec_voc_size=dec_voc_size,
                               max_len=max_len,
                               num_layers=num_layers,
                               num_heads=num_heads,
                               d_model=d_model,
                               d_ff=d_ff,
                               dropout=dropout)

    def forward(self, src, trg):
        #src: 源序列，形状为[batch_size, src_len]的token IDs
        #trg: 目标序列，形状为[batch_size, trg_len]的token IDs
        #在训练时使用教师强制,包含完整的序列

        # 1. 生成三种不同的注意力掩码
        # 源序列自注意力掩码：防止关注padding位置
        # 形状: [batch, 1, src_len, src_len]
        src_mask = self.make_pad_mask(src, src, self.src_pad_idx, self.src_pad_idx)
        # 源-目标交叉注意力掩码：防止目标序列关注源序列的padding位置
        # 形状: [batch, 1, trg_len, src_len]
        src_trg_mask = self.make_pad_mask(trg, src, self.trg_pad_idx, self.src_pad_idx)
        # 目标序列自注意力掩码：因果掩码 + padding掩码
        # 因果掩码防止看到未来信息，padding掩码防止关注padding位置
        # 形状: [batch, 1, trg_len, trg_len]
        trg_mask = self.make_pad_mask(trg, trg, self.trg_pad_idx, self.trg_pad_idx) * \
                   self.make_causal_mask(trg, trg)
        # 2. 编码器处理：将源序列编码为上下文表示
        # 输入: [batch, src_len] -> 输出: [batch, src_len, d_model]
        enc_src = self.encoder(src, src_mask)
        # 3. 解码器处理：基于编码器输出生成目标序列
        # 输入: [batch, trg_len] + 编码器输出 -> 输出: [batch, trg_len, dec_voc_size]
        output = self.decoder(trg, enc_src, trg_mask, src_trg_mask)

        return output  #解码器输出，形状为 [batch, trg_len, dec_voc_size]

    def make_pad_mask(self, q, k, q_pad_idx, k_pad_idx):
        """
        生成padding掩码，防止注意力机制关注padding位置;掩码生成发生在嵌入之前
        q: 查询序列，形状 [batch, len_q]
        k: 键序列，形状 [batch, len_k]
        q_pad_idx: 查询序列的padding索引
        k_pad_idx: 键序列的padding索引
        """
        len_q, len_k = q.size(1), k.size(1)
        # 生成键序列的掩码：标记非padding位置为True,, padding为False
        k_mask = k.ne(k_pad_idx).unsqueeze(1).unsqueeze(2)  # k_mask形状: [batch, 1, 1, len_k]
        # 在第二维度复制len_q次,扩展为: [batch, 1, len_q, len_k]
        k_mask = k_mask.repeat(1, 1, len_q, 1)

        # 生成查询序列的掩码：标记非padding位置为True
        q_mask = q.ne(q_pad_idx).unsqueeze(1).unsqueeze(3)  # q_mask形状: [batch, 1, len_q, 1]
        # 在第三维度复制len_k次,扩展为: [batch, 1, len_q, len_k]
        q_mask = q_mask.repeat(1, 1, 1, len_k)

        #合并掩码 - 只有q和k都是非padding时才为True
        mask = k_mask & q_mask  #mask形状: [batch, 1, len_q, len_k]
        return mask

    def make_causal_mask(self, q, k):
        """
        生成因果掩码（下三角掩码），防止解码器看到未来信息
        q: 查询序列，形状 [batch, len_q]
        k: 键序列，形状 [batch, len_k]
        mask: 因果掩码，形状 [len_q, len_k]
        下三角为True（允许关注），上三角为False（禁止关注未来）
        """
        len_q, len_k = q.size(1), k.size(1)

        # 创建下三角矩阵：确保每个位置只能关注之前和当前位置
        # torch.tril(): 生成下三角矩阵，对角线及以上为1，其余为0;mask形状为[len_q, len_k]
        mask = torch.tril(torch.ones(len_q, len_k)).type(torch.BoolTensor).to(self.device)

        return mask



