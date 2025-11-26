import torch.nn as nn
from src.model.PositionalEncoding import PositionalEncoding


class TokenEmbedding(nn.Embedding):
    def __init__(self, vocab_size: int, d_model: int):
        """
            vocab_size: 词表大小
            d_model: 嵌入维度
            输入形状为[batch_size, seq_len]，每一行元素表示是一个 token 在 vocab 中的 index
        """
        #将离散的token ID映射到连续的向量空间；形状从 (batch, seq_len) 到 (batch, seq_len, d_model)
        super(TokenEmbedding, self).__init__(vocab_size, d_model, padding_idx=1)

class TransformEmbedding(nn.Module):
    def __init__(self, vocab_size: int, d_model: int, max_len: int, dropout):
        """
        Args:
            vocab_size: 词表大小
            d_model: 嵌入维度，每个词向量的长度
            max_len: 最大序列长度
            device: 设备类型
        """
        super(TransformEmbedding, self).__init__()
        self.token_embed = TokenEmbedding(vocab_size, d_model)  ## 词嵌入层：将token ID映射为d_model维的向量
        self.pos_enc = PositionalEncoding(d_model, max_len,dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x: 输入序列，形状为 (batch, seq_len) 的token IDs
        # 词嵌入,输入: (batch, seq_len) -> 输出: (batch, seq_len, d_model)
        tok_emb = self.token_embed(x)
        pos_enc = self.pos_enc(tok_emb)  #位置编码
        return self.dropout(tok_emb + pos_enc)

