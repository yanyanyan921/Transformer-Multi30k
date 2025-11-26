from torch import nn
import math
import torch
import torch.nn.functional as F

class MultiHeadSelfAttention(nn.Module):
    def __init__(self,d_model,num_heads):
        super().__init__()
        assert d_model % num_heads == 0  # 确保维度可被头数整除
        self.d_model=d_model
        self.num_heads=num_heads  #num_heads是分头的数量
        self.head_dim=d_model // num_heads  # head_dim是分头后词向量的维度
        self.Q=nn.Linear(self.d_model,self.d_model)  #Q,K,V是实例，实际上创建了两个可训练参数:权重矩阵W,形状 (d_model, d_model),偏置向量b,形状 (d_model,)
        self.K=nn.Linear(self.d_model,self.d_model)  #Q (Query - 查询):表示"我想要什么信息？"
        self.V=nn.Linear(self.d_model,self.d_model)  #K (Key - 键):表示"我有什么信息？"，V (Value - 值):表示"我的实际内容是什么？"
        self.Output_proj=nn.Linear(self.d_model,self.d_model)  #将合并结果投影到输出维度

    def MultiHeadCut(self,input):
        #切分多头,input的形状为(batch,seq_len,d_model)
        #将input的维度从(batch,seq_len,d_model)变换为(batch,seq_len,num_heads,head_dim)，再调整为(batch,num_heads,seq_len,head_dim)
        #这样每个bacth包括了该句子所有的分头数据
        batch= input.size(0)
        #contiguous解决reshape可能的内存连续性问题，contiguous重新分配连续内存块
        #input的形状从(batch,seq_len,d_model)变换为(batch,seq_len,num_heads,head_dim)
        #再调整为(batch, num_heads, seq_len, head_dim)
        input_Cut=input.view(batch,-1,self.num_heads,self.head_dim).transpose(1,2).contiguous()
        return input_Cut

    def AttentionCompute(self,Q_Cut,K_Cut,V_Cut,mask=None):
        #缩放点积注意力，每头独立计算注意力
        # Q_Cut,K_Cut,V_Cut的形状为(batch, num_heads, seq_len, head_dim)
        # mask: 掩码（可选），形状为（batch,seq_len,seq_len）或可广播的形状，0 / False表示遮挡
        assert Q_Cut.shape[-1] == K_Cut.shape[-1]  #保证词向量维度一致
        batch, num_heads, seq_len, head_dim = Q_Cut.shape
        # 1.Q·K^T，计算相似度并缩放
        # attentionscore是注意力分数，形状为（batch,num_heads,seq_len,seq_len）
        #  对于每个头和批次,第(i,j)位置表示第i个查询词与第j个键词的相关性分数
        attentionscore=torch.matmul(Q_Cut,K_Cut.transpose(-2, -1))/math.sqrt(self.head_dim)
        # 2.应用掩码（如果有）—— 遮挡位置设为负无穷，softmax后为0
        # 处理掩码
        if mask is not None:
            # 根据mask的维度进行不同的处理
            if mask.dim() == 4:
                # 4D mask: (batch, num_heads, seq_len, seq_len) 或 (batch, 1, seq_len, seq_len)
                if mask.size(1) == 1:
                    # 扩展头维度
                    mask = mask.repeat(1, num_heads, 1, 1)
                elif mask.size(1) != num_heads:
                    # 如果头数不匹配，使用广播
                    mask = mask.repeat(1, num_heads, 1, 1)
            elif mask.dim() == 3:
                # 3D mask: (batch, seq_len, seq_len)
                mask = mask.unsqueeze(1).repeat(1, num_heads, 1, 1)
            elif mask.dim() == 2:
                # 2D mask: (seq_len, seq_len)
                mask = mask.unsqueeze(0).unsqueeze(0).repeat(batch, num_heads, 1, 1)

            # mask == 0 的位置表示需要被掩码（禁止关注）,将这些位置的注意力分数设为负无穷，softmax后变为0
            attentionscore = attentionscore.masked_fill(mask == 0, -1e9)  # mask==0 的位置填 -1e9

        #3.softmax 归一化，得到注意力权重，形状为（batch,num_heads,seq_len,seq_len）
        Attention = F.softmax(attentionscore, dim=-1)

        #4.权重 × V，得到注意力输出，形状为（batch,num_heads,seq_len,head_dim）
        qkv_attentioned= torch.matmul(Attention,V_Cut)
        return qkv_attentioned  #得到注意力计算后的分头结果

    def MultiHeadCombine(self, input):
        # 合并多头,input的形状为(batch,num_heads,seq_len,head_dim)
        batch=input.size(0)
        # 先交换维度：num_heads 放回最后一维前面,形状为(batch,seq_len,num_heads,head_dim)
        input=input.transpose(1, 2)
        #把维度从(batch,seq_len,num_heads,head_dim)变为(batch,seq_len,num_heads*head_dim)
        #拼接最后两维：num_heads × head_dim → d_model
        input=input.contiguous().view(batch, -1, self.d_model)
        return input

    def forward(self,q,k,v,mask=None):
        # 输入的形状都为（batch，seq_len，d_model）,batch为句子数量,seq_len为一个句子中词的个数,d_model为词向量维度
        assert q.shape[0] == k.shape[0]  #确保q,k,v批次一致
        assert q.shape[0] == v.shape[0]
        assert k.shape[1] == v.shape[1]  #k,v的句子长度也要一致
        # 1. 生成 Q、K、V（线性变换）
        #准备QKV投影,q,k,v的形状为(batch,seq_len,d_model)
        q=self.Q(q)  #调用Q对象的函数,将输入与的Q权重矩阵相乘，加上对应的偏置向量
        k=self.K(k)  #调用K对象的函数,将输入与的K权重矩阵相乘，加上对应的偏置向量
        v=self.V(v)  #调用V对象的函数,将输入与的V权重矩阵相乘，加上对应的偏置向量

        # 2. 拆分多头（并行计算每个头的注意力）
        #q_cut,k_cut,v_cut的形状为(batch, num_heads, seq_len, head_dim)
        q_cut=self.MultiHeadCut(q)
        k_cut=self.MultiHeadCut(k)
        v_cut=self.MultiHeadCut(v)

        #3.计算缩放点积注意力（每个头独立计算）
        qkv_attentioned_cut=self.AttentionCompute(q_cut,k_cut,v_cut,mask)

        # 4. 拼接所有头的输出
        # 最终维度为(batch,seq_len,num_heads*head_dim)=(batch,seq_len,d_model)
        qkv_attentioned = self.MultiHeadCombine(qkv_attentioned_cut)

        #5.输出投影
        output=self.Output_proj.forward(qkv_attentioned)  #调用Output对象的函数,将输入与的Output权重矩阵相乘，加上对应的偏置向量，将不同头的特征进行加权整合
        return output

