markdown
# Transformer 机器翻译项目

基于 PyTorch 实现的完整 Transformer 模型，用于机器翻译任务，包含消融实验和超参数分析。

## 项目特性

- 完整的 Transformer 编码器-解码器架构
- 详细的消融实验（位置编码、多头注意力等）
- 全面的超参数分析（层数、头数、维度等）
- 可视化训练过程和结果分析
- 支持 Multi30k 英德翻译数据集
- 完全可重现的实验设置

## 硬件要求

### 推荐配置
- **GPU**: NVIDIA GeForce RTX 3090(24GB VRAM) 
- **CPU**: 20 vCPU AMD EPYC 7642 48-Core Processor 

## 环境设置

```bash
# 创建conda环境（推荐）
conda create -n transformer python=3.9
conda activate transformer

# 安装项目依赖
pip install -r requirements.txt

# 如果没有requirements.txt，手动安装主要依赖
pip install torch==1.9.0+cu111 torchvision==0.10.0+cu111 torchaudio==0.9.0 -f https://download.pytorch.org/whl/torch_stable.html
pip install numpy==1.26.4 matplotlib==3.5.3 tqdm==4.67.1 nltk==3.8.1
项目结构
text
transformer/
|-- .idea
|-- README.md
|-- __init__.py
|-- __pycache__
|-- requirements.txt
|-- results
|-- saved
|-- scripts
    |-- run.sh                    
|-- src
    |-- .gitignore
    |-- .name
    |-- deployment.xml
    |-- inspectionProfiles
    |-- misc.xml
    |-- modules.xml
    |-- remote-mappings.xml
    |-- wanziying_transformer.iml
    |-- workspace.xml
        |-- profiles_settings.xml
    |-- ablation_results
    |-- hyperparam_analysis
    |-- result
        |-- baseline
            |-- best_model.pt
            |-- test_results.txt
            |-- training_curves.png
            |-- training_log.txt
        |-- no_pos_encoding
            |-- best_model.pt
            |-- test_results.txt
            |-- training_curves.png
            |-- training_log.txt
        |-- single_head
            |-- best_model.pt
            |-- test_results.txt
            |-- training_curves.png
            |-- training_log.txt
        |-- summary_report.txt
        |-- dataset_info.json
        |-- final_summary.txt
        |-- heads
            |-- comparison.png
            |-- num_heads_16_curves.png
            |-- num_heads_4_curves.png
            |-- num_heads_8_curves.png
            |-- report.txt
        |-- hyperparam_config.json
        |-- layers
            |-- comparison.png
            |-- num_layers_2_curves.png
            |-- num_layers_4_curves.png
            |-- num_layers_6_curves.png
            |-- num_layers_8_curves.png
            |-- report.txt
        |-- learning_rates
            |-- comparison.png
            |-- init_lr_0.0001_curves.png
            |-- init_lr_0.0005_curves.png
            |-- init_lr_0.001_curves.png
            |-- init_lr_0.005_curves.png
            |-- report.txt
        |-- model_dims
            |-- comparison.png
            |-- d_model_256_curves.png
            |-- d_model_512_curves.png
            |-- d_model_768_curves.png
            |-- report.txt
        |-- test_bleu_score.txt
        |-- test_translations.txt
        |-- training_curves.png
        |-- training_log.txt
    |-- best_model.pt
    |-- __init__.py
    |-- __pycache__
    |-- ablation.py
    |-- data
        |-- .data
            |-- multi30k
                |-- test2016.de
                |-- test2016.en
                |-- train.de
                |-- train.en
                |-- val.de
                |-- val.en
        |-- Dataloader.py
        |-- __init__.py
        |-- __pycache__
    |-- hyperparam.py
    |-- model
        |-- Decoder.py
        |-- DecoderLayer.py
        |-- Embedding.py
        |-- Encoder.py
        |-- EncoderLayer.py
        |-- LayerNorm.py
        |-- MultiHeadSelfAttention.py
        |-- PositionWiseFFN.py
        |-- PositionalEncoding.py
        |-- Transformer.py
        |-- __init__.py
        |-- __pycache__
    |-- train.py
快速开始
1. 数据准备
项目使用 Multi30k 英德翻译数据集，数据会自动下载到 src/data/.data/multi30k/ 目录。如果下载失败，会优先读取本地数据集进行处理。

2. 训练模型（完全可重现）
bash
# 方法1：使用提供的运行脚本（推荐）
cd /d/Pycharm/transformer/scripts
./run.sh

# 方法2：直接运行训练脚本
cd /d/Pycharm/transformer/src
python train.py --random_seed 42

# 方法3：使用精确的命令行（确保完全重现）
cd /d/Pycharm/transformer/src
python -c "import random; random.seed(42); import torch; torch.manual_seed(42); from train import *; set_seed(42); main()"
3. 运行消融实验
bash
cd /d/Pycharm/transformer/src
python ablation.py --random_seed 42
4. 超参数分析
bash
cd /d/Pycharm/transformer/src
python hyperparam.py --random_seed 42
精确重现实验的命令
bash
# 完整的环境设置和训练命令
cd /d/Pycharm/transformer
conda activate transformer
pip install -r requirements.txt
cd scripts
./run.sh

# 或者直接运行（设置随机种子为42）
cd /d/Pycharm/transformer/src
python -c "
import random
random.seed(42)
import numpy as np
np.random.seed(42)
import torch
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed(42)
    torch.cuda.manual_seed_all(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

from train import *
set_seed(42)
main()
"
训练配置
模型参数
python
{
    'd_model': 512,           # 模型维度
    'num_heads': 8,           # 注意力头数
    'd_ff': 2048,             # 前馈网络维度
    'num_layers': 6,          # 编码器/解码器层数
    'dropout': 0.1,           # Dropout率
    'max_len': 100,           # 最大序列长度
    'batch_size': 128,        # 批次大小
    'total_epoch': 30,        # 训练轮数
    'init_lr': 0.0005,        # 初始学习率
    'weight_decay': 0.001,    # 权重衰减
    'clip': 2.0,              # 梯度裁剪
    'random_seed': 42         # 随机种子
}
训练特性
教师强制训练: 使用目标序列的前n-1个token预测后n-1个token

梯度裁剪: 防止梯度爆炸，提升训练稳定性

学习率调度: ReduceLROnPlateau 自动调整学习率

确定性训练: 设置随机种子确保结果可重现

评估指标
BLEU分数
使用NLTK的BLEU-4评分评估翻译质量：

python
sentence_bleu([reference], hypotheses, weights=(0.25, 0.25, 0.25, 0.25))
困惑度(PPL)
基于交叉熵损失计算：

python
math.exp(loss)
输出文件
训练过程
training_log.txt: 详细的训练日志（表格格式）

training_curves.png: 损失和BLEU分数曲线图

saved/best_model.pt: 最佳模型权重

测试结果
test_translations.txt: 前100条测试样本的详细翻译结果

test_bleu_score.txt: 整体测试集评估分数

使用训练好的模型
python
from src.model.Transformer import Transformer
from src.data.Dataloader import Dataset, Tokenizer

# 加载模型
model = Transformer.load_from_checkpoint('../saved/best_model.pt')
model.eval()

# 进行翻译
translation = translate(model, "Hello world", src_vocab, trg_vocab, src_pad_idx, trg_pad_idx, trg_sos_idx)
print(translation)  # 输出德语翻译
实验结果
消融实验分析
基线模型: 完整 Transformer 架构

无位置编码: 分析位置信息的重要性

单头注意力: 验证多头机制的效果

超参数分析
注意力头数: 4, 8, 16 头对比

网络层数: 2, 4, 6, 8 层对比

模型维度: 256, 512, 768 维度对比

学习率: 0.0001, 0.0005, 0.001, 0.005 对比

贡献
欢迎提交 Issue 和 Pull Request 来改进项目！