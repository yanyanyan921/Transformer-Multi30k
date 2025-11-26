import math
import time
import torch
import matplotlib.pyplot as plt
from torch import nn, optim
from nltk.translate.bleu_score import sentence_bleu
import nltk
import warnings

warnings.filterwarnings('ignore')
import os
import random
import numpy as np
from src.data.Dataloader import Tokenizer, Dataset
from src.model.Embedding import TokenEmbedding
from src.model.Transformer import Transformer
from src.model.LayerNorm import LayerNorm
from src.model.PositionWiseFFN import PositionWiseFFN
from src.model.MultiHeadSelfAttention import MultiHeadSelfAttention
from src.model.PositionalEncoding import PositionalEncoding
from src.model.EncoderLayer import EncoderLayer
from src.model.DecoderLayer import DecoderLayer

import math
import time
import torch
import matplotlib.pyplot as plt
from torch import nn, optim
from src.data.Dataloader import Tokenizer, Dataset
from src.model.Transformer import Transformer
from nltk.translate.bleu_score import sentence_bleu
import nltk
import warnings

warnings.filterwarnings('ignore')
import os
import random
import numpy as np
import gc
import copy

# 消融实验配置
ablation_config = {
    'experiments': ['baseline', 'single_head', 'no_pos_encoding'],
    'baseline': {
        'num_heads': 8,
        'use_pos_encoding': True
    },
    'single_head': {
        'num_heads': 1,
        'use_pos_encoding': True
    },
    'no_pos_encoding': {
        'num_heads': 8,
        'use_pos_encoding': False
    }
}

# 基础配置参数
config = {
    'd_model': 512,
    'num_heads': 8,
    'd_ff': 2048,
    'num_layers': 6,
    'dropout': 0.1,
    'max_len': 100,
    'batch_size': 128,
    'total_epoch': 5,  # 暂时减少epoch进行测试
    'init_lr': 0.0005,
    'weight_decay': 0.001,
    'adam_eps': 1e-9,
    'clip': 2.0,
    'min_freq': 2,
    'random_seed': 42,
    'factor': 0.5,
    'patience': 3,
    'ext': ('.en', '.de'),
    'init_token': '<sos>',
    'eos_token': '<eos>',
    'device': torch.device('cuda' if torch.cuda.is_available() else 'cpu')
}


# 设置随机种子
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# 创建目录
def create_directories():
    directories = ['saved', 'result', 'ablation_results']
    for dir_name in directories:
        os.makedirs(dir_name, exist_ok=True)


# 统计模型参数
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# 权重初始化
def initialize_weights(m):
    if hasattr(m, 'weight') and m.weight.dim() > 1:
        nn.init.xavier_uniform_(m.weight.data)
    elif hasattr(m, 'bias') and m.bias is not None:
        nn.init.constant_(m.bias.data, 0)


# 索引转单词
def idx_to_word(indices, vocab):
    if torch.is_tensor(indices):
        indices = indices.cpu().numpy()

    words = []
    for idx in indices:
        if idx == vocab.stoi[config['init_token']]:
            continue
        if idx == vocab.stoi[config['eos_token']]:
            break
        if idx == vocab.stoi['<pad>']:
            continue
        words.append(vocab.itos[idx])

    return ' '.join(words)


# BLEU分数计算
def get_bleu(hypotheses, reference):
    return sentence_bleu([reference], hypotheses,
                         weights=(0.25, 0.25, 0.25, 0.25),
                         smoothing_function=nltk.translate.bleu_score.SmoothingFunction().method1)


# 训练函数
def train(model, iterator, optimizer, criterion, clip, trg_pad_idx):
    model.train()
    epoch_loss = 0
    for i, batch in enumerate(iterator):
        src = batch.src.to(config['device'])
        trg = batch.trg.to(config['device'])

        optimizer.zero_grad()
        output = model(src, trg[:, :-1])
        output_reshape = output.contiguous().view(-1, output.shape[-1])
        trg_flat = trg[:, 1:].contiguous().view(-1)
        loss = criterion(output_reshape, trg_flat)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()
        epoch_loss += loss.item()

        if i % 50 == 0:
            progress = (i / len(iterator)) * 100
            print(f'Step: {i}/{len(iterator)} ({progress:.1f}%) , Loss: {loss.item():.4f}')
    return epoch_loss / len(iterator)


# 评估函数 - 使用当前实验的词汇表
def evaluate(model, iterator, criterion, trg_vocab):
    model.eval()
    epoch_loss = 0
    batch_bleu = []
    with torch.no_grad():
        for i, batch in enumerate(iterator):
            src = batch.src.to(config['device'])
            trg = batch.trg.to(config['device'])
            output = model(src, trg[:, :-1])
            output_reshape = output.contiguous().view(-1, output.shape[-1])
            trg_flat = trg[:, 1:].contiguous().view(-1)
            loss = criterion(output_reshape, trg_flat)
            epoch_loss += loss.item()

            total_bleu = []
            for j in range(trg.size(0)):
                try:
                    trg_words = idx_to_word(batch.trg[j], trg_vocab)
                    output_words = output[j].max(dim=1)[1].cpu()
                    output_words = idx_to_word(output_words, trg_vocab)
                    bleu = get_bleu(hypotheses=output_words.split(), reference=trg_words.split())
                    total_bleu.append(bleu)
                except Exception as e:
                    print(f"样本 {j} BLEU计算失败: {e}")
                    total_bleu.append(0.0)
            total_bleu = sum(total_bleu) / len(total_bleu)
            batch_bleu.append(total_bleu)

    batch_bleu = sum(batch_bleu) / len(batch_bleu)
    batch_bleu = batch_bleu * 100
    return epoch_loss / len(iterator), batch_bleu


# 消融实验模型定义
class SingleHeadTransformer(nn.Module):
    def __init__(self, src_pad_idx, trg_pad_idx, trg_sos_idx, enc_voc_size, dec_voc_size, max_len, d_model, num_heads,
                 d_ff, num_layers, dropout, device):
        super().__init__()
        self.src_pad_idx = src_pad_idx
        self.trg_pad_idx = trg_pad_idx
        self.trg_sos_idx = trg_sos_idx
        self.device = device

        # 使用原始Transformer，但强制num_heads=1
        from src.model.Transformer import Transformer
        self.model = Transformer(
            src_pad_idx=src_pad_idx,
            trg_pad_idx=trg_pad_idx,
            trg_sos_idx=trg_sos_idx,
            enc_voc_size=enc_voc_size,
            dec_voc_size=dec_voc_size,
            max_len=max_len,
            d_model=d_model,
            num_heads=1,  # 强制单头
            d_ff=d_ff,
            num_layers=num_layers,
            dropout=dropout,
            device=device
        )

    def forward(self, src, trg):
        return self.model(src, trg)


class NoPosEncodingTransformer(nn.Module):
    def __init__(self, src_pad_idx, trg_pad_idx, trg_sos_idx, enc_voc_size, dec_voc_size, max_len, d_model, num_heads,
                 d_ff, num_layers, dropout, device):
        super().__init__()
        self.src_pad_idx = src_pad_idx
        self.trg_pad_idx = trg_pad_idx
        self.trg_sos_idx = trg_sos_idx
        self.device = device

        # 使用原始Transformer
        from src.model.Transformer import Transformer
        self.model = Transformer(
            src_pad_idx=src_pad_idx,
            trg_pad_idx=trg_pad_idx,
            trg_sos_idx=trg_sos_idx,
            enc_voc_size=enc_voc_size,
            dec_voc_size=dec_voc_size,
            max_len=max_len,
            d_model=d_model,
            num_heads=num_heads,
            d_ff=d_ff,
            num_layers=num_layers,
            dropout=dropout,
            device=device
        )

        # 移除位置编码,替换整个TransformEmbedding
        self.model.encoder.emb = nn.Sequential(
            TokenEmbedding(enc_voc_size, d_model),
            nn.Dropout(dropout)
        )
        self.model.decoder.emb = nn.Sequential(
            TokenEmbedding(dec_voc_size, d_model),
            nn.Dropout(dropout)
        )

    def forward(self, src, trg):
        return self.model(src, trg)


# 完整的消融实验运行函数
def run_complete_ablation_experiment(exp_name, exp_config):
    """
    完全独立的消融实验，从头开始创建所有组件
    """
    print(f"\n{'=' * 60}")
    print(f"开始消融实验: {exp_name}")
    print(f"{'=' * 60}")

    # 为每个实验设置不同的随机种子
    exp_seed = config['random_seed'] + hash(exp_name) % 1000
    set_seed(exp_seed)

    # 创建实验专属目录
    exp_dir = f'results/ablation_results/{exp_name}'
    os.makedirs(exp_dir, exist_ok=True)

    # 1. 创建独立的数据加载器
    print(f"{exp_name}: 初始化数据加载器...")
    tokenizer = Tokenizer()
    loader = Dataset(
        ext=config['ext'],
        tokenize_en=tokenizer.tokenize_en,
        tokenize_de=tokenizer.tokenize_de,
        init_token=config['init_token'],
        eos_token=config['eos_token']
    )

    # 创建数据集
    train_data, valid_data, test_data = loader.make_dataset()

    # 构建词汇表
    loader.build_vocab(train_data, min_freq=config['min_freq'])

    # 创建独立的数据迭代器
    train_iter, valid_iter, test_iter = loader.make_iter(
        train_data, valid_data, test_data,
        batch_size=config['batch_size'],
        device=config['device']
    )

    # 获取词汇表信息
    src_pad_idx = loader.source.vocab.stoi['<pad>']
    trg_pad_idx = loader.target.vocab.stoi['<pad>']
    trg_sos_idx = loader.target.vocab.stoi[config['init_token']]
    enc_voc_size = len(loader.source.vocab)
    dec_voc_size = len(loader.target.vocab)

    print(f"{exp_name}: 源语言词汇表大小: {enc_voc_size}")
    print(f"{exp_name}: 目标语言词汇表大小: {dec_voc_size}")

    # 2. 创建模型
    print(f"{exp_name}: 初始化模型...")
    if exp_name == 'baseline':
        model = Transformer(
            src_pad_idx=src_pad_idx,
            trg_pad_idx=trg_pad_idx,
            trg_sos_idx=trg_sos_idx,
            enc_voc_size=enc_voc_size,
            dec_voc_size=dec_voc_size,
            max_len=config['max_len'],
            d_model=config['d_model'],
            num_heads=exp_config['num_heads'],
            d_ff=config['d_ff'],
            num_layers=config['num_layers'],
            dropout=config['dropout'],
            device=config['device']
        ).to(config['device'])
    elif exp_name == 'single_head':
        model = SingleHeadTransformer(
            src_pad_idx=src_pad_idx,
            trg_pad_idx=trg_pad_idx,
            trg_sos_idx=trg_sos_idx,
            enc_voc_size=enc_voc_size,
            dec_voc_size=dec_voc_size,
            max_len=config['max_len'],
            d_model=config['d_model'],
            num_heads=exp_config['num_heads'],
            d_ff=config['d_ff'],
            num_layers=config['num_layers'],
            dropout=config['dropout'],
            device=config['device']
        ).to(config['device'])
    elif exp_name == 'no_pos_encoding':
        model = NoPosEncodingTransformer(
            src_pad_idx=src_pad_idx,
            trg_pad_idx=trg_pad_idx,
            trg_sos_idx=trg_sos_idx,
            enc_voc_size=enc_voc_size,
            dec_voc_size=dec_voc_size,
            max_len=config['max_len'],
            d_model=config['d_model'],
            num_heads=exp_config['num_heads'],
            d_ff=config['d_ff'],
            num_layers=config['num_layers'],
            dropout=config['dropout'],
            device=config['device']
        ).to(config['device'])

    # 初始化权重
    model.apply(initialize_weights)
    print(f'{exp_name} 模型有 {count_parameters(model):,} 个可训练参数')

    # 3. 训练组件
    optimizer = optim.Adam(
        model.parameters(),
        lr=config['init_lr'],
        weight_decay=config['weight_decay'],
        eps=config['adam_eps']
    )

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer=optimizer,
        verbose=True,
        factor=config['factor'],
        patience=config['patience']
    )

    criterion = nn.CrossEntropyLoss(ignore_index=trg_pad_idx).to(config['device'])

    # 4. 训练循环
    train_losses, val_losses, bleus = [], [], []
    best_loss = float('inf')

    # 创建实验专属日志文件
    log_file = f'{exp_dir}/training_log.txt'
    with open(log_file, 'w') as f:
        f.write(f"消融实验 - {exp_name}\n")
        f.write("=" * 80 + "\n")
        f.write(
            f"{'Epoch':<6} {'Train_Loss':<10} {'Val_Loss':<10} {'BLEU_Score':<12} {'Train_PPL':<10} {'Val_PPL':<10}\n")
        f.write("-" * 80 + "\n")

    for step in range(config['total_epoch']):
        start_time = time.time()
        train_loss = train(model, train_iter, optimizer, criterion, config['clip'], trg_pad_idx)
        valid_loss, bleu = evaluate(model, valid_iter, criterion, loader.target.vocab)
        end_time = time.time()
        scheduler.step(valid_loss)

        # 记录结果
        train_losses.append(train_loss)
        val_losses.append(valid_loss)
        bleus.append(bleu)

        epoch_mins, epoch_secs = int((end_time - start_time) / 60), int((end_time - start_time) % 60)

        # 保存最佳模型
        if valid_loss < best_loss:
            best_loss = valid_loss
            torch.save(model.state_dict(), f'{exp_dir}/best_model.pt')
            print(f"保存最佳模型，验证损失: {valid_loss:.4f}")

        # 记录到日志文件
        with open(log_file, 'a') as f:
            f.write(
                f"{step + 1:<6} {train_loss:<10.4f} {valid_loss:<10.4f} {bleu:<12.4f} {math.exp(train_loss):<10.2f} {math.exp(valid_loss):<10.2f}\n")

        # 打印进度
        print(f'Epoch: {step + 1} | Time: {epoch_mins}m {epoch_secs}s')
        print(f'\tTrain Loss: {train_loss:.3f} | Train PPL: {math.exp(train_loss):7.3f}')
        print(f'\tVal Loss: {valid_loss:.3f} |  Val PPL: {math.exp(valid_loss):7.3f}')
        print(f'\tBLEU Score: {bleu:.3f}')
        print('-' * 60)

    # 5. 测试集评估
    print(f"\n>>> 在测试集上评估 {exp_name}...")
    model.load_state_dict(torch.load(f'{exp_dir}/best_model.pt', map_location=config['device']))
    test_loss, test_bleu = evaluate(model, test_iter, criterion, loader.target.vocab)

    print(f"{exp_name} - 测试集损失: {test_loss:.4f} | 测试集PPL: {math.exp(test_loss):.2f}")
    print(f"{exp_name} - 测试集BLEU分数: {test_bleu:.2f}")

    # 保存测试结果
    with open(f'{exp_dir}/test_results.txt', 'w') as f:
        f.write(f"{exp_name} 测试结果:\n")
        f.write(f"测试集损失: {test_loss:.4f}\n")
        f.write(f"测试集PPL: {math.exp(test_loss):.2f}\n")
        f.write(f"测试集BLEU分数: {test_bleu:.2f}\n")

    # 6. 绘制训练曲线
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, 'b-', label='Train Loss')
    plt.plot(val_losses, 'r-', label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title(f'{exp_name} - Loss Curves')
    plt.grid(True, alpha=0.3)

    plt.subplot(1, 2, 2)
    plt.plot(bleus, 'g-', label='BLEU Score')
    plt.xlabel('Epoch')
    plt.ylabel('BLEU Score')
    plt.legend()
    plt.title(f'{exp_name} - BLEU Score Progress')
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f'{exp_dir}/training_curves.png', dpi=300, bbox_inches='tight')
    plt.close()

    # 7. 彻底清理
    del model, train_iter, valid_iter, test_iter, optimizer, scheduler, criterion
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return {
        'name': exp_name,
        'best_val_loss': best_loss,
        'final_train_loss': train_losses[-1],
        'final_val_loss': val_losses[-1],
        'final_bleu': bleus[-1],
        'test_loss': test_loss,
        'test_bleu': test_bleu
    }


# 主程序
if __name__ == "__main__":
    # 设置全局随机种子
    set_seed(config['random_seed'])
    create_directories()

    print("开始消融实验...")

    # 运行所有消融实验
    results = []
    for exp_name in ablation_config['experiments']:
        exp_config = ablation_config[exp_name]
        result = run_complete_ablation_experiment(exp_name, exp_config)
        results.append(result)

    # 生成总结报告
    print(f"\n{'=' * 80}")
    print("消融实验总结报告")
    print(f"{'=' * 80}")

    summary_file = '../results/ablation_results/summary_report.txt'
    with open(summary_file, 'w') as f:
        f.write("消融实验总结报告\n")
        f.write("=" * 80 + "\n")
        f.write(f"{'实验名称':<15} {'最佳验证损失':<12} {'最终BLEU':<10} {'测试损失':<10} {'测试BLEU':<10}\n")
        f.write("-" * 80 + "\n")

        for result in results:
            f.write(
                f"{result['name']:<15} {result['best_val_loss']:<12.4f} {result['final_bleu']:<10.2f} {result['test_loss']:<10.4f} {result['test_bleu']:<10.2f}\n")
            print(
                f"{result['name']:<15} {result['best_val_loss']:<12.4f} {result['final_bleu']:<10.2f} {result['test_loss']:<10.4f} {result['test_bleu']:<10.2f}")

    print(f"\n消融实验完成！总结报告已保存到: {summary_file}")