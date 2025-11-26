import math
import time
import torch
import matplotlib.pyplot as plt
from torch import nn, optim
from src.data.Dataloader import Tokenizer,Dataset
from src.model.Transformer import Transformer
from nltk.translate.bleu_score import sentence_bleu
import nltk
import warnings
warnings.filterwarnings('ignore')
import os
import random
import numpy as np
import argparse
# 添加命令行参数解析
def parse_args():
    parser = argparse.ArgumentParser(description='Transformer 训练脚本')
    parser.add_argument('--seed', type=int, default=42, help='随机种子')
    parser.add_argument('--epochs', type=int, default=30, help='训练轮数')
    parser.add_argument('--batch_size', type=int, default=128, help='批次大小')
    parser.add_argument('--lr', type=float, default=0.0005, help='学习率')
    return parser.parse_args()
# 配置参数
config = {
    # 模型参数
    'd_model': 512,
    'num_heads': 8,
    'd_ff': 2048,
    'num_layers': 6,
    'dropout': 0.1,
    'max_len': 100,

    # 训练参数
    'batch_size': 128,
    'total_epoch': 30,
    'init_lr': 0.0005,
    'weight_decay': 0.001,
    'adam_eps': 1e-9,
    'clip': 2.0,
    'min_freq': 2,
    'random_seed': 42,  # 随机种子


    # 学习率调度器参数
    'factor': 0.5,
    'patience': 3,

    # 数据参数
    'ext': ('.en','.de'),  # 源语言到目标语言
    'init_token': '<sos>',
    'eos_token': '<eos>',

    # 设备
    'device': torch.device('cuda' if torch.cuda.is_available() else 'cpu')
}
# 添加随机种子设置函数
def set_seed(seed=42):
    """设置随机种子以确保实验结果可重现"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"随机种子已设置为: {seed}")

# 在训练开始前创建目录
def create_directories():
    """创建必要的目录"""
    directories = ['saved', 'result']
    for dir_name in directories:
        os.makedirs(dir_name, exist_ok=True)
        print(f"创建目录: {dir_name}")

#计算epoch耗时
def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs

# 统计模型可训练参数数量
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def initialize_weights(m):
    """Xavier初始化权重，让训练更稳定"""
    if hasattr(m, 'weight') and m.weight.dim() > 1:
        nn.init.xavier_uniform_(m.weight.data)
    elif hasattr(m, 'bias') and m.bias is not None:
        nn.init.constant_(m.bias.data, 0)

def idx_to_word(indices, vocab):
    """
    将索引序列转换为单词序列

    参数:
        indices: 索引序列 (Tensor或List[int])
        vocab: torchtext.vocab.Vocab对象

    返回:
        str: 转换后的句子字符串
    """
    if torch.is_tensor(indices):
        indices = indices.cpu().numpy()

    words = []
    for idx in indices:
        # 跳过特殊token和padding
        if idx == vocab.stoi[config['init_token']]:
            continue
        if idx == vocab.stoi[config['eos_token']]:
            break
        if idx == vocab.stoi['<pad>']:
            continue
        words.append(vocab.itos[idx])

    return ' '.join(words)

def get_bleu(hypotheses, reference):
    """
    计算单个句子的BLEU分数

    参数:
        hypotheses: 模型生成的假设文本 (List[str])
        reference: 真实参考文本 (List[str])

    返回:
        float: BLEU-4分数
    """
    # 使用平滑方法避免除零错误
    return sentence_bleu([reference], hypotheses,
                         weights=(0.25, 0.25, 0.25, 0.25),
                         smoothing_function=nltk.translate.bleu_score.SmoothingFunction().method1)

def train(model, iterator, optimizer, criterion, clip):
    model.train()
    epoch_loss = 0
    for i, batch in enumerate(iterator):
        src = batch.src.to(config['device'])  # 源语言 (英语),形状为[batch_size, src_len]
        trg = batch.trg.to(config['device'])  # 目标语言 (德语),形状为[batch_size, trg_len]

        optimizer.zero_grad()
        # output: [batch_size, trg_len - 1, vocab_size]
        output = model(src, trg[:, :-1])  # 前向传播：输入给解码器不包括目标序列的最后一个token
        # output_reshape: [batch_size*(trg_len-1), vocab_size] - 展平用于损失计算
        output_reshape = output.contiguous().view(-1, output.shape[-1])
        # trg: [batch_size*(trg_len-1)] - 目标序列（右移一位），展平
        trg = trg[:, 1:].contiguous().view(-1)
        # 计算损失：比较输出与目标序列（不包括第一个token）,实现了"教师强制"训练
        loss = criterion(output_reshape, trg)
        # 反向传播 + 梯度裁剪 + 优化
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()
        epoch_loss += loss.item()
        # 改进进度显示
        if i % 50 == 0:  # 每50个batch显示一次
            progress = (i / len(iterator)) * 100
            print(f'Step: {i}/{len(iterator)} ({progress:.1f}%) , Loss: {loss.item():.4f}')
    return epoch_loss / len(iterator)


def evaluate(model, iterator, criterion):
    model.eval()
    epoch_loss = 0
    batch_bleu = []
    with torch.no_grad():
        for i, batch in enumerate(iterator):
            # 1. 前向传播和损失计算
            src = batch.src.to(config['device'])
            trg = batch.trg.to(config['device'])
            output = model(src, trg[:, :-1])
            output_reshape = output.contiguous().view(-1, output.shape[-1])
            trg_flat = trg[:, 1:].contiguous().view(-1)
            loss = criterion(output_reshape, trg_flat)
            epoch_loss += loss.item()
            # 2. BLEU分数计算
            total_bleu = []
            for j in range(trg.size(0)):
                try:
                    # batch.trg[j]: 第j个样本的真实目标序列，形状为[trg_len];idx_to_word(): 将索引序列转换为单词序列
                    trg_words = idx_to_word(batch.trg[j], loader.target.vocab)
                    # output[j]: 第j个样本的模型输出，形状为[trg_len - 1, vocab_size]
                    output_words = output[j].max(dim=1)[1].cpu()   #取每个词概率最高的token索引
                    # output_words:形状为[trg_len - 1]的索引序列,再转换为单词序列
                    output_words = idx_to_word(output_words, loader.target.vocab)
                    #hypotheses: 模型生成的假设文本（分词后的列表）;reference: 真实参考文本（分词后的列表）
                    bleu = get_bleu(hypotheses=output_words.split(), reference=trg_words.split())
                    total_bleu.append(bleu)
                except Exception as e:
                    print(f"样本 {j} BLEU计算失败: {e}")
                    total_bleu.append(0.0)  # 添加默认值
            total_bleu = sum(total_bleu) / len(total_bleu)
            batch_bleu.append(total_bleu)

    batch_bleu = sum(batch_bleu) / len(batch_bleu)
    batch_bleu=batch_bleu*100
    return epoch_loss / len(iterator), batch_bleu

def run(total_epoch, best_loss):
    # 在训练开始前创建统一的记录文件（表格格式）
    with open('../results/result/training_log.txt', 'w') as f:
        f.write("训练日志 - Transformer英德翻译\n")
        f.write("=" * 80 + "\n")
        f.write(
            f"{'Epoch':<6} {'Train_Loss':<10} {'Val_Loss':<10} {'BLEU_Score':<12} {'Train_PPL':<10} {'Val_PPL':<10}\n")
        f.write("-" * 80 + "\n")
    # 用于记录训练过程的列表;初始化三个列表来记录每个epoch的训练损失、验证损失和BLEU分数。
    train_losses, val_losses, bleus = [], [], []
    for step in range(total_epoch):
        start_time = time.time()
        train_loss = train(model, train_iter, optimizer, criterion, config['clip'])  #训练一个epoch
        valid_loss, bleu = evaluate(model, valid_iter, criterion)  # 在验证集上评估
        end_time = time.time()
        scheduler.step(valid_loss)
        # 记录当前epoch的结果
        train_losses.append(train_loss)
        val_losses.append(valid_loss)
        bleus.append(bleu)
        epoch_mins, epoch_secs = epoch_time(start_time, end_time)
        # 保存最佳模型
        if valid_loss < best_loss:
            best_loss = valid_loss
            torch.save(model.state_dict(), 'saved/best_model.pt')
            print(f"保存最佳模型，验证损失: {valid_loss:.4f}")

        # 统一记录到单个文件（表格格式）
        with open('../results/result/training_log.txt', 'a') as f:
            f.write(
                f"{step + 1:<6} {train_loss:<10.4f} {valid_loss:<10.4f} {bleu:<12.4f} {math.exp(train_loss):<10.2f} {math.exp(valid_loss):<10.2f}\n")
        # 打印当前epoch的结果
        print(f'Epoch: {step + 1} | Time: {epoch_mins}m {epoch_secs}s')
        print(f'\tTrain Loss: {train_loss:.3f} | Train PPL: {math.exp(train_loss):7.3f}')
        print(f'\tVal Loss: {valid_loss:.3f} |  Val PPL: {math.exp(valid_loss):7.3f}')
        print(f'\tBLEU Score: {bleu:.3f}')
        print('-' * 60)
    # 训练结束后在日志文件末尾添加总结
    with open('../results/result/training_log.txt', 'a') as f:
        f.write("-" * 80 + "\n")
        f.write(f"最佳验证损失: {best_loss:.4f}\n")
        f.write(f"最终训练损失: {train_losses[-1]:.4f}\n")
        f.write(f"最终验证损失: {val_losses[-1]:.4f}\n")
        f.write(f"最终BLEU分数: {bleus[-1]:.2f}\n")

    # 训练结束后直接画图
    plt.figure(figsize=(12, 4))

    # 画损失曲线
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, 'b-', label='Train Loss')
    plt.plot(val_losses, 'r-', label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training and Validation Loss')
    plt.grid(True, alpha=0.3)

    # 画BLEU曲线
    plt.subplot(1, 2, 2)
    plt.plot(bleus, 'g-', label='BLEU Score')
    plt.xlabel('Epoch')
    plt.ylabel('BLEU Score')
    plt.legend()
    plt.title('BLEU Score Progress')
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('result/training_curves.png', dpi=300, bbox_inches='tight')
    plt.show()

    print(f"\nTraining completed! Best validation loss: {best_loss:.4f}")

    return best_loss


def encode_sentence(sentence, src_vocab, src_pad_idx, max_len):
    """
    将输入句子编码为模型可处理的张量
    sentence: 输入句子字符串
    vocab: 词汇表字典 {word: index}
    pad_idx: padding token的索引
    """
    # 1. 文本预处理：转小写、去除首尾空格、分词
    tokens = sentence.lower().strip().split()

    # 2. 将单词转换为索引，未知词用<unk>表示
    idxs = []
    for token in tokens:
        try:
            # 使用 stoi
            idx = src_vocab.stoi[token]
        except KeyError:
            # 如果单词不在词汇表中，使用<unk>
            try:
                idx = src_vocab.stoi['<unk>']
            except KeyError:
                # 如果<unk>也不存在，使用0
                idx = 0
        idxs.append(idx)

    # 3. 序列长度处理：padding或截断
    if len(idxs) < max_len:
        # 填充padding tokens到max_len长度
        idxs += [src_pad_idx] * (max_len - len(idxs))
    else:
        # 截断超过max_len的部分
        idxs = idxs[:max_len]

    # 4. 转换为张量并添加batch维度
    #    形状: [max_len] -> [1, max_len]
    return torch.tensor(idxs).unsqueeze(0).to(config['device'])

def translate(model,sentence,src_vocab,trg_vocab,src_pad_idx,trg_pad_idx,trg_sos_idx):
    """
    执行翻译：将英语句子翻译为德语
    sentence: 输入的英语句子
    """
    #1: 编码源序列
    # 将输入句子编码为张量
    # src_tensor形状: [1, max_len]
    src_tensor = encode_sentence(sentence,src_vocab,src_pad_idx,config['max_len'])

    # 2: 自回归解码
    # 初始化目标序列，以起始符开始
    # trg_indexes: 初始为 [trg_sos_idx]
    trg_indexes = [trg_sos_idx]

    # 逐个token生成，最多生成max_len个token
    for _ in range(config['max_len']):
        # 2.1 准备当前已生成的目标序列
        # trg_tensor形状: [1, current_seq_len]
        trg_tensor = torch.tensor(trg_indexes).unsqueeze(0).to(config['device'])

        # 2.2 通过解码器生成下一个token的概率分布
        # output形状: [1, current_seq_len, dec_voc_size]
        output = model(src_tensor,trg_tensor)

        # 2.3 选择概率最高的token（贪心解码）
        # output[:, -1, :] 形状: [1, dec_voc_size] (最后一个位置的输出)
        # argmax(-1) 形状: [1] (在词汇表维度取最大值)
        pred_token = output[:, -1, :].argmax(-1).item()

        # 2.4 将预测的token添加到序列中
        trg_indexes.append(pred_token)

        # 2.5 停止条件检查
        # 如果预测到padding或结束符，停止生成
        if pred_token == trg_pad_idx or pred_token ==trg_vocab.stoi[config['eos_token']]:
            break

    # 3: 后处理
    # 将索引序列转换回单词序列
    translated_tokens = idx_to_word(trg_indexes,trg_vocab)
    return translated_tokens  #返回德语句子

# 主程序
if __name__ == "__main__":
    args = parse_args()
    # 使用命令行参数覆盖配置
    config['random_seed'] = args.seed
    config['total_epoch'] = args.epochs
    config['batch_size'] = args.batch_size
    config['init_lr'] = args.lr
    # 设置随机种子
    set_seed(config['random_seed'])
    print("开始初始化数据加载器...")
    # 创建必要的目录
    create_directories()
    # 初始化tokenizer和数据加载器
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

    # 创建数据迭代器
    train_iter, valid_iter, test_iter = loader.make_iter(
        train_data, valid_data, test_data,
        batch_size=config['batch_size'],
        device=config['device']
    )

    # 获取词汇表相关信息
    src_pad_idx = loader.source.vocab.stoi['<pad>']
    trg_pad_idx = loader.target.vocab.stoi['<pad>']
    trg_sos_idx = loader.target.vocab.stoi[config['init_token']]
    enc_voc_size = len(loader.source.vocab)
    dec_voc_size = len(loader.target.vocab)

    print(f"源语言词汇表大小: {enc_voc_size}")
    print(f"目标语言词汇表大小: {dec_voc_size}")
    print(f"使用设备: {config['device']}")

    # 初始化模型
    print("初始化Transformer模型...")

    model = Transformer(
        src_pad_idx=src_pad_idx,  # 源语言padding索引
        trg_pad_idx=trg_pad_idx,  # 目标语言padding索引
        trg_sos_idx=trg_sos_idx,  # 目标语言起始符索引
        enc_voc_size=enc_voc_size,  # 编码器词汇表大小
        dec_voc_size=dec_voc_size,  # 解码器词汇表大小
        max_len=config['max_len'],
        d_model=config['d_model'],
        num_heads=config['num_heads'],
        d_ff=config['d_ff'],
        num_layers=config['num_layers'],
        dropout=config['dropout'],
        device=config['device']
    ).to(config['device'])

    print(f'模型有 {count_parameters(model):,} 个可训练参数')
    model.apply(initialize_weights)

    # 确保权重初始化也是确定性的
    torch.manual_seed(config['random_seed'])  # 再次设置确保权重初始化一致

    # 训练组件设置
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

    # 开始训练
    print("开始训练...")
    best_loss = float('inf')
    run(config['total_epoch'], best_loss)

    # 在训练完成后添加翻译测试
    print("\n" + "=" * 60)
    print("训练完成，开始测试集翻译评估...")
    print("=" * 60)

    # 重新加载最佳模型
    try:
        # 加载验证集上表现最好的模型
        model.load_state_dict(torch.load("saved/best_model.pt", map_location=config['device']))
        print("已加载最佳模型进行测试")
    except:
        print("使用当前训练完成的模型进行测试")

    model.eval()

    # 执行翻译测试
    print(">>> 开始翻译测试集英语文本...")

    # 创建保存翻译结果的目录
    os.makedirs('result', exist_ok=True)

    # 打开文件保存翻译结果
    with open('../results/result/test_translations.txt', 'w', encoding='utf-8') as f:
        f.write("测试集前100条翻译结果\n")
        f.write("=" * 80 + "\n\n")

        # 计数器
        total_samples = min(100, len(test_data))
        print(f"开始翻译测试集前{total_samples}条数据...")
        for i in range(total_samples):
            # 获取测试集中的英语原文和德语参考译文
            english_src = ' '.join(test_data.examples[i].src)
            german_ref = ' '.join(test_data.examples[i].trg)
            # 使用模型进行翻译
            try:
                german_translation = translate(model,english_src,loader.source.vocab,loader.target.vocab,src_pad_idx,trg_pad_idx,trg_sos_idx)
                # 写入结果到文件
                f.write(f"样本 {i + 1}:\n")
                f.write(f"英语原文: {english_src}\n")
                f.write(f"参考德语: {german_ref}\n")
                f.write(f"翻译德语: {german_translation}\n")
                # 在控制台显示进度
                if (i + 1) % 10 == 0:
                    print(f"已完成 {i + 1}/{total_samples} 个样本翻译")
                # 显示前3个样本的详细结果
                if i < 3:
                    print(f"\n样本 {i + 1}:")
                    print(f"  英语: {english_src}")
                    print(f"  参考: {german_ref}")
                    print(f"  翻译: {german_translation}")
                    print("-" * 40)
            except Exception as e:
                print(f"翻译样本 {i + 1} 时出错: {e}")
                f.write(f"样本 {i + 1}: 翻译失败 - {str(e)}\n")
                f.write("-" * 50 + "\n\n")
    print(f"\n翻译完成！结果已保存到 'result/test_translations.txt'")
    # 计算测试集BLEU分数
    print("\n>>> 计算测试集整体BLEU分数...")
    test_loss, test_bleu = evaluate(model, test_iter, criterion)
    print(f"测试集损失: {test_loss:.4f} | 测试集PPL: {math.exp(test_loss):.2f}")
    print(f"测试集BLEU分数: {test_bleu:.2f}")
    # 将BLEU分数也保存到文件
    with open('../results/result/test_bleu_score.txt', 'w', encoding='utf-8') as f:
        f.write(f"测试集评估结果:\n")
        f.write(f"测试集损失: {test_loss:.4f}\n")
        f.write(f"测试集PPL: {math.exp(test_loss):.2f}\n")
        f.write(f"测试集BLEU分数: {test_bleu:.2f}\n")

    print("所有测试完成！")
    print("结果文件:")
    print("  - result/test_translations.txt (详细翻译结果)")
    print("  - result/test_bleu_score.txt (整体评估分数)")