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
import os
import random
import numpy as np
import gc

warnings.filterwarnings('ignore')

# 基础配置
BASE_CONFIG = {
    'd_model': 512,
    'num_heads': 8,
    'd_ff': 2048,
    'num_layers': 6,
    'dropout': 0.1,
    'max_len': 100,
    'batch_size': 128,
    'total_epoch': 10,
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

# 超参数实验配置
HYPERPARAM_EXPERIMENTS = {
    'layers': {
        'description': 'Transformer层数对比',
        'values': [2, 4, 6, 8],
        'param_name': 'num_layers'
    },
    'learning_rates': {
        'description': '学习率对比',
        'values': [0.0001, 0.0005, 0.001, 0.005],
        'param_name': 'init_lr'
    },
    'model_dims': {
        'description': '模型维度对比',
        'values': [256, 512, 768],
        'param_name': 'd_model'
    },
    'heads': {
        'description': '注意力头数对比',
        'values': [4, 8, 12, 16],
        'param_name': 'num_heads'
    }
}


def set_seed(seed=42):
    """设置随机种子"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def create_directories():
    """创建必要的目录"""
    base_dir = '../results/hyperparam_results/hyperparam_analysis'
    directories = [base_dir]

    for exp_name in HYPERPARAM_EXPERIMENTS:
        directories.append(f'{base_dir}/{exp_name}')

    for dir_name in directories:
        os.makedirs(dir_name, exist_ok=True)
        print(f"创建目录: {dir_name}")

    return base_dir


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def initialize_weights(m):
    if hasattr(m, 'weight') and m.weight.dim() > 1:
        nn.init.xavier_uniform_(m.weight.data)
    elif hasattr(m, 'bias') and m.bias is not None:
        nn.init.constant_(m.bias.data, 0)


def idx_to_word(indices, vocab):
    if torch.is_tensor(indices):
        indices = indices.cpu().numpy()

    words = []
    for idx in indices:
        if idx == vocab.stoi[BASE_CONFIG['init_token']]:
            continue
        if idx == vocab.stoi[BASE_CONFIG['eos_token']]:
            break
        if idx == vocab.stoi['<pad>']:
            continue
        words.append(vocab.itos[idx])

    return ' '.join(words)


def get_bleu(hypotheses, reference):
    return sentence_bleu([reference], hypotheses,
                         weights=(0.25, 0.25, 0.25, 0.25),
                         smoothing_function=nltk.translate.bleu_score.SmoothingFunction().method1)


def train(model, iterator, optimizer, criterion, clip):
    model.train()
    epoch_loss = 0
    for i, batch in enumerate(iterator):
        src = batch.src.to(BASE_CONFIG['device'])
        trg = batch.trg.to(BASE_CONFIG['device'])

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


def evaluate(model, iterator, criterion, trg_vocab):
    model.eval()
    epoch_loss = 0
    batch_bleu = []
    with torch.no_grad():
        for i, batch in enumerate(iterator):
            src = batch.src.to(BASE_CONFIG['device'])
            trg = batch.trg.to(BASE_CONFIG['device'])
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


def run_single_experiment(exp_name, param_name, param_value, loader, train_data, valid_data, test_data, base_dir):
    """运行单个超参数实验"""
    print(f"\n开始实验: {exp_name} - {param_name}={param_value}")

    # 创建实验配置
    config = BASE_CONFIG.copy()
    config[param_name] = param_value

    # 设置实验特定随机种子
    exp_seed = BASE_CONFIG['random_seed'] + hash(f"{exp_name}_{param_value}") % 1000
    set_seed(exp_seed)

    # 获取词汇表信息 - 使用已经构建好的词汇表
    src_pad_idx = loader.source.vocab.stoi['<pad>']
    trg_pad_idx = loader.target.vocab.stoi['<pad>']
    trg_sos_idx = loader.target.vocab.stoi[config['init_token']]
    enc_voc_size = len(loader.source.vocab)
    dec_voc_size = len(loader.target.vocab)

    # 创建数据迭代器 - 使用已经创建好的数据
    train_iter, valid_iter, test_iter = loader.make_iter(
        train_data, valid_data, test_data,
        batch_size=config['batch_size'],
        device=config['device']
    )

    # 初始化模型
    model = Transformer(
        src_pad_idx=src_pad_idx,
        trg_pad_idx=trg_pad_idx,
        trg_sos_idx=trg_sos_idx,
        enc_voc_size=enc_voc_size,
        dec_voc_size=dec_voc_size,
        max_len=config['max_len'],
        d_model=config['d_model'],
        num_heads=config['num_heads'],
        d_ff=config['d_ff'],
        num_layers=config['num_layers'],
        dropout=config['dropout'],
        device=config['device']
    ).to(config['device'])

    model.apply(initialize_weights)
    param_count = count_parameters(model)
    print(f"模型参数量: {param_count:,}")

    # 训练组件
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

    # 训练循环
    train_losses, val_losses, bleus = [], [], []
    best_val_loss = float('inf')

    for epoch in range(config['total_epoch']):
        start_time = time.time()

        # 训练
        train_loss = train(model, train_iter, optimizer, criterion, config['clip'])

        # 验证
        val_loss, bleu = evaluate(model, valid_iter, criterion, loader.target.vocab)
        scheduler.step(val_loss)

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        bleus.append(bleu)

        # 保存最佳模型
        if val_loss < best_val_loss:
            best_val_loss = val_loss

        epoch_mins, epoch_secs = int((time.time() - start_time) / 60), int((time.time() - start_time) % 60)

        print(f'Epoch: {epoch + 1} | Time: {epoch_mins}m {epoch_secs}s')
        print(f'\tTrain Loss: {train_loss:.3f} | Train PPL: {math.exp(train_loss):7.3f}')
        print(f'\tVal Loss: {val_loss:.3f} |  Val PPL: {math.exp(val_loss):7.3f}')
        print(f'\tBLEU Score: {bleu:.3f}')
        print('-' * 60)

    # 最终测试
    test_loss, test_bleu = evaluate(model, test_iter, criterion, loader.target.vocab)

    print(f"\n>>> 测试结果:")
    print(f"测试集损失: {test_loss:.4f} | 测试集PPL: {math.exp(test_loss):.2f}")
    print(f"测试集BLEU分数: {test_bleu:.2f}")

    # 保存结果
    result = {
        'exp_name': exp_name,
        'param_name': param_name,
        'param_value': param_value,
        'param_count': param_count,
        'best_val_loss': best_val_loss,
        'final_train_loss': train_losses[-1],
        'final_val_loss': val_losses[-1],
        'final_bleu': bleus[-1],
        'test_loss': test_loss,
        'test_bleu': test_bleu,
        'test_ppl': math.exp(test_loss)
    }

    # 绘制训练曲线
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, 'b-', label='Train Loss')
    plt.plot(val_losses, 'r-', label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title(f'{param_name}={param_value} - Loss')
    plt.grid(True, alpha=0.3)

    plt.subplot(1, 2, 2)
    plt.plot(bleus, 'g-', label='BLEU Score')
    plt.xlabel('Epoch')
    plt.ylabel('BLEU Score')
    plt.legend()
    plt.title(f'{param_name}={param_value} - BLEU')
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f'{base_dir}/{exp_name}/{param_name}_{param_value}_curves.png', dpi=300, bbox_inches='tight')
    plt.close()

    # 清理内存
    del model, train_iter, valid_iter, test_iter, optimizer, scheduler, criterion
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return result


def run_hyperparam_analysis(loader, train_data, valid_data, test_data, base_dir):
    """运行所有超参数分析实验"""
    all_results = {}

    for exp_name, exp_config in HYPERPARAM_EXPERIMENTS.items():
        print(f"\n{'=' * 60}")
        print(f"开始: {exp_config['description']}")
        print(f"{'=' * 60}")

        results = []
        param_name = exp_config['param_name']

        for param_value in exp_config['values']:
            try:
                result = run_single_experiment(
                    exp_name, param_name, param_value,
                    loader, train_data, valid_data, test_data, base_dir
                )
                results.append(result)
            except Exception as e:
                print(f"实验 {param_name}={param_value} 失败: {e}")
                continue

        all_results[exp_name] = results

        # 生成实验报告
        generate_experiment_report(exp_name, results, base_dir)

    return all_results


def generate_experiment_report(exp_name, results, base_dir):
    """生成单个实验的报告"""
    if not results:
        return

    # 文本报告
    report_file = f'{base_dir}/{exp_name}/report.txt'
    with open(report_file, 'w') as f:
        f.write(f"{HYPERPARAM_EXPERIMENTS[exp_name]['description']}\n")
        f.write("=" * 60 + "\n")
        f.write(f"{'参数值':<10} {'参数量':<12} {'测试损失':<10} {'测试BLEU':<10} {'测试PPL':<10}\n")
        f.write("-" * 60 + "\n")

        for result in results:
            f.write(f"{result['param_value']:<10} {result['param_count']:<12,} "
                    f"{result['test_loss']:<10.4f} {result['test_bleu']:<10.2f} "
                    f"{result['test_ppl']:<10.2f}\n")

    # 可视化对比
    plt.figure(figsize=(15, 5))

    # 损失对比
    plt.subplot(1, 3, 1)
    param_values = [r['param_value'] for r in results]
    test_losses = [r['test_loss'] for r in results]
    plt.plot(param_values, test_losses, 'bo-', linewidth=2)
    plt.xlabel('Parameter Value')
    plt.ylabel('Test Loss')
    plt.title('Test Loss Comparison')
    plt.grid(True, alpha=0.3)

    # BLEU对比
    plt.subplot(1, 3, 2)
    test_bleus = [r['test_bleu'] for r in results]
    plt.plot(param_values, test_bleus, 'go-', linewidth=2)
    plt.xlabel('Parameter Value')
    plt.ylabel('Test BLEU')
    plt.title('Test BLEU Comparison')
    plt.grid(True, alpha=0.3)

    # 参数量对比
    plt.subplot(1, 3, 3)
    param_counts = [r['param_count'] for r in results]
    plt.bar(range(len(param_counts)), param_counts, color='orange', alpha=0.7)
    plt.xticks(range(len(param_counts)), [str(v) for v in param_values])
    plt.xlabel('Parameter Value')
    plt.ylabel('Parameter Count')
    plt.title('Model Size Comparison')

    plt.tight_layout()
    plt.savefig(f'{base_dir}/{exp_name}/comparison.png', dpi=300, bbox_inches='tight')
    plt.close()

    print(f"实验报告已保存: {report_file}")


def generate_final_summary(all_results, base_dir):
    """生成最终总结报告"""
    summary_file = f'{base_dir}/final_summary.txt'

    with open(summary_file, 'w') as f:
        f.write("超参数分析最终总结\n")
        f.write("=" * 80 + "\n\n")

        for exp_name, results in all_results.items():
            if not results:
                continue

            f.write(f"{HYPERPARAM_EXPERIMENTS[exp_name]['description']}\n")
            f.write("-" * 50 + "\n")

            # 找到最佳配置
            best_by_bleu = max(results, key=lambda x: x['test_bleu'])
            best_by_loss = min(results, key=lambda x: x['test_loss'])

            f.write(f"最佳BLEU: {best_by_bleu['param_value']} "
                    f"(BLEU: {best_by_bleu['test_bleu']:.2f}, "
                    f"参数量: {best_by_bleu['param_count']:,})\n")

            f.write(f"最佳损失: {best_by_loss['param_value']} "
                    f"(Loss: {best_by_loss['test_loss']:.4f}, "
                    f"参数量: {best_by_loss['param_count']:,})\n\n")

    print(f"最终总结已保存: {summary_file}")


def main():
    """主函数"""
    # 设置随机种子
    set_seed(BASE_CONFIG['random_seed'])

    # 创建目录
    base_dir = create_directories()

    print("开始初始化数据加载器...")

    # 初始化数据加载器
    tokenizer = Tokenizer()
    loader = Dataset(
        ext=BASE_CONFIG['ext'],
        tokenize_en=tokenizer.tokenize_en,
        tokenize_de=tokenizer.tokenize_de,
        init_token=BASE_CONFIG['init_token'],
        eos_token=BASE_CONFIG['eos_token']
    )

    # 创建数据集
    train_data, valid_data, test_data = loader.make_dataset()

    # 构建词汇表
    loader.build_vocab(train_data, min_freq=BASE_CONFIG['min_freq'])

    print(f"数据加载完成:")
    print(f"  源语言词汇表: {len(loader.source.vocab)}")
    print(f"  目标语言词汇表: {len(loader.target.vocab)}")
    print(f"  训练样本: {len(train_data)}")
    print(f"  验证样本: {len(valid_data)}")
    print(f"  测试样本: {len(test_data)}")

    # 运行超参数分析
    print(f"\n开始超参数分析...")
    all_results = run_hyperparam_analysis(loader, train_data, valid_data, test_data, base_dir)

    # 生成最终总结
    generate_final_summary(all_results, base_dir)

    print(f"\n超参数分析完成!")
    print(f"所有结果保存在: {base_dir}/")


if __name__ == "__main__":
    main()