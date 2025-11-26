#!/bin/bash

echo "=== Transformer 英德翻译训练开始 ==="
echo "随机种子: 42"

# 直接运行src目录下的train.py
python ../src/train.py

echo "=== 训练完成 ==="
