#!/usr/bin/env python3
"""
训练 DnCNN 模型
"""
import torch
import torch.nn as nn
import numpy as np
import scipy.io as sio
import os
import time
from models.dncnn import DnCNN

IMG_SIZE = 128
NUM_ITERATIONS = 200
LEARNING_RATE = 0.001
SAVE_PATH = 'weights/dncnn_sidd.pth'

print("=" * 50)
print("DnCNN 训练")
print("=" * 50)

# 创建模型
device = torch.device('cpu')
model = DnCNN(channels=1, num_of_layers=17).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
criterion = nn.L1Loss()
print(f"  参数量: {sum(p.numel() for p in model.parameters()):,}")

# 加载数据
print("\n加载 SIDD 样本...")
from PIL import Image

mat_noisy = sio.loadmat('data/ValidationNoisyBlocksSrgb.mat')
noisy_img = np.float32(mat_noisy['ValidationNoisyBlocksSrgb'][0, 0]) / 255.0
del mat_noisy

mat_gt = sio.loadmat('data/ValidationGtBlocksSrgb.mat')
clean_img = np.float32(mat_gt['ValidationGtBlocksSrgb'][0, 0]) / 255.0
del mat_gt

# 转灰度
if noisy_img.ndim == 3:
    noisy_gray = np.mean(noisy_img, axis=2)
    clean_gray = np.mean(clean_img, axis=2)
else:
    noisy_gray = noisy_img
    clean_gray = clean_img

# 缩小
noisy_small = np.array(Image.fromarray((noisy_gray * 255).astype(np.uint8)).resize((IMG_SIZE, IMG_SIZE))) / 255.0
clean_small = np.array(Image.fromarray((clean_gray * 255).astype(np.uint8)).resize((IMG_SIZE, IMG_SIZE))) / 255.0

noisy_tensor = torch.from_numpy(noisy_small).float().unsqueeze(0).unsqueeze(0)
clean_tensor = torch.from_numpy(clean_small).float().unsqueeze(0).unsqueeze(0)

# 训练
total_start = time.time()
os.makedirs('weights', exist_ok=True)

print(f"\n开始训练 ({NUM_ITERATIONS} 次迭代)...")
model.train()
for i in range(NUM_ITERATIONS):
    # DnCNN 使用残差学习：预测噪声
    output = model(noisy_tensor)
    # 损失：预测的噪声 vs 实际噪声
    noise = noisy_tensor - clean_tensor
    loss = criterion(output, noise)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if (i + 1) % 50 == 0:
        print(f"  迭代 {i + 1}/{NUM_ITERATIONS}, 损失: {loss.item():.6f}")

# 评估
model.eval()
with torch.no_grad():
    pred_noise = model(noisy_tensor)
    denoised = noisy_tensor - pred_noise
    pred_np = denoised.squeeze().numpy()
    clean_np = clean_tensor.squeeze().numpy()
    mse = np.mean((pred_np - clean_np) ** 2)
    psnr = 10 * np.log10(1.0 / mse) if mse > 0 else 100
    print(f"\n  最终 PSNR: {psnr:.2f} dB")

    torch.save(model.state_dict(), SAVE_PATH)
    print(f"  模型已保存到: {SAVE_PATH}")

total_time = time.time() - total_start
print(f"  总耗时: {total_time:.1f} 秒")
