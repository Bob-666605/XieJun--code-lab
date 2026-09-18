"""
MASH 离线预训练脚本（增强版）
- 100+ 样本训练
- 动态 tau [0.10, 0.25]
- 复合 Loss: 0.8 * L1 + 0.2 * (1 - SSIM)
"""

import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import RandomCrop, ToTensor, Compose

from model import UNet_n2n_un, generate_blind_spot_mask


# ==================== 配置 ====================

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
PATCH_SIZE = 256
BATCH_SIZE = 4
NUM_WORKERS = 0
LR = 1e-3
NUM_EPOCHS = 50
WEIGHT_SAVE_PATH = 'weights/mash_pretrained.pth'
TAU_MIN = 0.10
TAU_MAX = 0.25


# ==================== SSIM 工具 ====================

def _ssim_map(x, y, window_size=11):
    """可导的逐像素 SSIM 图"""
    C1 = 0.01 ** 2
    C2 = 0.03 ** 2
    channels = x.shape[1]
    kernel = torch.ones(channels, 1, window_size, window_size, device=x.device) / (window_size ** 2)

    mu_x = F.conv2d(x, kernel, groups=channels, padding=window_size // 2)
    mu_y = F.conv2d(y, kernel, groups=channels, padding=window_size // 2)
    mu_x2, mu_y2, mu_xy = mu_x ** 2, mu_y ** 2, mu_x * mu_y

    sigma_x2 = F.conv2d(x ** 2, kernel, groups=channels, padding=window_size // 2) - mu_x2
    sigma_y2 = F.conv2d(y ** 2, kernel, groups=channels, padding=window_size // 2) - mu_y2
    sigma_xy = F.conv2d(x * y, kernel, groups=channels, padding=window_size // 2) - mu_xy

    return ((2 * mu_xy + C1) * (2 * sigma_xy + C2)) / \
           ((mu_x2 + mu_y2 + C1) * (sigma_x2 + sigma_y2 + C2))


def compound_loss(output, target, mask):
    """
    复合损失: 0.8 * L1 + 0.2 * (1 - SSIM)，仅在遮挡区域计算。
    mask: 1=可见, 0=遮挡 → 用 (1-mask) 提取遮挡区域
    """
    masked_out = (1 - mask) * output
    masked_tgt = (1 - mask) * target
    n = torch.sum(1 - mask)

    # L1
    l1 = torch.sum(torch.abs(masked_out - masked_tgt)) / n if n > 0 else torch.tensor(0.0)

    # SSIM（对遮挡区域计算）
    ssim_map = _ssim_map(masked_out, masked_tgt)
    ssim_loss = 1.0 - ssim_map.mean()

    return 0.8 * l1 + 0.2 * ssim_loss


# ==================== 数据集 ====================

class SIDDFromMat(Dataset):
    """从本地 .mat 加载，提取所有 40 张图的多个 block 作为样本"""

    def __init__(self):
        import scipy.io as sio
        noisy_path = 'data/ValidationNoisyBlocksSrgb.mat'
        gt_path = 'data/ValidationGtBlocksSrgb.mat'

        if not os.path.exists(noisy_path):
            raise FileNotFoundError(f"找不到: {noisy_path}")

        print(f"[DATA] 加载 {noisy_path}")
        noisy_data = sio.loadmat(noisy_path)['ValidationNoisyBlocksSrgb']  # (40, 32, 256, 256, 3)
        gt_data = sio.loadmat(gt_path)['ValidationGtBlocksSrgb']

        self.noisy_imgs = []
        self.gt_imgs = []
        num_images = noisy_data.shape[0]  # 40 张
        blocks_per_image = 4  # 每张图取 4 个 block

        for i in range(num_images):
            for b in range(min(blocks_per_image, noisy_data.shape[1])):
                noisy_np = noisy_data[i, b]
                gt_np = gt_data[i, b]
                self.noisy_imgs.append(torch.from_numpy(noisy_np.transpose(2, 0, 1)).float() / 255.0)
                self.gt_imgs.append(torch.from_numpy(gt_np.transpose(2, 0, 1)).float() / 255.0)

        print(f"[DATA] 已加载 {len(self.noisy_imgs)} 个样本 ({num_images} 张图 × {blocks_per_image} blocks)")
        self.crop = RandomCrop(PATCH_SIZE)

    def __len__(self):
        return len(self.noisy_imgs) * 5  # 每个样本多次裁剪

    def __getitem__(self, idx):
        real_idx = idx % len(self.noisy_imgs)
        stacked = torch.stack([self.noisy_imgs[real_idx], self.gt_imgs[real_idx]])
        cropped = self.crop(stacked)
        return cropped[0], cropped[1]


class SIDDFromHF(Dataset):
    """从 Hugging Face 加载"""

    def __init__(self, num_samples=100):
        from datasets import load_dataset
        self.dataset = load_dataset("Bingsu/SIDD_Validation", split="train")
        self.num_samples = min(num_samples, len(self.dataset))
        self.crop = RandomCrop(PATCH_SIZE)
        self.to_tensor = ToTensor()

    def __len__(self):
        return self.num_samples * 5

    def __getitem__(self, idx):
        real_idx = idx % self.num_samples
        sample = self.dataset[real_idx]
        noisy_tensor = self.to_tensor(sample['noisy'])
        clean_tensor = self.to_tensor(sample['clean'])
        stacked = torch.stack([noisy_tensor, clean_tensor])
        cropped = self.crop(stacked)
        return cropped[0], cropped[1]


def get_dataset():
    """自动选择数据加载方式"""
    try:
        from datasets import load_dataset
        print("[DATA] 使用 Hugging Face datasets (100 samples)")
        return SIDDFromHF(num_samples=100)
    except (ImportError, Exception) as e:
        print(f"[DATA] datasets 不可用 ({e})，使用本地 .mat 文件")
        return SIDDFromMat()


# ==================== 训练 ====================

def train():
    print(f"[TRAIN] 设备: {DEVICE}")
    print(f"[TRAIN] 加载 SIDD 数据...")

    dataset = get_dataset()
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True,
                            num_workers=NUM_WORKERS, drop_last=True)

    print(f"[TRAIN] 数据集: {len(dataset)} patches ({len(dataloader)} batches/epoch)")

    model = UNet_n2n_un(in_channels=3, out_channels=3).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"[TRAIN] 模型参数: {total_params / 1e6:.2f}M")
    print(f"[TRAIN] tau 范围: [{TAU_MIN}, {TAU_MAX}], Loss: 0.8*L1 + 0.2*(1-SSIM)")

    for epoch in range(NUM_EPOCHS):
        model.train()
        epoch_loss = 0.0
        num_batches = 0

        for noisy_batch, clean_batch in dataloader:
            noisy_batch = noisy_batch.to(DEVICE)

            # 动态 tau：每个 batch 随机采样
            tau = torch.empty(1).uniform_(TAU_MIN, TAU_MAX).item()

            mask = generate_blind_spot_mask(noisy_batch, tau)
            output = model(mask * noisy_batch)

            loss = compound_loss(output, noisy_batch, mask)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            num_batches += 1

        scheduler.step()
        avg_loss = epoch_loss / max(num_batches, 1)
        lr_now = optimizer.param_groups[0]['lr']
        print(f"  Epoch [{epoch+1}/{NUM_EPOCHS}] Loss: {avg_loss:.6f}  LR: {lr_now:.6f}")

    os.makedirs(os.path.dirname(WEIGHT_SAVE_PATH), exist_ok=True)
    torch.save(model.state_dict(), WEIGHT_SAVE_PATH)
    print(f"[TRAIN] 权重已保存: {WEIGHT_SAVE_PATH}")
    print("[TRAIN] 训练完成！")


if __name__ == '__main__':
    train()
