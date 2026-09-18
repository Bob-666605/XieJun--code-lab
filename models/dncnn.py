import torch
import torch.nn as nn
import numpy as np
from .base import BaseDenoiser


class DnCNNNet(nn.Module):
    """
    DnCNN: Beyond a Gaussian Denoiser: Residual Learning of Deep CNN for Image Denoising
    残差学习：学习噪声残差 R(y)，去噪结果 = y - R(y)
    """
    def __init__(self, in_channels=3, out_channels=3, depth=17, features=64):
        super().__init__()

        layers = []
        # 第一层：Conv + ReLU
        layers.append(nn.Conv2d(in_channels, features, 3, padding=1, bias=True))
        layers.append(nn.ReLU(inplace=True))

        # 中间层：Conv + BN + ReLU
        for _ in range(depth - 2):
            layers.append(nn.Conv2d(features, features, 3, padding=1, bias=False))
            layers.append(nn.BatchNorm2d(features))
            layers.append(nn.ReLU(inplace=True))

        # 输出层：Conv（无激活）
        layers.append(nn.Conv2d(features, out_channels, 3, padding=1, bias=True))

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        """输出噪声残差"""
        return self.model(x)


class DnCNN(BaseDenoiser):
    """
    DnCNN 去噪器
    监督训练：需要 noisy-clean 图像对
    推理：输出 = 输入 - 预测的噪声残差
    """
    def __init__(self, config: dict = None):
        super().__init__(config)
        cfg = self.config

        in_ch = cfg.get('in_channels', 3)
        out_ch = cfg.get('out_channels', 3)
        depth = cfg.get('depth', 17)
        features = cfg.get('features', 64)

        self.model = DnCNNNet(in_ch, out_ch, depth, features).to(self.device)

        # 训练参数
        self.lr = cfg.get('lr', 1e-3)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        self.scheduler = torch.optim.lr_scheduler.StepLR(
            self.optimizer, step_size=30, gamma=0.5
        )

    def train(self, noisy_img: np.ndarray, clean_img: np.ndarray = None) -> float:
        """
        监督训练

        Args:
            noisy_img: 含噪图像 (H, W, C) 或 (H, W), [0, 1]
            clean_img: 干净图像（必须提供）

        Returns:
            loss: 训练损失值
        """
        if clean_img is None:
            raise ValueError("DnCNN 需要干净图像进行监督训练")

        self.model.train()
        noisy_tensor = self._to_tensor(noisy_img)
        clean_tensor = self._to_tensor(clean_img)

        # 目标：噪声残差 = noisy - clean
        noise_residual = noisy_tensor - clean_tensor

        # 前向传播
        predicted_residual = self.model(noisy_tensor)

        # L1 损失
        loss = nn.functional.l1_loss(predicted_residual, noise_residual)

        # 反向传播
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()

    def denoise(self, noisy_img: np.ndarray) -> np.ndarray:
        """
        推理去噪

        Args:
            noisy_img: 含噪图像 [0, 1]

        Returns:
            denoised: 去噪图像 [0, 1]
        """
        self.model.eval()

        # 将图像填充到 32 的倍数（确保兼容性）
        h, w = noisy_img.shape[:2]
        pad_h = (32 - h % 32) % 32
        pad_w = (32 - w % 32) % 32

        if pad_h > 0 or pad_w > 0:
            # 使用反射填充
            noisy_padded = np.pad(noisy_img, ((0, pad_h), (0, pad_w)), mode='reflect')
        else:
            noisy_padded = noisy_img

        noisy_tensor = self._to_tensor(noisy_padded)

        with torch.no_grad():
            # 预测噪声残差
            noise_residual = self.model(noisy_tensor)
            # 去噪 = 输入 - 残差
            denoised = noisy_tensor - noise_residual

        # 转回 numpy 并裁剪到原始尺寸
        denoised = self._to_numpy(denoised).clip(0, 1)

        # 裁剪回原始尺寸
        if pad_h > 0 or pad_w > 0:
            denoised = denoised[:h, :w]

        return denoised

    def train_step_scheduler(self):
        """学习率调度器步进"""
        self.scheduler.step()
