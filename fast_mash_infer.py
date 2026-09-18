"""
Fast MASH 在线推理接口
加载预训练权重，8x TTA 几何自集成极速去噪。
"""

import os
import torch
import torch.nn.functional as F
import numpy as np

from model import UNet_n2n_un, generate_blind_spot_mask


def _ssim_map(x, y, window_size=11):
    """可导的逐像素 SSIM 图（无需外部库）"""
    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    channels = x.shape[1]
    kernel = torch.ones(channels, 1, window_size, window_size, device=x.device) / (window_size ** 2)

    mu_x = F.conv2d(x, kernel, groups=channels, padding=window_size // 2)
    mu_y = F.conv2d(y, kernel, groups=channels, padding=window_size // 2)

    mu_x2 = mu_x ** 2
    mu_y2 = mu_y ** 2
    mu_xy = mu_x * mu_y

    sigma_x2 = F.conv2d(x ** 2, kernel, groups=channels, padding=window_size // 2) - mu_x2
    sigma_y2 = F.conv2d(y ** 2, kernel, groups=channels, padding=window_size // 2) - mu_y2
    sigma_xy = F.conv2d(x * y, kernel, groups=channels, padding=window_size // 2) - mu_xy

    ssim_map = ((2 * mu_xy + C1) * (2 * sigma_xy + C2)) / \
               ((mu_x2 + mu_y2 + C1) * (sigma_x2 + sigma_y2 + C2))
    return ssim_map


class FastMASHDenoiser:
    """
    MASH 去噪器：预训练权重 + 4x TTA 几何自集成。

    用法:
        denoiser = FastMASHDenoiser()
        denoised = denoiser.denoise(img_numpy)  # 输入输出均为 [0,1] numpy
    """

    def __init__(self, weight_path='weights/mash_pretrained.pth', device='auto'):
        if device == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)

        self.model = UNet_n2n_un(in_channels=3, out_channels=3).to(self.device)

        if os.path.exists(weight_path):
            state_dict = torch.load(weight_path, map_location=self.device, weights_only=True)
            self.model.load_state_dict(state_dict)
            print(f"[FastMASH] 已加载预训练权重: {weight_path}")
        else:
            print(f"[FastMASH] 警告: 权重文件不存在 ({weight_path})，使用随机初始化")

        self.model.eval()

        # 4 种几何变换及其逆变换
        self.transforms = [
            (lambda t: t,                          lambda t: t),          # 原图
            (lambda t: torch.flip(t, [3]),         lambda t: torch.flip(t, [3])),  # 左右翻转
            (lambda t: torch.flip(t, [2]),         lambda t: torch.flip(t, [2])),  # 上下翻转
            (lambda t: torch.flip(t, [2, 3]),      lambda t: torch.flip(t, [2, 3])),  # 旋转180°
        ]

    def denoise(self, img_numpy, num_passes=3, tau=0.1):
        """
        8x TTA 极速去噪推理

        Args:
            img_numpy: numpy 数组, shape (H, W, 3), 值范围 [0, 1]
            num_passes: 每种 TTA 方向的 MC 推断次数
            tau: 推理时掩蔽率

        Returns:
            denoised: numpy 数组, shape (H, W, 3), 值范围 [0, 1]
        """
        # numpy → tensor (1, 3, H, W)
        if img_numpy.ndim == 2:
            img_numpy = img_numpy[:, :, None].repeat(3, axis=2)
        if img_numpy.shape[2] == 1:
            img_numpy = img_numpy.repeat(3, axis=2)

        tensor = torch.from_numpy(img_numpy.transpose(2, 0, 1)).unsqueeze(0).float().to(self.device)

        _, _, h, w = tensor.shape
        pad_h = (32 - h % 32) % 32
        pad_w = (32 - w % 32) % 32
        if pad_h > 0 or pad_w > 0:
            tensor = F.pad(tensor, (0, pad_w, 0, pad_h), mode='reflect')

        # 8x TTA：每种变换做 num_passes 次 MC 推断
        accumulator = torch.zeros_like(tensor)
        count = 0

        with torch.no_grad():
            for fwd, inv in self.transforms:
                augmented = fwd(tensor)
                for _ in range(num_passes):
                    mask = generate_blind_spot_mask(augmented, tau)
                    pred = self.model(mask * augmented)
                    pred = inv(pred)
                    accumulator += pred
                    count += 1

        output = accumulator / count
        output = torch.clamp(output, 0.0, 1.0)
        output = output[:, :, :h, :w]

        denoised = output.cpu().squeeze(0).numpy().transpose(1, 2, 0)
        return denoised


# ==================== 快捷函数 ====================

_default_denoiser = None

def fast_denoise(img_numpy, **kwargs):
    """一行调用的快捷函数（自动缓存模型）"""
    global _default_denoiser
    if _default_denoiser is None:
        _default_denoiser = FastMASHDenoiser(**{k: v for k, v in kwargs.items()
                                                 if k in ('weight_path', 'device')})
    return _default_denoiser.denoise(img_numpy,
                                     num_passes=kwargs.get('num_passes', 3),
                                     tau=kwargs.get('tau', 0.1))


if __name__ == '__main__':
    import time
    np.random.seed(42)
    test_img = np.random.rand(128, 128, 3).astype(np.float32)
    denoiser = FastMASHDenoiser()
    start = time.time()
    result = denoiser.denoise(test_img)
    elapsed = (time.time() - start) * 1000
    print(f"输入: {test_img.shape}, range=[{test_img.min():.2f}, {test_img.max():.2f}]")
    print(f"输出: {result.shape}, range=[{result.min():.2f}, {result.max():.2f}]")
    print(f"耗时: {elapsed:.0f}ms (4x TTA, 3 MC)")
