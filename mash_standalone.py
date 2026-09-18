"""
MASH Standalone Denoiser — 平衡版（速度+画质）

策略:
- 快速探测（50 iters × 2）→ 判断是否需要 LPS
- Patch 裁剪训练（256x256），tau=0.15 保细节
- LPS 在需要时启用（平坦区域像素洗牌，破坏噪声相关性）
- 3 次 MC 推断融合

用法:
    from mash_standalone import MASHDenoiser
    denoiser = MASHDenoiser()
    denoised = denoiser.denoise(noisy_image)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


# ==================== U-Net 网络 ====================

class UNet_n2n_un(nn.Module):
    """5 级 U-Net，跳跃连接"""

    def __init__(self, in_channels=3, out_channels=3):
        super().__init__()
        nf = 48
        self.en1 = nn.Sequential(
            nn.Conv2d(in_channels, nf, 3, padding=1), nn.LeakyReLU(0.1, True),
            nn.Conv2d(nf, nf, 3, padding=1), nn.LeakyReLU(0.1, True),
            nn.MaxPool2d(2))
        self.en2 = nn.Sequential(
            nn.Conv2d(nf, nf, 3, padding=1), nn.LeakyReLU(0.1, True),
            nn.MaxPool2d(2))
        self.en3 = nn.Sequential(
            nn.Conv2d(nf, nf, 3, padding=1), nn.LeakyReLU(0.1, True),
            nn.MaxPool2d(2))
        self.en4 = nn.Sequential(
            nn.Conv2d(nf, nf, 3, padding=1), nn.LeakyReLU(0.1, True),
            nn.MaxPool2d(2))
        self.en5 = nn.Sequential(
            nn.Conv2d(nf, nf, 3, padding=1), nn.LeakyReLU(0.1, True),
            nn.MaxPool2d(2),
            nn.Conv2d(nf, nf, 3, padding=1), nn.LeakyReLU(0.1, True),
            nn.Upsample(scale_factor=2, mode='nearest'))
        self.de1 = nn.Sequential(
            nn.Conv2d(nf * 2, nf * 2, 3, padding=1), nn.LeakyReLU(0.1, True),
            nn.Conv2d(nf * 2, nf * 2, 3, padding=1), nn.LeakyReLU(0.1, True),
            nn.Upsample(scale_factor=2, mode='nearest'))
        self.de2 = nn.Sequential(
            nn.Conv2d(nf * 3, nf * 2, 3, padding=1), nn.LeakyReLU(0.1, True),
            nn.Conv2d(nf * 2, nf * 2, 3, padding=1), nn.LeakyReLU(0.1, True),
            nn.Upsample(scale_factor=2, mode='nearest'))
        self.de3 = nn.Sequential(
            nn.Conv2d(nf * 3, nf * 2, 3, padding=1), nn.LeakyReLU(0.1, True),
            nn.Conv2d(nf * 2, nf * 2, 3, padding=1), nn.LeakyReLU(0.1, True),
            nn.Upsample(scale_factor=2, mode='nearest'))
        self.de4 = nn.Sequential(
            nn.Conv2d(nf * 3, nf * 2, 3, padding=1), nn.LeakyReLU(0.1, True),
            nn.Conv2d(nf * 2, nf * 2, 3, padding=1), nn.LeakyReLU(0.1, True),
            nn.Upsample(scale_factor=2, mode='nearest'))
        self.out = nn.Sequential(
            nn.Conv2d(nf * 2 + in_channels, 64, 3, padding=1), nn.LeakyReLU(0.1, True),
            nn.Conv2d(64, 32, 3, padding=1), nn.LeakyReLU(0.1, True),
            nn.Conv2d(32, out_channels, 3, padding=1))

    def forward(self, x):
        e1 = self.en1(x)
        e2 = self.en2(e1)
        e3 = self.en3(e2)
        e4 = self.en4(e3)
        u5 = self.en5(e4)
        u4 = self.de1(torch.cat([u5, e4], 1))
        u3 = self.de2(torch.cat([u4, e3], 1))
        u2 = self.de3(torch.cat([u3, e2], 1))
        u1 = self.de4(torch.cat([u2, e1], 1))
        return self.out(torch.cat([u1, x], 1))


# ==================== 工具函数 ====================

def _box_smooth(x, ksize):
    ch = x.shape[1]
    conv = nn.Conv2d(ch, ch, ksize, stride=ksize, padding=0, bias=False, groups=ch)
    with torch.no_grad():
        conv.weight.fill_(1.0 / (ksize * ksize))
    return conv(x)


def _sliding_std(img, ksize):
    mean = _box_smooth(img, ksize)
    mean_up = F.interpolate(mean, size=img.shape[2:], mode='nearest')
    var = _box_smooth((img - mean_up) ** 2, ksize)
    var_up = F.interpolate(var, size=img.shape[2:], mode='nearest')
    return var_up.sqrt()


# ==================== MASH 去噪器 ====================

class MASHDenoiser:
    """
    MASH 平衡版：快速探测 + patch 训练 + 条件 LPS + 全图推理

    输入: numpy uint8 (H,W) 或 (H,W,C)
    输出: numpy uint8 (H,W,C)
    """

    def __init__(self, device='auto', config=None):
        if device == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)

        self.cfg = {
            'lr': 1e-3,
            'num_iterations': 350,       # 训练迭代次数
            'probe_iterations': 50,      # 探测迭代数（大幅削减）
            'num_predictions': 3,        # MC 推断次数
            'tau': 0.15,                 # 固定掩码率
            'patch_size': 256,           # 训练 patch 尺寸
            'epsilon_high': 2.0,         # 触发 LPS 的阈值
            'shuffling_iteration': 175,  # LPS 触发时机（50% 处）
            'shuffling_tile_size': 4,
            'std_kernel_size': 4,
            'masking_threshold': 0.5,
        }
        if config:
            self.cfg.update(config)

    def _to_tensor(self, img):
        """numpy → padded float32 tensor"""
        if img.dtype == np.uint8:
            img = img.astype(np.float32) / 255.0
        elif img.max() > 1.0:
            img = img.astype(np.float32) / 255.0

        if img.ndim == 2:
            img = img[:, :, None]

        h_orig, w_orig, c = img.shape
        tensor = torch.from_numpy(img.transpose(2, 0, 1)).unsqueeze(0).float().to(self.device)

        pad_h = (32 - h_orig % 32) % 32
        pad_w = (32 - w_orig % 32) % 32
        if pad_h > 0 or pad_w > 0:
            tensor = F.pad(tensor, (0, pad_w, 0, pad_h), mode='reflect')

        return tensor, h_orig, w_orig

    def _probe_noise(self, img_tensor, tau):
        """快速探测：50 次迭代估算噪声空间相关性"""
        _, c, h, w = img_tensor.shape
        model = UNet_n2n_un(c, c).to(self.device)
        model.train()
        opt = torch.optim.Adam(model.parameters(), lr=self.cfg['lr'])

        for _ in range(self.cfg['probe_iterations']):
            mask = (torch.rand(1, c, h, w, device=self.device) < tau).float()
            out = model(mask * img_tensor)
            loss = F.mse_loss((1 - mask) * out, (1 - mask) * img_tensor)
            opt.zero_grad()
            loss.backward()
            opt.step()

        model.eval()
        avg = torch.zeros_like(img_tensor)
        with torch.no_grad():
            for _ in range(3):
                mask = (torch.rand(1, c, h, w, device=self.device) < tau).float()
                avg += model(mask * img_tensor)
        avg /= 3

        return torch.std(avg * 255.0 - img_tensor * 255.0).item()

    def _apply_lps(self, img_tensor, shuffling_mask, tile_size):
        """局部像素洗牌"""
        b, c, h, w = img_tensor.shape
        k = tile_size
        img_blocks = img_tensor.view(b, c, h // k, k, w // k, k) \
            .permute(0, 1, 2, 4, 3, 5).reshape(c, (h // k) * (w // k), k * k)
        mask_blocks = shuffling_mask.view(b, 1, h // k, k, w // k, k) \
            .permute(0, 1, 2, 4, 3, 5).reshape(1, (h // k) * (w // k), k * k)

        rand_idx = torch.argsort(
            torch.rand(c, (h // k) * (w // k), k * k, device=self.device), dim=-1)
        shuffled = torch.gather(img_blocks, dim=-1, index=rand_idx)

        m, _ = torch.max(mask_blocks, dim=-1, keepdim=True)
        result = m * shuffled + (1 - m) * img_blocks

        return result.reshape(b, c, h // k, w // k, k, k) \
            .permute(0, 1, 2, 4, 3, 5).reshape(b, c, h, w)

    def denoise(self, noisy_img, callback=None):
        """
        完整去噪流水线：快速探测 → patch 训练（条件LPS）→ 全图 MC 推断

        Args:
            noisy_img: numpy uint8 (H,W) 或 (H,W,C)
            callback: callback(progress, msg)

        Returns:
            numpy uint8 (H,W,C)
        """
        if callback:
            callback(0.0, "预处理...")

        img_tensor, h_orig, w_orig = self._to_tensor(noisy_img)
        _, c, h, w = img_tensor.shape
        tau = self.cfg['tau']

        # ===== 快速探测（50 iters × 2）=====
        apply_lps = False
        if callback:
            callback(0.02, "快速探测噪声...")

        std_high = self._probe_noise(img_tensor, 0.8)
        std_low = self._probe_noise(img_tensor, 0.2)
        epsilon = abs(std_high - std_low)

        if epsilon > self.cfg['epsilon_high']:
            apply_lps = True

        if callback:
            callback(0.06, f"ε={epsilon:.2f}, LPS={'ON' if apply_lps else 'OFF'}")

        # ===== Patch 训练 =====
        model = UNet_n2n_un(c, c).to(self.device)
        model.train()
        opt = torch.optim.Adam(model.parameters(), lr=self.cfg['lr'])
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, self.cfg['num_iterations'])

        patch_size = self.cfg['patch_size']
        noisy_target = img_tensor.detach().clone()
        num_iters = self.cfg['num_iterations']

        for step in range(num_iters):
            # 随机裁剪 256x256 patch
            if h > patch_size and w > patch_size:
                y = np.random.randint(0, h - patch_size + 1)
                x = np.random.randint(0, w - patch_size + 1)
                patch_in = img_tensor[:, :, y:y+patch_size, x:x+patch_size]
                patch_tgt = noisy_target[:, :, y:y+patch_size, x:x+patch_size]
            else:
                patch_in = img_tensor
                patch_tgt = noisy_target

            ph, pw = patch_in.shape[2], patch_in.shape[3]
            mask = (torch.rand(1, c, ph, pw, device=self.device) < tau).float()

            out = model(mask * patch_in)
            loss = F.mse_loss((1 - mask) * out, (1 - mask) * patch_tgt)

            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()

            if callback and step % 50 == 0:
                pct = 0.08 + 0.72 * step / num_iters
                callback(pct, f"训练 {step}/{num_iters}")

            # LPS 触发（仅在探测发现强噪声相关时）
            if step == self.cfg['shuffling_iteration'] and apply_lps:
                if callback:
                    callback(0.5, "LPS 洗牌...")
                model.eval()
                avg_pred = torch.zeros_like(img_tensor)
                with torch.no_grad():
                    for _ in range(3):
                        m = (torch.rand(1, c, h, w, device=self.device) < tau).float()
                        avg_pred += model(m * img_tensor)
                avg_pred = (avg_pred / 3) * 255.0

                if c > 1:
                    avg_pred = avg_pred.mean(dim=1, keepdim=True)

                std_map = _sliding_std(avg_pred, self.cfg['std_kernel_size'])
                shuffling_mask = (std_map / (std_map.max() + 1e-8) >= self.cfg['masking_threshold']).float()
                noisy_target = self._apply_lps(img_tensor, shuffling_mask,
                                               self.cfg['shuffling_tile_size'])
                model.train()

        # ===== 全图 MC 推断（3 次）=====
        if callback:
            callback(0.88, "全图多掩码推断...")

        model.eval()
        avg_final = torch.zeros_like(img_tensor)
        with torch.no_grad():
            for _ in range(self.cfg['num_predictions']):
                mask = (torch.rand(1, c, h, w, device=self.device) < tau).float()
                avg_final += model(mask * img_tensor)
        avg_final /= self.cfg['num_predictions']

        result = (avg_final[:, :, :h_orig, :w_orig] * 255.0).clamp(0, 255)
        result_np = result.cpu().squeeze(0).numpy().transpose(1, 2, 0)

        if result_np.shape[2] == 1:
            result_np = result_np[:, :, 0]

        if callback:
            callback(1.0, "去噪完成")

        return result_np.astype(np.uint8)


def denoise_image(noisy_img, **kwargs):
    return MASHDenoiser(**kwargs).denoise(noisy_img)


if __name__ == '__main__':
    np.random.seed(42)
    clean = np.random.randint(100, 200, (128, 128, 3), dtype=np.uint8)
    noise = np.random.normal(0, 25, clean.shape).astype(np.float32)
    noisy = np.clip(clean.astype(np.float32) + noise, 0, 255).astype(np.uint8)

    def progress(p, msg):
        print(f"  [{p:.0%}] {msg}")

    print("MASH 平衡版测试")
    denoiser = MASHDenoiser()
    result = denoiser.denoise(noisy, callback=progress)
    print(f"  输出: {result.shape}, dtype={result.dtype}")

    mse = np.mean((clean.astype(float) - result.astype(float)) ** 2)
    psnr = 10 * np.log10(255.0 ** 2 / mse) if mse > 0 else float('inf')
    print(f"  PSNR: {psnr:.2f} dB")
