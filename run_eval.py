"""
MASH 评估脚本 — 在 SIDD 验证集上测试去噪效果

优先使用 Hugging Face datasets 库加载数据，
若不可用则自动 fallback 到本地 .mat 文件。
"""

import sys
import os
import time
import subprocess

# ==================== 自动安装依赖 ====================

def ensure_packages():
    """确保必要依赖已安装"""
    required = {'datasets': 'datasets', 'torchvision': 'torchvision', 'scikit-image': 'skimage'}
    missing = []
    for pkg, import_name in required.items():
        try:
            __import__(import_name)
        except ImportError:
            missing.append(pkg)
    if missing:
        print(f"[SETUP] 安装缺失依赖: {missing}")
        subprocess.check_call([sys.executable, '-m', 'pip', 'install'] + missing)

try:
    ensure_packages()
except Exception as e:
    print(f"[SETUP] 自动安装失败: {e}")
    print("[SETUP] 将尝试使用已安装的包继续...")

import numpy as np
import torch
import cv2

# ==================== 数据加载 ====================

USE_HF_DATASETS = False

try:
    from datasets import load_dataset
    USE_HF_DATASETS = True
    print("[DATA] 使用 Hugging Face datasets 加载 SIDD 验证集")
except ImportError:
    print("[DATA] datasets 库不可用，使用本地 .mat 文件加载 SIDD 验证集")

try:
    import torchvision.transforms as T
    HAS_TORCHVISION = True
except ImportError:
    HAS_TORCHVISION = False
    print("[DATA] torchvision 不可用，将手动转换为 tensor")


def load_sidd_samples_hf(num_samples=3):
    """从 Hugging Face 加载 SIDD 验证样本"""
    dataset = load_dataset("Bingsu/SIDD_Validation", split="train")
    samples = []
    for i in range(min(num_samples, len(dataset))):
        noisy_pil = dataset[i]['noisy']
        clean_pil = dataset[i]['clean']
        noisy_np = np.array(noisy_pil).astype(np.uint8)
        clean_np = np.array(clean_pil).astype(np.uint8)
        samples.append({
            'noisy': noisy_np,
            'clean': clean_np,
            'idx': i
        })
        print(f"  样本 {i}: {noisy_np.shape}")
    return samples


def load_sidd_samples_mat(num_samples=3):
    """从本地 .mat 文件加载 SIDD 验证样本"""
    import scipy.io as sio

    noisy_path = 'data/ValidationNoisyBlocksSrgb.mat'
    gt_path = 'data/ValidationGtBlocksSrgb.mat'

    if not os.path.exists(noisy_path):
        raise FileNotFoundError(f"找不到数据文件: {noisy_path}")

    print(f"[DATA] 加载 {noisy_path}")
    noisy_data = sio.loadmat(noisy_path)['ValidationNoisyBlocksSrgb']  # (40, 32, 256, 256, 3)
    print(f"[DATA] 加载 {gt_path}")
    gt_data = sio.loadmat(gt_path)['ValidationGtBlocksSrgb']

    samples = []
    for i in range(min(num_samples, noisy_data.shape[0])):
        # 每张图取第一个 block
        noisy_np = noisy_data[i, 0]  # (256, 256, 3)
        clean_np = gt_data[i, 0]
        samples.append({
            'noisy': noisy_np,
            'clean': clean_np,
            'idx': i
        })
        print(f"  样本 {i}: {noisy_np.shape}")
    return samples


def to_tensor(img_np):
    """numpy uint8 (H,W,C) → tensor (1,C,H,W) float32 [0,1] → GPU"""
    if HAS_TORCHVISION:
        from PIL import Image
        pil = Image.fromarray(img_np)
        tensor = T.ToTensor()(pil).unsqueeze(0)  # (1,C,H,W)
    else:
        img = img_np.astype(np.float32) / 255.0
        tensor = torch.from_numpy(img.transpose(2, 0, 1)).unsqueeze(0)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    return tensor.to(device)


def tensor_to_numpy(tensor):
    """tensor (1,C,H,W) → numpy uint8 (H,W,C)"""
    img = tensor.cpu().squeeze(0).clamp(0, 1).numpy().transpose(1, 2, 0)
    return (img * 255).astype(np.uint8)


# ==================== 指标计算 ====================

def calc_psnr(clean, denoised):
    """计算 PSNR"""
    mse = np.mean((clean.astype(float) - denoised.astype(float)) ** 2)
    if mse == 0:
        return float('inf')
    return 10 * np.log10(255.0 ** 2 / mse)


def calc_ssim_simple(clean, denoised):
    """简化版 SSIM（无需 skimage）"""
    c1 = (0.01 * 255) ** 2
    c2 = (0.03 * 255) ** 2

    clean_f = clean.astype(float)
    denoised_f = denoised.astype(float)

    mu_x = cv2.GaussianBlur(clean_f, (11, 11), 1.5)
    mu_y = cv2.GaussianBlur(denoised_f, (11, 11), 1.5)

    mu_x2 = mu_x ** 2
    mu_y2 = mu_y ** 2
    mu_xy = mu_x * mu_y

    sigma_x2 = cv2.GaussianBlur(clean_f ** 2, (11, 11), 1.5) - mu_x2
    sigma_y2 = cv2.GaussianBlur(denoised_f ** 2, (11, 11), 1.5) - mu_y2
    sigma_xy = cv2.GaussianBlur(clean_f * denoised_f, (11, 11), 1.5) - mu_xy

    ssim_map = ((2 * mu_xy + c1) * (2 * sigma_xy + c2)) / \
               ((mu_x2 + mu_y2 + c1) * (sigma_x2 + sigma_y2 + c2))
    return float(ssim_map.mean())


# ==================== 主流程 ====================

def evaluate():
    sys.path.insert(0, '.')
    from mash_standalone import MASHDenoiser

    print("=" * 60)
    print("MASH SIDD 验证集评估")
    print("=" * 60)

    # 加载数据
    num_samples = 3
    if USE_HF_DATASETS:
        samples = load_sidd_samples_hf(num_samples)
    else:
        samples = load_sidd_samples_mat(num_samples)

    # 初始化 MASH
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\n[MODEL] 设备: {device}")
    denoiser = MASHDenoiser(device=device)

    results = []
    for sample in samples:
        idx = sample['idx']
        noisy_np = sample['noisy']
        clean_np = sample['clean']

        print(f"\n--- 样本 {idx}: {noisy_np.shape} ---")

        # 计算含噪图指标
        noisy_psnr = calc_psnr(clean_np, noisy_np)
        noisy_ssim = calc_ssim_simple(clean_np, noisy_np)
        print(f"  含噪图 PSNR={noisy_psnr:.2f} dB, SSIM={noisy_ssim:.4f}")

        # MASH 去噪
        start = time.time()

        def cb(p, msg):
            if p % 0.25 < 0.05:
                print(f"    [{p:.0%}] {msg}")

        denoised_np = denoiser.denoise(noisy_np, callback=cb)
        elapsed = time.time() - start

        # 计算去噪后指标
        denoised_psnr = calc_psnr(clean_np, denoised_np)
        denoised_ssim = calc_ssim_simple(clean_np, denoised_np)

        print(f"  去噪后 PSNR={denoised_psnr:.2f} dB (+{denoised_psnr - noisy_psnr:.2f})")
        print(f"  去噪后 SSIM={denoised_ssim:.4f} (+{denoised_ssim - noisy_ssim:.4f})")
        print(f"  耗时: {elapsed:.1f}s")

        results.append({
            'idx': idx,
            'noisy_psnr': noisy_psnr,
            'noisy_ssim': noisy_ssim,
            'denoised_psnr': denoised_psnr,
            'denoised_ssim': denoised_ssim,
            'time': elapsed
        })

        # 保存结果图
        os.makedirs('static/results', exist_ok=True)
        cv2.imwrite(f'static/results/eval_{idx}_noisy.png', cv2.cvtColor(noisy_np, cv2.COLOR_RGB2BGR))
        cv2.imwrite(f'static/results/eval_{idx}_denoised.png', cv2.cvtColor(denoised_np, cv2.COLOR_RGB2BGR))
        cv2.imwrite(f'static/results/eval_{idx}_clean.png', cv2.cvtColor(clean_np, cv2.COLOR_RGB2BGR))

    # 汇总
    print("\n" + "=" * 60)
    print("汇总结果")
    print("=" * 60)
    print(f"{'样本':>6} | {'含噪PSNR':>10} | {'去噪PSNR':>10} | {'提升':>8} | {'耗时':>8}")
    print("-" * 60)
    for r in results:
        print(f"  {r['idx']:>4} | {r['noisy_psnr']:>10.2f} | {r['denoised_psnr']:>10.2f} | "
              f"+{r['denoised_psnr'] - r['noisy_psnr']:>6.2f} | {r['time']:>6.1f}s")

    avg_psnr = np.mean([r['denoised_psnr'] for r in results])
    avg_ssim = np.mean([r['denoised_ssim'] for r in results])
    avg_time = np.mean([r['time'] for r in results])
    print("-" * 60)
    print(f"  平均: PSNR={avg_psnr:.2f} dB, SSIM={avg_ssim:.4f}, 耗时={avg_time:.1f}s")


if __name__ == '__main__':
    evaluate()
