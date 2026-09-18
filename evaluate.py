"""
统一评估脚本
支持多种方法在数据集上的批量评估
"""

import os
import argparse
import yaml
import numpy as np
import time
from PIL import Image
from glob import glob

from models import MASHDenoiser, DnCNN, TraditionalDenoiser
from engine import MetricsCalculator


def load_config(method, dataset):
    """加载配置"""
    config_path = f"configs/{method}_{dataset}.yaml"
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    return {}


def get_denoiser(method, dataset):
    """获取去噪器"""
    config = load_config(method, dataset)

    if method == 'mash':
        denoiser = MASHDenoiser(config)
        weight_path = f'weights/mash_{dataset}.pth'
        if os.path.exists(weight_path):
            denoiser.load_weights(weight_path)
        return denoiser

    elif method == 'dncnn':
        denoiser = DnCNN(config)
        weight_path = f'weights/dncnn_{dataset}.pth'
        if os.path.exists(weight_path):
            denoiser.load_weights(weight_path)
        return denoiser

    elif method == 'traditional':
        return TraditionalDenoiser(config)

    else:
        raise ValueError(f"Unknown method: {method}")


def load_fmdd_data(data_dir):
    """加载 FMDD 数据集"""
    raw_paths = sorted(glob(os.path.join(data_dir, 'raw', '*.png')))
    gt_paths = sorted(glob(os.path.join(data_dir, 'gt', '*.png')))

    pairs = []
    for raw_path, gt_path in zip(raw_paths, gt_paths):
        noisy = np.float32(Image.open(raw_path)) / 255.0
        clean = np.float32(Image.open(gt_path)) / 255.0
        pairs.append((noisy, clean, os.path.basename(raw_path)))

    return pairs


def load_sidd_data(data_dir, mode='val'):
    """加载 SIDD 数据集"""
    pairs = []
    scene_dirs = sorted(glob(os.path.join(data_dir, mode, '*')))

    for scene_dir in scene_dirs:
        noisy_path = os.path.join(scene_dir, 'NOISY_SRGB_010.PNG')
        clean_path = os.path.join(scene_dir, 'GT_SRGB_010.PNG')

        if os.path.exists(noisy_path) and os.path.exists(clean_path):
            noisy = np.float32(Image.open(noisy_path)) / 255.0
            clean = np.float32(Image.open(clean_path)) / 255.0
            pairs.append((noisy, clean, os.path.basename(scene_dir)))

    return pairs


def evaluate_method(method, dataset, data_pairs, trad_method='median'):
    """评估单个方法"""
    denoiser = get_denoiser(method, dataset)
    metrics = MetricsCalculator()

    psnr_list, ssim_list, time_list = [], [], []

    for noisy, clean, name in data_pairs:
        # 计时
        start = time.time()

        # 去噪
        if method == 'traditional':
            denoised = denoiser.denoise(noisy, method=trad_method)
        elif method == 'mash':
            if noisy.ndim == 2:
                denoised = denoiser.denoise(noisy)
            else:
                # 对每个通道分别去噪
                channels = []
                for c in range(noisy.shape[2]):
                    denoised_ch = denoiser.denoise(noisy[:, :, c])
                    channels.append(denoised_ch)
                denoised = np.stack(channels, axis=2)
        else:
            denoised = denoiser.denoise(noisy)

        elapsed = (time.time() - start) * 1000

        # 计算指标
        psnr = metrics.calculate_psnr(denoised, clean, data_range=1.0)
        ssim = metrics.calculate_ssim(denoised, clean, data_range=1.0)

        psnr_list.append(psnr)
        ssim_list.append(ssim)
        time_list.append(elapsed)

        print(f"  {name}: PSNR={psnr:.2f}, SSIM={ssim:.4f}, Time={elapsed:.1f}ms")

    return {
        'method': method,
        'psnr': np.mean(psnr_list),
        'psnr_std': np.std(psnr_list),
        'ssim': np.mean(ssim_list),
        'ssim_std': np.std(ssim_list),
        'avg_time': np.mean(time_list),
        'num_images': len(data_pairs)
    }


def main():
    parser = argparse.ArgumentParser(description='Image Denoising Evaluation')
    parser.add_argument('--methods', nargs='+', required=True,
                        choices=['mash', 'dncnn', 'traditional'],
                        help='Methods to evaluate')
    parser.add_argument('--dataset', type=str, required=True,
                        choices=['fmdd', 'sidd'],
                        help='Dataset name')
    parser.add_argument('--trad_method', type=str, default='median',
                        choices=['mean', 'median', 'gaussian', 'bilateral', 'wavelet'],
                        help='Traditional denoising method')
    parser.add_argument('--max_images', type=int, default=None,
                        help='Max number of images to evaluate')
    args = parser.parse_args()

    # 加载数据
    print(f"Loading {args.dataset} dataset...")
    if args.dataset == 'fmdd':
        data_pairs = load_fmdd_data('data/FMDD')
    else:
        data_pairs = load_sidd_data('data/SIDD')

    if args.max_images:
        data_pairs = data_pairs[:args.max_images]

    print(f"Loaded {len(data_pairs)} image pairs")
    print("=" * 60)

    # 评估各方法
    results = []
    for method in args.methods:
        print(f"\nEvaluating {method}...")
        result = evaluate_method(method, args.dataset, data_pairs, args.trad_method)
        results.append(result)

    # 打印结果表格
    print("\n" + "=" * 60)
    print(f"{'Method':<15} {'PSNR (dB)':<15} {'SSIM':<15} {'Time (ms)':<15}")
    print("-" * 60)

    for r in results:
        print(f"{r['method']:<15} {r['psnr']:.2f} ± {r['psnr_std']:.2f}   "
              f"{r['ssim']:.4f} ± {r['ssim_std']:.4f}   {r['avg_time']:.1f}")

    print("=" * 60)

    # 找出最佳方法
    best_psnr = max(results, key=lambda x: x['psnr'])
    best_ssim = max(results, key=lambda x: x['ssim'])
    fastest = min(results, key=lambda x: x['avg_time'])

    print(f"\nBest PSNR: {best_psnr['method']} ({best_psnr['psnr']:.2f} dB)")
    print(f"Best SSIM: {best_ssim['method']} ({best_ssim['ssim']:.4f})")
    print(f"Fastest: {fastest['method']} ({fastest['avg_time']:.1f} ms)")


if __name__ == '__main__':
    main()
