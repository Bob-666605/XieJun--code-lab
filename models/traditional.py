import numpy as np
import cv2
from .base import BaseDenoiser


class TraditionalDenoiser(BaseDenoiser):
    """
    传统去噪方法集合
    包含：均值滤波、中值滤波、高斯滤波、双边滤波、小波去噪
    无需训练，直接推理
    """
    def __init__(self, config: dict = None):
        super().__init__(config)
        self.model = None  # 传统方法无需模型

    def train(self, noisy_img: np.ndarray = None, clean_img: np.ndarray = None) -> float:
        """传统方法无需训练"""
        return 0.0

    def denoise(self, noisy_img: np.ndarray, method: str = 'median', **kwargs) -> np.ndarray:
        """
        执行去噪

        Args:
            noisy_img: 含噪图像, uint8 [0, 255] 或 float [0, 1]
            method: 去噪方法
                - 'mean': 均值滤波
                - 'median': 中值滤波
                - 'gaussian': 高斯滤波
                - 'bilateral': 双边滤波
                - 'wavelet': 小波去噪
            **kwargs: 方法特定参数

        Returns:
            denoised: 去噪图像, 与输入同类型同范围
        """
        # 处理输入
        is_float = noisy_img.dtype in [np.float32, np.float64]
        if is_float:
            img = (noisy_img * 255).astype(np.uint8)
        else:
            img = noisy_img.copy()

        # 灰度图处理
        is_gray = img.ndim == 2
        if not is_gray and img.shape[2] == 1:
            img = img.squeeze()
            is_gray = True

        # 执行去噪
        if method == 'mean':
            denoised = self._mean_filter(img, **kwargs)
        elif method == 'median':
            denoised = self._median_filter(img, **kwargs)
        elif method == 'gaussian':
            denoised = self._gaussian_filter(img, **kwargs)
        elif method == 'bilateral':
            denoised = self._bilateral_filter(img, **kwargs)
        elif method == 'wavelet':
            denoised = self._wavelet_denoise(img, **kwargs)
        else:
            raise ValueError(f"Unknown method: {method}")

        # 恢复输入格式
        if is_float:
            denoised = denoised.astype(np.float32) / 255.0
        if is_gray and denoised.ndim == 2:
            pass  # 保持灰度

        return denoised

    def _mean_filter(self, img, ksize=5, **kwargs):
        """均值滤波"""
        return cv2.blur(img, (ksize, ksize))

    def _median_filter(self, img, ksize=5, **kwargs):
        """中值滤波"""
        ksize = ksize if ksize % 2 == 1 else ksize + 1
        return cv2.medianBlur(img, ksize)

    def _gaussian_filter(self, img, ksize=5, sigma=1.0, **kwargs):
        """高斯滤波"""
        ksize = ksize if ksize % 2 == 1 else ksize + 1
        return cv2.GaussianBlur(img, (ksize, ksize), sigma)

    def _bilateral_filter(self, img, d=9, sigma_color=75, sigma_space=75, **kwargs):
        """双边滤波（保边去噪）"""
        return cv2.bilateralFilter(img, d, sigma_color, sigma_space)

    def _wavelet_denoise(self, img, wavelet='db1', level=2, threshold=30, **kwargs):
        """
        小波去噪
        使用 PyWavelets 库
        """
        try:
            import pywt
        except ImportError:
            print("PyWavelets not installed, falling back to median filter")
            return self._median_filter(img)

        # 处理多通道
        if img.ndim == 3:
            channels = []
            for c in range(img.shape[2]):
                channels.append(self._wavelet_single(img[:, :, c], wavelet, level, threshold))
            return np.stack(channels, axis=2)
        else:
            return self._wavelet_single(img, wavelet, level, threshold)

    def _wavelet_single(self, channel, wavelet, level, threshold):
        """单通道小波去噪"""
        import pywt

        # 小波分解
        coeffs = pywt.wavedec2(channel.astype(np.float64), wavelet, level=level)

        # 软阈值处理
        denoised_coeffs = [coeffs[0]]  # 保留低频系数
        for detail in coeffs[1:]:
            denoised_coeffs.append(tuple(
                pywt.threshold(c, value=threshold, mode='soft')
                for c in detail
            ))

        # 小波重构
        denoised = pywt.waverec2(denoised_coeffs, wavelet)
        return np.clip(denoised, 0, 255).astype(np.uint8)
