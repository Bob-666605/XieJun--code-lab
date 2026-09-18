import numpy as np
import time
from skimage.metrics import peak_signal_noise_ratio
from skimage.metrics import structural_similarity


class MetricsCalculator:
    """
    去噪质量评估指标计算
    支持：PSNR、SSIM、MAE、推理耗时
    """

    @staticmethod
    def calculate_psnr(denoised: np.ndarray, clean: np.ndarray, data_range: float = 255.0) -> float:
        """
        峰值信噪比 (Peak Signal-to-Noise Ratio)
        值越高越好

        Args:
            denoised: 去噪图像
            clean: 干净图像
            data_range: 像素值范围 (255 或 1.0)

        Returns:
            psnr: dB
        """
        denoised = np.clip(denoised, 0, data_range)
        clean = np.clip(clean, 0, data_range)
        return float(peak_signal_noise_ratio(clean, denoised, data_range=data_range))

    @staticmethod
    def calculate_ssim(denoised: np.ndarray, clean: np.ndarray, data_range: float = 255.0) -> float:
        """
        结构相似性 (Structural Similarity Index)
        值越高越好，范围 [0, 1]

        Args:
            denoised: 去噪图像
            clean: 干净图像
            data_range: 像素值范围

        Returns:
            ssim: 结构相似性
        """
        denoised = np.clip(denoised, 0, data_range).astype(np.float64)
        clean = np.clip(clean, 0, data_range).astype(np.float64)

        # 自动判断是否为多通道
        if denoised.ndim == 3 and denoised.shape[2] in [3, 4]:
            return float(structural_similarity(
                denoised, clean,
                multichannel=True,
                channel_axis=2,
                data_range=data_range
            ))
        else:
            return float(structural_similarity(denoised, clean, data_range=data_range))

    @staticmethod
    def calculate_mae(denoised: np.ndarray, clean: np.ndarray) -> float:
        """
        平均绝对误差 (Mean Absolute Error)
        值越低越好

        Args:
            denoised: 去噪图像
            clean: 干净图像

        Returns:
            mae: 平均绝对误差
        """
        return float(np.mean(np.abs(denoised.astype(np.float64) - clean.astype(np.float64))))

    @staticmethod
    def calculate_mse(denoised: np.ndarray, clean: np.ndarray) -> float:
        """
        均方误差 (Mean Squared Error)
        值越低越好

        Args:
            denoised: 去噪图像
            clean: 干净图像

        Returns:
            mse: 均方误差
        """
        return float(np.mean((denoised.astype(np.float64) - clean.astype(np.float64)) ** 2))

    @staticmethod
    def measure_time(func, *args, **kwargs) -> tuple:
        """
        测量函数执行时间

        Args:
            func: 要测量的函数

        Returns:
            (result, elapsed_ms): 函数结果和耗时（毫秒）
        """
        start = time.time()
        result = func(*args, **kwargs)
        elapsed = (time.time() - start) * 1000
        return result, elapsed

    def evaluate_single(self, denoised: np.ndarray, clean: np.ndarray,
                        data_range: float = 255.0) -> dict:
        """
        评估单张图像的所有指标

        Args:
            denoised: 去噪图像
            clean: 干净图像
            data_range: 像素值范围

        Returns:
            dict: 包含所有指标
        """
        return {
            'psnr': self.calculate_psnr(denoised, clean, data_range),
            'ssim': self.calculate_ssim(denoised, clean, data_range),
            'mae': self.calculate_mae(denoised, clean),
            'mse': self.calculate_mse(denoised, clean)
        }

    def evaluate_batch(self, denoised_list: list, clean_list: list,
                       data_range: float = 255.0) -> dict:
        """
        批量评估，返回平均指标和标准差

        Args:
            denoised_list: 去噪图像列表
            clean_list: 干净图像列表
            data_range: 像素值范围

        Returns:
            dict: 包含平均指标和标准差
        """
        psnr_list, ssim_list, mae_list = [], [], []

        for denoised, clean in zip(denoised_list, clean_list):
            psnr_list.append(self.calculate_psnr(denoised, clean, data_range))
            ssim_list.append(self.calculate_ssim(denoised, clean, data_range))
            mae_list.append(self.calculate_mae(denoised, clean))

        return {
            'psnr': np.mean(psnr_list),
            'psnr_std': np.std(psnr_list),
            'ssim': np.mean(ssim_list),
            'ssim_std': np.std(ssim_list),
            'mae': np.mean(mae_list),
            'mae_std': np.std(mae_list),
            'num_images': len(denoised_list)
        }
