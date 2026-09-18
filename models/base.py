import numpy as np
import os

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


class BaseDenoiser:
    """
    去噪方法统一基类
    所有去噪方法（MASH、DnCNN、CBDNet、传统方法）需继承此类并实现接口
    """

    def __init__(self, config: dict = None):
        self.config = config or {}
        self.model = None
        if TORCH_AVAILABLE:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = None

    def train(self, noisy_img: np.ndarray, clean_img: np.ndarray = None) -> float:
        """
        训练方法

        Args:
            noisy_img: 含噪图像, shape (H, W) 或 (H, W, C)
            clean_img: 干净图像（自监督方法可为 None）

        Returns:
            loss: 训练损失值
        """
        raise NotImplementedError

    def denoise(self, noisy_img: np.ndarray) -> np.ndarray:
        """
        推理去噪

        Args:
            noisy_img: 含噪图像, shape (H, W) 或 (H, W, C), 像素值范围 [0, 1] 或 [0, 255]

        Returns:
            denoised: 去噪后图像, 与输入同尺寸同范围
        """
        raise NotImplementedError

    def load_weights(self, path: str):
        """加载模型权重"""
        if not TORCH_AVAILABLE:
            print("PyTorch not available, cannot load weights")
            return
        if self.model is not None and os.path.exists(path):
            state_dict = torch.load(path, map_location=self.device)
            self.model.load_state_dict(state_dict)
            print(f"Loaded weights from {path}")
        else:
            print(f"Weight file not found: {path}")

    def save_weights(self, path: str):
        """保存模型权重"""
        if not TORCH_AVAILABLE:
            print("PyTorch not available, cannot save weights")
            return
        if self.model is not None:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            torch.save(self.model.state_dict(), path)
            print(f"Saved weights to {path}")

    def _to_tensor(self, img: np.ndarray):
        """numpy 图像转 tensor，自动处理维度"""
        if not TORCH_AVAILABLE:
            raise RuntimeError("PyTorch is required for this operation")
        if img.ndim == 2:
            # 灰度图 (H, W) -> (1, 1, H, W)
            tensor = torch.from_numpy(img).float().unsqueeze(0).unsqueeze(0)
        elif img.ndim == 3:
            # RGB 图 (H, W, C) -> (1, C, H, W)
            tensor = torch.from_numpy(img.transpose(2, 0, 1)).float().unsqueeze(0)
        else:
            tensor = torch.from_numpy(img).float()
        return tensor.to(self.device)

    def _to_numpy(self, tensor) -> np.ndarray:
        """tensor 转 numpy 图像"""
        img = tensor.detach().cpu().squeeze().numpy()
        if img.ndim == 3 and img.shape[0] in [1, 3]:
            img = img.transpose(1, 2, 0)
        return img
