import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from model import UNet_n2n_un
from utils import calculate_sliding_std

class MASHDenoiser:
    """
    面向自然场景去噪系统的 MASH 核心引擎。
    已修复：支持任意分辨率长宽比（智能边缘反射 Padding），彻底脱离原版代码对正方形的依赖。
    """
    def __init__(self, device_type='auto', config_overrides=None):
        # 自动探测硬件加速
        if device_type == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device_type)
            
        # 默认算法超参数
        self.config = {
            'lr': 1e-4,
            'num_iterations': 800,       
            'num_predictions': 10,       
            'std_kernel_size': 4,
            'mask_high': 0.8,
            'mask_low': 0.2,
            'mask_medium': 0.5,
            'epsilon_high': 2.5,         
            'epsilon_low': 1.5,          
            'shuffling_iteration': 400,  
            'shuffling_tile_size': 4,
            'masking_threshold': 0.5
        }
        if config_overrides:
            self.config.update(config_overrides)

    def _apply_lps_safe(self, img_tensor, mask_tensor, k):
        """
        纯 PyTorch 实现的局部像素洗牌 (LPS)。
        修复了原版 einops.rearrange 只能处理正方形图像的 Bug。
        """
        b, c, h, w = img_tensor.shape
        # 将图像折叠成 k x k 的块
        img_blocks = img_tensor.view(b, c, h//k, k, w//k, k).permute(0, 1, 2, 4, 3, 5).reshape(c, (h//k)*(w//k), k*k)
        mask_blocks = mask_tensor.view(b, 1, h//k, k, w//k, k).permute(0, 1, 2, 4, 3, 5).reshape(1, (h//k)*(w//k), k*k)
        
        # 在每一个块内部生成随机排列索引
        rand_idx = torch.argsort(torch.rand(c, (h//k)*(w//k), k*k, device=self.device), dim=-1)
        img_shuffled_blocks = torch.gather(img_blocks, dim=-1, index=rand_idx)
        
        # 仅在平坦区域 (mask == 1) 应用洗牌
        mask_blocks_max, _ = torch.max(mask_blocks, dim=-1, keepdim=True)
        img_final_blocks = mask_blocks_max * img_shuffled_blocks + (1 - mask_blocks_max) * img_blocks
        
        # 还原回原始形状
        img_final = img_final_blocks.reshape(b, c, h//k, w//k, k, k).permute(0, 1, 2, 4, 3, 5).reshape(b, c, h, w)
        return img_final

    def _estimate_noise_std(self, img_torch, mask_ratio, callback=None, prefix=""):
        """内部方法：用指定的掩码率进行初步探测训练"""
        _, c, h, w = img_torch.shape
        model = UNet_n2n_un(c, c).to(self.device)
        model.train()
        
        criteron = nn.L1Loss(reduction='mean')
        optimizer = torch.optim.Adam(model.parameters(), lr=self.config['lr'])
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, self.config['num_iterations'])
        
        for i in range(self.config['num_iterations']):
            with torch.no_grad():
                mask = torch.rand(1, c, h, w, device=self.device)
                mask = (mask < mask_ratio).float()

            output = model(mask * img_torch)
            loss = criteron((1 - mask) * output, (1 - mask) * img_torch)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
            
            if callback and i % 20 == 0:
                progress = i / self.config['num_iterations']
                callback(progress, f"{prefix} 探测阶段 (掩码率 {mask_ratio}): {i}/{self.config['num_iterations']}")

        model.eval()
        avg_pred = 0.
        with torch.no_grad():
            for _ in range(self.config['num_predictions']):
                mask = torch.rand(1, c, h, w, device=self.device)
                mask = (mask < mask_ratio).float()
                output = model(mask * img_torch)
                avg_pred += output.detach()
                
        avg_pred = avg_pred / float(self.config['num_predictions'])
        # 计算预测残差的标准差
        estimated_std = torch.std(avg_pred * 255. - img_torch * 255.).item()
        return estimated_std

    def denoise(self, noisy_img_np, force_mask=None, force_lps=None, callback=None):
        # 1. 数据预处理
        if len(noisy_img_np.shape) == 2:
            noisy_img_np = np.expand_dims(noisy_img_np, axis=-1)
        
        h_orig, w_orig, c = noisy_img_np.shape
        noisy_np_chw = np.transpose(noisy_img_np, (2, 0, 1))
        img_L_torch = torch.from_numpy(noisy_np_chw / 255.).unsqueeze(0).float().to(self.device)
        
        # [核心修复1] 将图像宽高 Pad 至 32 的倍数，解决 UNet 下采样时丢失维度的致命问题
        pad_h = (32 - h_orig % 32) % 32
        pad_w = (32 - w_orig % 32) % 32
        if pad_h > 0 or pad_w > 0:
            # 使用 reflect 镜像填充边缘，防止在边缘产生黑边或人工痕迹
            img_L_torch = F.pad(img_L_torch, (0, pad_w, 0, pad_h), mode='reflect')
            
        _, _, h, w = img_L_torch.shape # 获取 Pad 后的尺寸，此时 h 和 w 必然是 32 的倍数
        
        # 2. 探测阶段
        apply_local_shuffling = False
        optimal_mask_ratio = self.config['mask_medium']
        
        if force_mask is not None:
            optimal_mask_ratio = force_mask
            apply_local_shuffling = bool(force_lps)
            if callback: callback(0.0, "用户强制跳过探测，直接进入优化阶段...")
        else:
            if callback: callback(0.0, "开始探测噪声相关性 (epsilon)...")
            std_high = self._estimate_noise_std(img_L_torch, 0.8, callback, prefix="[1/2]")
            std_low = self._estimate_noise_std(img_L_torch, 0.2, callback, prefix="[2/2]")
            
            epsilon = abs(std_high - std_low)
            if callback: callback(0.0, f"噪声相关性探测完成, epsilon={epsilon:.2f}")
            
            if epsilon > self.config['epsilon_high']:
                apply_local_shuffling = True
                optimal_mask_ratio = self.config['mask_high']
            elif epsilon < self.config['epsilon_low']:
                optimal_mask_ratio = self.config['mask_low']
            else:
                optimal_mask_ratio = self.config['mask_medium']

        # 3. 最终优化阶段
        if callback: callback(0.0, f"锁定最优掩码率 {optimal_mask_ratio}, LPS洗牌={'开启' if apply_local_shuffling else '关闭'}")
        
        model = UNet_n2n_un(c, c).to(self.device)
        model.train()
        
        criteron = nn.L1Loss(reduction='mean')
        optimizer = torch.optim.Adam(model.parameters(), lr=self.config['lr'])
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, self.config['num_iterations'])
        upsampler = nn.Upsample(scale_factor=self.config['std_kernel_size'], mode='nearest')
        
        noisy_shuffled_torch = img_L_torch.detach().clone()
        
        for iter_step in range(self.config['num_iterations']):
            with torch.no_grad():
                mask = torch.rand(1, c, h, w, device=self.device)
                mask = (mask < (1. - optimal_mask_ratio)).float()

            output = model(mask * img_L_torch)
            loss = criteron((1 - mask) * output, (1 - mask) * noisy_shuffled_torch)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
            
            if callback and iter_step % 10 == 0:
                progress = iter_step / self.config['num_iterations']
                callback(progress, f"最终优化中... 迭代: {iter_step}/{self.config['num_iterations']}")

            # 触发 LPS
            if iter_step == self.config['shuffling_iteration'] and apply_local_shuffling:
                if callback: callback(progress, "正在执行局部像素洗牌 (LPS) 以解耦相关噪声...")
                avg_tensor = 0.
                with torch.no_grad():
                    for _ in range(self.config['num_predictions']):
                        mask_tmp = torch.rand(1, c, h, w, device=self.device)
                        mask_tmp = (mask_tmp < (1. - optimal_mask_ratio)).float()
                        avg_tensor += model(mask_tmp * img_L_torch)
                
                avg_tensor = (avg_tensor / float(self.config['num_predictions'])) * 255.
                if c > 1:
                    avg_tensor = torch.mean(avg_tensor, dim=1, keepdim=True)
                    
                std_map_torch = calculate_sliding_std(avg_tensor, upsampler, self.config['std_kernel_size'], self.config['std_kernel_size'])
                
                # [核心修复2] 计算掩码并在 GPU 上调用重写的安全的 LPS 算法
                shuffling_mask = (std_map_torch / std_map_torch.max() >= self.config['masking_threshold']).float()
                noisy_shuffled_torch = self._apply_lps_safe(img_L_torch, shuffling_mask, self.config['shuffling_tile_size'])

        # 4. 推理阶段
        if callback: callback(0.95, "正在进行蒙特卡洛多掩码推断...")
        model.eval()
        avg_final = 0.
        with torch.no_grad():
            for _ in range(self.config['num_predictions']):
                mask = torch.rand(1, c, h, w, device=self.device)
                mask = (mask < (1. - optimal_mask_ratio)).float()
                avg_final += model(mask * img_L_torch)
                
        denoised_img_chw = (avg_final / float(self.config['num_predictions'])) * 255.
        
        # [核心修复3] 将推理完成的图像，无缝裁剪回最初加载的真实尺寸 (去除多余的 Padding 边角)
        denoised_img_chw = denoised_img_chw[:, :, :h_orig, :w_orig]
        
        denoised_img_chw = torch.clamp(denoised_img_chw, 0., 255.).cpu().squeeze(0).numpy()
        denoised_img_hwc = np.transpose(denoised_img_chw, (1, 2, 0)).astype(np.uint8)
        
        if callback: callback(1.0, "去噪完成！")
        return denoised_img_hwc