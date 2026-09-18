import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk
import numpy as np
import threading
import os
import time
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
import torch

# 导入我们第一阶段封装的 MASH 核心推理引擎
from mash_core import MASHDenoiser

class NoiseEliminationSystem:
    def __init__(self, root):
        self.root = root
        self.root.title("自然场景图像噪声消除系统 (基于 MASH 深度学习算法)")
        self.root.geometry("1100x700")
        self.root.minsize(900, 600)
        
        # 核心数据
        self.clean_img_np = None   # 干净原图（用于计算 PSNR/SSIM）
        self.noisy_img_np = None   # 含噪图（输入模型的实际数据）
        self.denoised_img_np = None # 去噪结果图
        self.display_size = (450, 450) # 界面预览图片的大小限制
        
        # UI 变量
        self.mask_strategy = tk.StringVar(value="auto") # auto, 0.2, 0.5, 0.8
        self.lps_enabled = tk.BooleanVar(value=True)    # 是否启用局部像素洗牌
        self.noise_level = tk.IntVar(value=25)          # 模拟噪声强度
        
        self._build_ui()
        
    def _build_ui(self):
        """构建图形交互界面"""
        # --- 左侧控制面板 ---
        control_frame = ttk.LabelFrame(self.root, text="控制面板", width=250)
        control_frame.pack(side=tk.LEFT, fill=tk.Y, padx=10, pady=10)
        control_frame.pack_propagate(False) # 固定宽度
        
        # 1. 图像操作区
        file_frame = ttk.LabelFrame(control_frame, text="图像操作")
        file_frame.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Button(file_frame, text="1. 加载自然场景图像", command=self.load_image).pack(fill=tk.X, padx=5, pady=5)
        
        # 模拟噪声配置
        noise_subframe = ttk.Frame(file_frame)
        noise_subframe.pack(fill=tk.X, padx=5, pady=5)
        ttk.Label(noise_subframe, text="模拟噪声 (σ):").pack(side=tk.LEFT)
        ttk.Spinbox(noise_subframe, from_=0, to=100, textvariable=self.noise_level, width=5).pack(side=tk.LEFT, padx=5)
        ttk.Button(file_frame, text="2. 添加模拟噪声 (测试用)", command=self.add_synthetic_noise).pack(fill=tk.X, padx=5, pady=5)
        
        # 2. 算法参数区
        algo_frame = ttk.LabelFrame(control_frame, text="MASH 算法参数配置")
        algo_frame.pack(fill=tk.X, padx=5, pady=10)
        
        ttk.Label(algo_frame, text="掩码比例 (Mask Ratio):").pack(anchor=tk.W, padx=5, pady=(5, 0))
        ttk.Radiobutton(algo_frame, text="自适应探测 (Auto)", variable=self.mask_strategy, value="auto").pack(anchor=tk.W, padx=15)
        ttk.Radiobutton(algo_frame, text="低掩码 (0.2 - iid噪声)", variable=self.mask_strategy, value="0.2").pack(anchor=tk.W, padx=15)
        ttk.Radiobutton(algo_frame, text="中掩码 (0.5 - 混合噪声)", variable=self.mask_strategy, value="0.5").pack(anchor=tk.W, padx=15)
        ttk.Radiobutton(algo_frame, text="高掩码 (0.8 - 相关噪声)", variable=self.mask_strategy, value="0.8").pack(anchor=tk.W, padx=15)
        
        ttk.Checkbutton(algo_frame, text="启用局部像素洗牌 (LPS)", variable=self.lps_enabled).pack(anchor=tk.W, padx=5, pady=10)
        
        # 3. 执行区
        exec_frame = ttk.Frame(control_frame)
        exec_frame.pack(fill=tk.X, padx=5, pady=10)
        
        self.btn_run = ttk.Button(exec_frame, text="3. 开始去噪处理", command=self.start_denoising)
        self.btn_run.pack(fill=tk.X, pady=5)
        
        self.btn_save = ttk.Button(exec_frame, text="4. 保存去噪结果", command=self.save_result, state=tk.DISABLED)
        self.btn_save.pack(fill=tk.X, pady=5)
        
        # --- 右侧图像与结果展示区 ---
        display_frame = ttk.Frame(self.root)
        display_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # 图片对比区
        images_frame = ttk.Frame(display_frame)
        images_frame.pack(fill=tk.BOTH, expand=True)
        
        self.lbl_noisy_img = ttk.Label(images_frame, text="含噪图像预览", background="#e0e0e0", anchor=tk.CENTER)
        self.lbl_noisy_img.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5)
        
        self.lbl_denoised_img = ttk.Label(images_frame, text="去噪结果预览", background="#e0e0e0", anchor=tk.CENTER)
        self.lbl_denoised_img.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5)
        
        # 状态与指标区
        status_frame = ttk.LabelFrame(display_frame, text="处理状态与评估指标")
        status_frame.pack(fill=tk.X, side=tk.BOTTOM, pady=10)
        
        self.lbl_metrics = ttk.Label(status_frame, text="PSNR: N/A  |  SSIM: N/A", font=("Helvetica", 11, "bold"), foreground="blue")
        self.lbl_metrics.pack(anchor=tk.W, padx=10, pady=5)
        
        self.progress_var = tk.DoubleVar()
        self.progressbar = ttk.Progressbar(status_frame, variable=self.progress_var, maximum=100)
        self.progressbar.pack(fill=tk.X, padx=10, pady=2)
        
        self.lbl_status = ttk.Label(status_frame, text="系统就绪。请加载图像。")
        self.lbl_status.pack(anchor=tk.W, padx=10, pady=(2, 5))

    def update_status(self, progress_val, text):
        """线程安全的 UI 更新回调方法"""
        self.root.after(0, self._update_status_ui, progress_val, text)
        
    def _update_status_ui(self, progress_val, text):
        self.progress_var.set(progress_val * 100)
        self.lbl_status.config(text=text)

    def load_image(self):
        file_path = filedialog.askopenfilename(filetypes=[("Image Files", "*.png *.jpg *.jpeg *.bmp")])
        if not file_path:
            return
        
        try:
            pil_img = Image.open(file_path).convert("RGB")
            self.clean_img_np = np.array(pil_img)
            self.noisy_img_np = self.clean_img_np.copy() # 默认假设加载的图即为输入图
            
            self._display_image(self.noisy_img_np, self.lbl_noisy_img)
            self.lbl_denoised_img.config(image='', text="等待处理...")
            self.btn_save.config(state=tk.DISABLED)
            self.lbl_metrics.config(text="PSNR: N/A  |  SSIM: N/A")
            self.update_status(0, f"成功加载图像: {os.path.basename(file_path)}")
        except Exception as e:
            messagebox.showerror("加载错误", f"无法读取图像: {str(e)}")

    def add_synthetic_noise(self):
        """添加高斯噪声以模拟退化（便于计算PSNR/SSIM验证模型）"""
        if self.clean_img_np is None:
            messagebox.showwarning("提示", "请先加载原始图像！")
            return
            
        sigma = self.noise_level.get()
        # 模拟高斯噪声
        noise = np.random.normal(0, sigma, self.clean_img_np.shape)
        noisy = self.clean_img_np.astype(np.float32) + noise
        self.noisy_img_np = np.clip(noisy, 0, 255).astype(np.uint8)
        
        self._display_image(self.noisy_img_np, self.lbl_noisy_img)
        self.lbl_denoised_img.config(image='', text="等待处理...")
        self.btn_save.config(state=tk.DISABLED)
        self.update_status(0, f"已添加强度为 {sigma} 的模拟高斯噪声。")

    def _display_image(self, img_np, label_widget):
        """缩放并显示 numpy 图像到指定的 Label"""
        img_pil = Image.fromarray(img_np)
        img_pil.thumbnail(self.display_size, Image.Resampling.LANCZOS)
        img_tk = ImageTk.PhotoImage(img_pil)
        label_widget.config(image=img_tk, text="")
        label_widget.image = img_tk # 保持引用防止被垃圾回收

    def start_denoising(self):
        if self.noisy_img_np is None:
            messagebox.showwarning("提示", "请先加载或生成含噪图像！")
            return
            
        # 禁用按钮防止重复点击
        self.btn_run.config(state=tk.DISABLED)
        self.lbl_metrics.config(text="处理中...")
        
        # 启动后台线程执行深度学习运算
        thread = threading.Thread(target=self._denoising_thread_task)
        thread.daemon = True
        thread.start()

    def _denoising_thread_task(self):
        try:
            self.update_status(0.01, "正在初始化深度学习引擎...")
            start_time = time.time()
            
            # 实例化我们在第一阶段编写的 MASH 核心引擎
            denoiser = MASHDenoiser(device_type='auto')
            
            # 解析 GUI 用户参数
            strategy = self.mask_strategy.get()
            force_mask = float(strategy) if strategy != "auto" else None
            force_lps = self.lps_enabled.get()
            
            # 调用核心管线
            self.denoised_img_np = denoiser.denoise(
                self.noisy_img_np, 
                force_mask=force_mask, 
                force_lps=force_lps,
                callback=self.update_status
            )
            
            elapsed = time.time() - start_time
            
            # 切换回主线程更新结果 UI
            self.root.after(0, self._on_denoising_complete, elapsed)
            
        except Exception as e:
            self.root.after(0, self._on_denoising_error, str(e))

    def _on_denoising_complete(self, elapsed_time):
        # 1. 显示去噪结果
        self._display_image(self.denoised_img_np, self.lbl_denoised_img)
        
        # 2. 如果存在真实干净原图（即用户添加了模拟噪声），则计算评估指标
        if self.clean_img_np is not None and not np.array_equal(self.clean_img_np, self.noisy_img_np):
            psnr_val = peak_signal_noise_ratio(self.clean_img_np, self.denoised_img_np)
            ssim_val = structural_similarity(self.clean_img_np, self.denoised_img_np, channel_axis=2)
            self.lbl_metrics.config(text=f"PSNR: {psnr_val:.2f} dB  |  SSIM: {ssim_val:.4f}  (耗时: {elapsed_time:.1f}s)")
        else:
            self.lbl_metrics.config(text=f"盲去噪完成 (未知 Ground Truth)。耗时: {elapsed_time:.1f}s")
            
        self.btn_run.config(state=tk.NORMAL)
        self.btn_save.config(state=tk.NORMAL)
        self.update_status(1.0, "图像去噪任务圆满完成。")

    def _on_denoising_error(self, error_msg):
        messagebox.showerror("深度学习处理异常", f"去噪过程中发生错误:\n{error_msg}")
        self.btn_run.config(state=tk.NORMAL)
        self.update_status(0.0, "处理失败。")

    def save_result(self):
        if self.denoised_img_np is None:
            return
            
        file_path = filedialog.asksaveasfilename(
            defaultextension=".png", 
            filetypes=[("PNG Image", "*.png"), ("JPEG Image", "*.jpg")]
        )
        if file_path:
            Image.fromarray(self.denoised_img_np).save(file_path)
            messagebox.showinfo("成功", "去噪图像已保存！")

if __name__ == "__main__":
    root = tk.Tk()
    # 设置 ttk 现代化主题
    style = ttk.Style()
    if 'clam' in style.theme_names():
        style.theme_use('clam')
        
    app = NoiseEliminationSystem(root)
    root.mainloop()