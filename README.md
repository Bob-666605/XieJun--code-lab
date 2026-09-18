# 自然场景图像噪声消除系统

基于 **MASH 自监督学习**的多方法对比图像去噪平台。

系统以 MASH（Masked Adaptive Self-supervised Hierarchical）自监督去噪算法为核心，
集成 DnCNN 监督学习方法和 5 种传统滤波方法，提供 Web 端可视化对比界面与量化评估。

---

## 一、环境要求

| 项目 | 版本 |
|---|---|
| 操作系统 | Windows / Linux / macOS |
| Python | 3.10 及以上（开发环境为 3.12） |
| PyTorch | >= 2.0.0 |
| 内存 | 建议 8GB 以上 |

> GPU 为可选。检测到 CUDA 时自动使用 GPU，否则回退到 CPU 推理（速度较慢）。

---

## 二、安装依赖

```bash
pip install -r requirements.txt
```

依赖清单：PyTorch、torchvision、NumPy、Pillow、OpenCV、scikit-image、SciPy、
PyWavelets、einops、PyYAML、Flask、Werkzeug

---

## 三、启动系统

### 方式一：Windows 批处理（推荐）

直接双击 `start_server.bat`

### 方式二：命令行

```bash
python app.py
```

启动成功后控制台会输出：

```
[APP] FastMASHDenoiser 已加载
 * Running on http://0.0.0.0:5000
```

### 方式三：桌面 GUI（可选）

```bash
python main_gui.py
```

提供基于 Tkinter 的本地图形界面，无需浏览器。

---

## 四、访问系统

服务启动后，浏览器打开：

```
http://localhost:5000
```

局域网内其他设备可通过 `http://<本机IP>:5000` 访问。

### 页面功能

| 页面 | 地址 | 功能 |
|---|---|---|
| 首页 | `/` | 系统介绍与特性展示 |
| 去噪对比 | `/denoise` | 上传图像、配置噪声、多方法并行去噪、结果对比 |
| 性能评估 | `/evaluate` | 批量评估与图表可视化 |

### 使用流程

1. 打开「去噪对比」页，上传图像（支持拖拽）或直接点击演示图
2. 勾选去噪方法（MASH / DnCNN / 传统方法）
3. 选择噪声类型（高斯 / 泊松 / 椒盐 / 真实噪声）并调节噪声强度
4. 点击「开始去噪」，查看各方法的去噪结果与 PSNR / SSIM / 耗时指标
5. 使用对比视图可并排比较任意两种方法

---

## 五、项目结构

```
去噪系统/
├── app.py                    # Flask Web 服务器主入口
├── main_gui.py               # Tkinter 桌面 GUI
│
├── model.py                  # UNet_n2n_un 网络定义、盲点掩模生成
├── fast_mash_infer.py        # MASH 快速推理（预训练权重 + TTA）★ Web 系统实际调用
├── mash_core.py              # MASH 完整流水线（噪声探测 + LPS + 训练 + MC 推理）
├── mash_standalone.py        # MASH 单文件完整实现
├── utils.py                  # 工具函数（滑动标准差、像素洗牌）
│
├── models/
│   ├── base.py               # BaseDenoiser 去噪器抽象基类
│   ├── dncnn.py              # DnCNN 监督去噪器
│   └── traditional.py        # 传统滤波方法（均值/中值/高斯/双边/小波）
│
├── engine/
│   └── metrics.py            # PSNR / SSIM / MAE / MSE 指标计算
│
├── configs/                  # YAML 超参数配置
├── weights/                  # 模型权重
│   └── mash_pretrained.pth   # MASH 预训练权重
│
├── templates/                # Jinja2 HTML 模板
├── static/
│   ├── css/  js/             # 前端样式与脚本
│   └── demo_images/          # 演示图像
│
├── train_dncnn.py            # DnCNN 训练脚本
├── train_mash_offline.py     # MASH 离线预训练脚本
├── evaluate.py               # 批量评估脚本
├── run_eval.py               # SIDD 数据集评估
└── requirements.txt
```

---

## 六、核心算法说明

### MASH 自监督去噪（核心）

**核心思想**：通过随机遮挡输入图像的部分像素（盲点掩模），迫使网络从邻域像素
推断被遮挡位置的真实值，从而在**不需要干净图像配对**的情况下完成去噪训练。

**推理流程**（`fast_mash_infer.py`）：

```
输入图像
  → 反射填充至 32 的倍数
  → 4 种几何变换（原图 / 左右翻转 / 上下翻转 / 旋转180°）
  → 每种变换执行 3 次随机掩模推理
  → 12 次结果平均融合
  → 裁剪回原尺寸输出
```

**训练流程**（`mash_core.py` / `train_mash_offline.py`）：

```
噪声相关性探测（双掩模率快速训练）
  → 计算 ε = |σ_high - σ_low| 判断噪声空间相关性
  → 若 ε 超阈值，启用 LPS 局部像素洗牌去相关
  → Patch 训练（复合损失 0.8×L1 + 0.2×(1-SSIM)）
  → 蒙特卡洛多次推理取平均
```

### 网络结构

5 级 U-Net 编解码器（`UNet_n2n_un`），48 个基础通道，3×3 卷积核，
LeakyReLU(0.1) 激活，编码器-解码器间通过跳跃连接融合多尺度特征。

### DnCNN 监督去噪

17 层深度 CNN（64 特征通道），采用**残差学习**策略预测噪声残差 R(y)，
去噪结果 = y − R(y)。需要含噪-干净图像对进行监督训练。

> ⚠️ 本目录 `weights/` 中未包含 DnCNN 权重文件，DnCNN 方法需先运行
> `train_dncnn.py` 生成 `weights/dncnn_sidd.pth` 后方可正常使用。
> 未加载权重时该路径使用随机初始化网络，结果无参考意义。

### 传统滤波方法

无需训练，直接调用 OpenCV / PyWavelets 实现：均值、中值、高斯、双边、小波软阈值。

---

## 七、API 接口

| 端点 | 方法 | 说明 |
|---|---|---|
| `/api/upload` | POST | 上传图像（multipart，限 10MB，支持 PNG/JPG/BMP/TIFF） |
| `/api/denoise` | POST | 执行去噪（指定方法、噪声类型、噪声强度） |
| `/api/evaluate` | POST | 批量评估 |
| `/api/methods` | GET | 查询可用方法列表 |

**请求示例**：

```json
POST /api/denoise
{
  "filename": "xxx.jpg",
  "methods": ["mash", "dncnn", "traditional"],
  "noise_type": "gaussian",
  "noise_level": 25,
  "traditional_method": "median"
}
```

---

## 八、数据集说明

以下脚本需要 SIDD 数据集的 `.mat` 文件，**未包含在本交付包中**（体积过大）：

| 脚本 | 所需文件 |
|---|---|
| `run_eval.py` | `data/ValidationNoisyBlocksSrgb.mat`、`data/ValidationGtBlocksSrgb.mat` |
| `train_dncnn.py` | 同上 |
| `evaluate.py` | `data/SIDD/` 或 `data/FMDD/` 目录 |

如需运行上述脚本，请自行下载 SIDD 验证集并放置于 `data/` 目录下：

- SIDD 数据集官网：https://www.eecs.yorku.ca/~kamel/sidd/

> **Web 系统（`app.py`）不依赖该数据集**，可直接运行。

---

## 九、注意事项

1. **性能评估页**：`/api/evaluate` 接口当前返回预置的演示数据，用于界面功能展示。
   真实数据集的批量评估请运行 `evaluate.py` 或 `run_eval.py`。

2. **MASH 推理速度**：默认执行 12 次推理融合（4 变换 × 3 次掩模），
   CPU 环境下单张图像约需数秒，GPU 环境下可显著加速。

3. **图像尺寸**：系统会自动将长边超过 800px 的图像等比缩放后再处理，
   以控制计算量。

4. **首次启动**：需要加载 MASH 预训练权重，启动耗时约 5-15 秒，请耐心等待。

---

## 十、引用

MASH 算法参考论文：

```
MASH: Masked Adaptive Self-supervised Hierarchical Image Denoising
arXiv:2404.09389
```

DnCNN 算法参考论文：

```
Zhang K, Zuo W, Chen Y, et al. Beyond a Gaussian Denoiser:
Residual Learning of Deep CNN for Image Denoising.
IEEE Transactions on Image Processing, 2017, 26(7): 3142-3155.
```
