import os
import time
import numpy as np
import cv2
import yaml
from flask import Flask, render_template, request, jsonify, send_from_directory
from werkzeug.utils import secure_filename

# PyTorch 可选（传统方法不需要）
try:
    import torch
    TORCH_AVAILABLE = True
    print(f"[DEBUG] PyTorch imported successfully. Version: {torch.__version__}")
except ImportError as e:
    TORCH_AVAILABLE = False
    print(f"[DEBUG] Warning: PyTorch not installed. Error: {e}")

from models.traditional import TraditionalDenoiser
from engine import MetricsCalculator

if TORCH_AVAILABLE:
    from fast_mash_infer import FastMASHDenoiser
    from models.dncnn import DnCNN

# ==================== 应用配置 ====================

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['RESULTS_FOLDER'] = 'static/results'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'bmp', 'tif', 'tiff'}

# 确保目录存在
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['RESULTS_FOLDER'], exist_ok=True)

# ==================== 全局对象 ====================

metrics_calc = MetricsCalculator()
denoiser_cache = {}  # 缓存已初始化的去噪器

# MASH 预训练模型：启动时加载一次，后续直接推理
if TORCH_AVAILABLE:
    print(f"[APP] CUDA available: {torch.cuda.is_available()}")
    mash_denoiser = FastMASHDenoiser()
    print("[APP] FastMASHDenoiser 已加载")


def load_config(method: str, dataset: str = 'sidd') -> dict:
    """加载配置文件"""
    config_path = f'configs/{method}_{dataset}.yaml'
    if os.path.exists(config_path):
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
            # 将嵌套的配置结构展平
            if config and 'model' in config:
                config.update(config['model'])
            if config and 'mash' in config:
                config.update(config['mash'])
            return config
    return {}


def get_denoiser(method: str, dataset: str = 'sidd'):
    """获取去噪器实例（带缓存）"""
    cache_key = f"{method}_{dataset}"
    if method == 'mash':
        return mash_denoiser
    if cache_key not in denoiser_cache:
        config = load_config(method, dataset)
        if method == 'dncnn':
            if not TORCH_AVAILABLE:
                raise ValueError("DnCNN 方法需要安装 PyTorch")
            denoiser_cache[cache_key] = DnCNN(config)
            weight_path = f'weights/dncnn_{dataset}.pth'
            if os.path.exists(weight_path):
                denoiser_cache[cache_key].load_weights(weight_path)
        elif method == 'traditional':
            denoiser_cache[cache_key] = TraditionalDenoiser(config)
        else:
            raise ValueError(f"Unknown method: {method}")
    return denoiser_cache[cache_key]


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def add_noise(img: np.ndarray, noise_type: str, noise_level: float) -> np.ndarray:
    """添加合成噪声"""
    img_float = img.astype(np.float32) / 255.0

    if noise_type == 'gaussian':
        noise = np.random.normal(0, noise_level / 255.0, img.shape)
        noisy = img_float + noise
    elif noise_type == 'poisson':
        noisy = np.random.poisson(img_float * noise_level) / noise_level
    elif noise_type == 'salt':
        # 椒盐噪声
        noisy = img_float.copy()
        prob = noise_level / 100.0
        # 盐噪声（白点）
        salt_mask = np.random.random(img.shape[:2]) < prob / 2
        noisy[salt_mask] = 1.0
        # 椒噪声（黑点）
        pepper_mask = np.random.random(img.shape[:2]) < prob / 2
        noisy[pepper_mask] = 0.0
    else:
        noisy = img_float

    return (np.clip(noisy, 0, 1) * 255).astype(np.uint8)


# ==================== 页面路由 ====================

@app.route('/')
def index():
    """首页"""
    return render_template('index.html')


@app.route('/denoise')
def denoise_page():
    """去噪对比页"""
    return render_template('denoise.html')


@app.route('/evaluate')
def evaluate_page():
    """性能评估页"""
    return render_template('evaluate.html')


# ==================== API 接口 ====================

@app.route('/api/upload', methods=['POST'])
def upload_image():
    """
    上传图像
    """
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        # 添加时间戳避免重名
        name, ext = os.path.splitext(filename)
        filename = f"{name}_{int(time.time())}{ext}"
        save_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(save_path)

        return jsonify({
            'success': True,
            'filename': filename,
            'path': f'/static/uploads/{filename}'
        })

    return jsonify({'error': 'File type not allowed'}), 400


@app.route('/api/denoise', methods=['POST'])
def denoise_image():
    """
    执行去噪
    Request: {
        "filename": "xxx.png",
        "methods": ["mash", "dncnn", "traditional"],
        "noise_type": "gaussian",  # gaussian/poisson/salt/real
        "noise_level": 25,
        "traditional_method": "median"  # mean/median/gaussian/bilateral/wavelet
    }
    """
    data = request.json
    filename = data.get('filename')
    methods = data.get('methods', ['mash'])
    noise_type = data.get('noise_type', 'gaussian')
    noise_level = data.get('noise_level', 25)
    trad_method = data.get('traditional_method', 'median')

    # 读取原始图像（先查 uploads，再查 demo_images）
    img_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    if not os.path.exists(img_path):
        demo_path = os.path.join('static/demo_images', filename)
        if os.path.exists(demo_path):
            img_path = demo_path
        else:
            return jsonify({'error': 'Image not found'}), 404

    img = cv2.imread(img_path)
    if img is None:
        return jsonify({'error': 'Failed to read image'}), 400

    # 全局缩放：长边超过 800px 则等比缩放
    MAX_SIDE = 800
    h, w = img.shape[:2]
    long_side = max(h, w)
    if long_side > MAX_SIDE:
        scale = MAX_SIDE / long_side
        img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)

    # 转灰度（如果是 RGB，保留双份用于不同方法）
    is_gray = len(img.shape) == 2
    if not is_gray:
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # 转换为RGB用于MASH处理
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    else:
        img_gray = img
        img_rgb = img

    # 如果不是真实噪声，添加合成噪声
    if noise_type != 'real':
        noisy_img = add_noise(img, noise_type, noise_level)
        noisy_gray = add_noise(img_gray, noise_type, noise_level) if not is_gray else noisy_img
        # RGB版本的噪声图
        if not is_gray:
            noisy_rgb = add_noise(img_rgb, noise_type, noise_level)
        else:
            noisy_rgb = noisy_img
    else:
        noisy_img = img
        noisy_gray = img_gray if is_gray else img_gray
        noisy_rgb = img_rgb

    # 保存噪声图像
    noisy_filename = f"noisy_{filename}"
    noisy_path = os.path.join(app.config['UPLOAD_FOLDER'], noisy_filename)
    cv2.imwrite(noisy_path, noisy_img)

    results = []
    for method in methods:
        try:
            denoiser = get_denoiser(method)

            # 执行去噪
            start_time = time.time()

            if method == 'traditional':
                # 传统方法
                denoised = denoiser.denoise(noisy_img, method=trad_method)
            elif method == 'mash':
                # MASH 极速推理：预训练权重 + 3 次 MC 融合
                noisy_float = noisy_rgb.astype(np.float32) / 255.0
                denoised_float = mash_denoiser.denoise(noisy_float)
                denoised_rgb = (np.clip(denoised_float, 0, 1) * 255).astype(np.uint8)
                if not is_gray:
                    denoised = cv2.cvtColor(denoised_rgb, cv2.COLOR_RGB2BGR)
                else:
                    denoised = denoised_rgb
            else:
                # DnCNN 监督方法
                # 将图像转为浮点数 [0, 1]
                noisy_float = noisy_rgb.astype(np.float32) / 255.0

                if is_gray:
                    # 灰度图：直接去噪
                    denoised = denoiser.denoise(noisy_float)
                    denoised = (denoised * 255).astype(np.uint8)
                else:
                    # RGB 图：对每个通道分别去噪
                    channels = []
                    for c in range(3):
                        ch = noisy_float[:, :, c]
                        denoised_ch = denoiser.denoise(ch)
                        channels.append((denoised_ch * 255).astype(np.uint8))
                    denoised_rgb = np.stack(channels, axis=2)
                    # 转换回BGR用于保存
                    denoised = cv2.cvtColor(denoised_rgb, cv2.COLOR_RGB2BGR)

            elapsed = (time.time() - start_time) * 1000

            # 保存去噪结果
            result_filename = f"{os.path.splitext(filename)[0]}_{method}.png"
            result_path = os.path.join(app.config['RESULTS_FOLDER'], result_filename)
            cv2.imwrite(result_path, denoised)

            # 计算指标（如果有参考图像）
            psnr_val = None
            ssim_val = None
            reference = img if not is_gray else img_gray
            psnr_val = metrics_calc.calculate_psnr(denoised, reference)
            ssim_val = metrics_calc.calculate_ssim(denoised, reference)

            results.append({
                'method': method,
                'denoised_path': f'/static/results/{result_filename}',
                'psnr': round(psnr_val, 2) if psnr_val else None,
                'ssim': round(ssim_val, 4) if ssim_val else None,
                'time_ms': round(elapsed, 1)
            })

        except Exception as e:
            results.append({
                'method': method,
                'error': str(e)
            })

    return jsonify({
        'success': True,
        'original_path': f'/static/uploads/{filename}',
        'noisy_path': f'/static/uploads/{noisy_filename}',
        'results': results
    })


@app.route('/api/evaluate', methods=['POST'])
def evaluate_dataset():
    """
    批量评估（模拟）
    实际使用时需要接入真实数据集
    """
    data = request.json
    methods = data.get('methods', ['mash', 'dncnn', 'traditional'])
    noise_level = data.get('noise_level', 25)

    # 模拟评估结果（实际应从数据集计算）
    mock_results = {
        'mash': {'psnr': 34.21, 'ssim': 0.9312, 'avg_time': 156},
        'dncnn': {'psnr': 32.85, 'ssim': 0.9087, 'avg_time': 42},
        'traditional': {'psnr': 28.34, 'ssim': 0.8456, 'avg_time': 8}
    }

    results = []
    for method in methods:
        if method in mock_results:
            results.append({
                'method': method,
                'psnr': mock_results[method]['psnr'],
                'ssim': mock_results[method]['ssim'],
                'avg_time': mock_results[method]['avg_time']
            })

    return jsonify({'metrics': results})


@app.route('/api/methods', methods=['GET'])
def get_methods():
    """获取可用方法列表"""
    return jsonify({
        'methods': [
            {'id': 'mash', 'name': 'MASH (核心)', 'type': 'self-supervised'},
            {'id': 'dncnn', 'name': 'DnCNN', 'type': 'supervised'},
            {'id': 'traditional', 'name': '传统方法', 'type': 'traditional'}
        ],
        'traditional_methods': ['mean', 'median', 'gaussian', 'bilateral', 'wavelet']
    })


# ==================== 静态文件 ====================

@app.route('/static/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)


@app.route('/static/results/<filename>')
def result_file(filename):
    return send_from_directory(app.config['RESULTS_FOLDER'], filename)


# ==================== 启动 ====================

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
