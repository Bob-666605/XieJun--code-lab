"""
生成答辩野外测试图集 eval_test_gallery/
4 种极限测试场景，仅依赖 urllib + numpy + cv2
"""
import os
import urllib.request
import numpy as np
import cv2

OUTPUT_DIR = 'eval_test_gallery'
os.makedirs(OUTPUT_DIR, exist_ok=True)


def download(url, name):
    path = os.path.join(OUTPUT_DIR, name)
    if os.path.exists(path):
        print(f"  {name} 已存在，跳过")
        return cv2.imread(path)
    print(f"  下载 {name} ...")
    urllib.request.urlretrieve(url, path)
    print(f"  {name} 完成")
    return cv2.imread(path)


# ==================== 图1: 文本去噪测试 ====================
print("[1/4] 生成文本去噪测试图...")
text_img = np.ones((512, 768, 3), dtype=np.uint8) * 240
lines = [
    "The quick brown fox jumps over the lazy dog.",
    "ABCD abcd 0123456789 !@#$%^&*()",
    "MASH: Masked and Shuffled Blind Spot Denoising",
    "Self-supervised learning for image restoration.",
    "Gaussian noise sigma=25, PSNR improvement +8dB.",
    "Convolutional neural network denoiser architecture.",
    "Test-Time Augmentation with 4x geometric transforms.",
    "Local Pixel Shuffling destroys noise correlation.",
]
for i, line in enumerate(lines):
    y = 40 + i * 55
    cv2.putText(text_img, line, (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (30, 30, 30), 1, cv2.LINE_AA)

# 加 σ=20 的轻微噪声（模拟扫描件噪点）
noise = np.random.normal(0, 20, text_img.shape).astype(np.float32)
text_noisy = np.clip(text_img.astype(np.float32) + noise, 0, 255).astype(np.uint8)
cv2.imwrite(os.path.join(OUTPUT_DIR, 'test_text_doc.png'), text_noisy)
print("  test_text_doc.png 完成")


# ==================== 图2: 平滑夜空测试 ====================
print("[2/4] 生成夜空平滑测试图...")
sky_img = np.zeros((512, 768, 3), dtype=np.float32)
# 渐变夜空：底部略亮（光污染）
for y in range(512):
    brightness = 5 + 15 * (y / 512) ** 2
    sky_img[y, :, 0] = brightness * 0.8  # R
    sky_img[y, :, 1] = brightness * 0.7  # G
    sky_img[y, :, 2] = brightness * 1.2  # B 偏蓝
# 添加星星（白色亮点）
np.random.seed(42)
star_y = np.random.randint(0, 350, 80)
star_x = np.random.randint(0, 768, 80)
for sy, sx in zip(star_y, star_x):
    cv2.circle(sky_img, (sx, sy), 1, (200, 200, 255), -1)
# 加强色噪（模拟高ISO红绿噪点）
color_noise = np.random.normal(0, 18, sky_img.shape).astype(np.float32)
color_noise[:, :, 0] *= 1.5  # 红色通道噪声明显
color_noise[:, :, 1] *= 1.3  # 绿色
sky_noisy = np.clip(sky_img + color_noise, 0, 255).astype(np.uint8)
cv2.imwrite(os.path.join(OUTPUT_DIR, 'test_smooth_sky.png'), sky_noisy)
print("  test_smooth_sky.png 完成")


# ==================== 图3: 高频纹理测试 ====================
print("[3/4] 下载高频纹理图 + 加 σ=35 噪声...")
# 用 Kodak 棕马图（毛发纹理丰富）
kodak_url = 'https://r0k.us/graphics/kodak/kodak/kodim23.png'
texture_img = download(kodak_url, '_kodak23_src.png')
if texture_img is not None:
    noise = np.random.normal(0, 35, texture_img.shape).astype(np.float32)
    texture_noisy = np.clip(texture_img.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    cv2.imwrite(os.path.join(OUTPUT_DIR, 'test_texture_hair.png'), texture_noisy)
    os.remove(os.path.join(OUTPUT_DIR, '_kodak23_src.png'))  # 清理临时文件
    print("  test_texture_hair.png 完成")


# ==================== 图4: 极限椒盐噪声 ====================
print("[4/4] 下载风景图 + 15% 椒盐噪声...")
# 用 Kodak 森林河流图
kodak_url2 = 'https://r0k.us/graphics/kodak/kodak/kodim13.png'
sp_img = download(kodak_url2, '_kodak13_src.png')
if sp_img is not None:
    sp = sp_img.copy()
    prob = 0.15
    salt = np.random.random(sp.shape[:2]) < prob / 2
    pepper = np.random.random(sp.shape[:2]) < prob / 2
    sp[salt] = 255
    sp[pepper] = 0
    cv2.imwrite(os.path.join(OUTPUT_DIR, 'test_extreme_saltpepper.png'), sp)
    os.remove(os.path.join(OUTPUT_DIR, '_kodak13_src.png'))
    print("  test_extreme_saltpepper.png 完成")


# ==================== 汇总 ====================
print("\n=== eval_test_gallery/ 文件清单 ===")
for f in sorted(os.listdir(OUTPUT_DIR)):
    if f.startswith('_'):
        continue
    size = os.path.getsize(os.path.join(OUTPUT_DIR, f)) // 1024
    print(f"  {f}  ({size} KB)")

print("\n老板，答辩野外测试模块已在 eval_test_gallery/ 准备就绪，包含 4 种极限测试场景！")
