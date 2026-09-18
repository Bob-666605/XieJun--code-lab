"""
下载 5 张 Kodak 经典自然场景图 + 生成带噪声版本
无需任何第三方库，仅用 urllib + numpy + cv2
"""
import os
import urllib.request
import numpy as np
import cv2

OUTPUT_DIR = 'static/demo_images'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Kodak 经典图片 URL
IMAGES = [
    ('demo_1.png', 'https://r0k.us/graphics/kodak/kodak/kodim23.png'),  # 棕马
    ('demo_2.png', 'https://r0k.us/graphics/kodak/kodak/kodim05.png'),  # 红白房子
    ('demo_3.png', 'https://r0k.us/graphics/kodak/kodak/kodim13.png'),  # 森林河流
    ('demo_4.png', 'https://r0k.us/graphics/kodak/kodak/kodim21.png'),  # 金门大桥
    ('demo_5.png', 'https://r0k.us/graphics/kodak/kodak/kodim06.png'),  # 老汉与船
]

# ==================== 下载 ====================

for name, url in IMAGES:
    save_path = os.path.join(OUTPUT_DIR, name)
    if os.path.exists(save_path):
        print(f"  {name} 已存在，跳过")
        continue
    print(f"  下载 {name} ...")
    urllib.request.urlretrieve(url, save_path)
    print(f"  {name} 完成")

# ==================== 加噪声 ====================

NOISE_STD = 25  # 高斯噪声标准差

for name, _ in IMAGES:
    img_path = os.path.join(OUTPUT_DIR, name)
    img = cv2.imread(img_path)
    if img is None:
        print(f"  读取失败: {name}")
        continue

    base = name.replace('.png', '')
    img_f = img.astype(np.float32)

    # 高斯噪声
    gauss = img_f + np.random.normal(0, NOISE_STD, img.shape)
    gauss = np.clip(gauss, 0, 255).astype(np.uint8)
    cv2.imwrite(os.path.join(OUTPUT_DIR, f'{base}_noisy_gauss.png'), gauss)

    # 椒盐噪声
    prob = 0.05
    salt_mask = np.random.random(img.shape[:2]) < prob / 2
    pepper_mask = np.random.random(img.shape[:2]) < prob / 2
    sp = img.copy()
    sp[salt_mask] = 255
    sp[pepper_mask] = 0
    cv2.imwrite(os.path.join(OUTPUT_DIR, f'{base}_noisy_sp.png'), sp)

    print(f"  {name} 噪声版本已生成")

print(f"\n完成！文件列表:")
for f in sorted(os.listdir(OUTPUT_DIR)):
    size = os.path.getsize(os.path.join(OUTPUT_DIR, f)) // 1024
    print(f"  {f}  ({size} KB)")
