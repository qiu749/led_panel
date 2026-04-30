"""
图像增强 + OCR 识别
尝试多种图像增强方法来提高 OCR 识别率
"""

import os
os.environ['PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK'] = 'True'

import cv2
import numpy as np
import logging

logging.basicConfig(level=logging.WARNING)

def enhance_image(image):
    """
    对图像进行多种增强处理
    
    Args:
        image: 输入图像
        
    Returns:
        dict: 各种增强方法处理后的图像
    """
    methods = {}
    
    # 原图
    methods['original'] = image.copy()
    
    # 灰度化
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()
    
    # CLAHE 增强
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    methods['clahe'] = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)
    
    # 二值化 (Otsu)
    _, binary = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    methods['binary'] = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)
    
    # 自适应二值化
    adaptive = cv2.adaptiveThreshold(enhanced, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    methods['adaptive'] = cv2.cvtColor(adaptive, cv2.COLOR_GRAY2BGR)
    
    # 放大 2 倍
    h, w = image.shape[:2]
    scaled = cv2.resize(image, (w * 2, h * 2), interpolation=cv2.INTER_CUBIC)
    methods['scaled_2x'] = scaled
    
    # 放大 3 倍
    scaled_3x = cv2.resize(image, (w * 3, h * 3), interpolation=cv2.INTER_CUBIC)
    methods['scaled_3x'] = scaled_3x
    
    # 锐化
    kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
    sharpened = cv2.filter2D(image, -1, kernel)
    methods['sharpened'] = sharpened
    
    # 放大 + 锐化
    scaled_sharpened = cv2.filter2D(scaled_3x, -1, kernel)
    methods['scaled_sharpened'] = scaled_sharpened
    
    return methods

def try_ocr(ocr, image):
    """对图像进行 OCR"""
    result = ocr(image)
    
    texts = []
    if result.boxes is not None and result.txts is not None:
        for box, txt, score in zip(result.boxes, result.txts, result.scores):
            if score > 0.1:
                texts.append(f"{txt}({score:.2f})")
    
    return texts

def analyze_image(image_path, ocr, output_dir):
    """分析单张图像"""
    image = cv2.imread(image_path)
    if image is None:
        return
    
    filename = os.path.basename(image_path)
    print(f"\n{'='*60}")
    print(f"分析: {filename}")
    print(f"图像尺寸: {image.shape[1]}x{image.shape[0]}")
    print(f"{'='*60}")
    
    methods = enhance_image(image)
    
    os.makedirs(output_dir, exist_ok=True)
    
    for method_name, enhanced in methods.items():
        enhanced_path = os.path.join(output_dir, f"{filename.replace('.jpg', '')}_{method_name}.jpg")
        cv2.imwrite(enhanced_path, enhanced)
        
        texts = try_ocr(ocr, enhanced)
        
        if texts:
            print(f"  [{method_name:15s}] OCR: {', '.join(texts)}")
        else:
            print(f"  [{method_name:15s}] 无文字")

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='图像增强 + OCR 识别')
    parser.add_argument('--input', type=str, required=True, help='输入图像路径或目录')
    parser.add_argument('--output', type=str, default='enhanced_ocr', help='输出目录')
    
    args = parser.parse_args()
    
    from rapidocr import RapidOCR
    ocr = RapidOCR()
    
    if os.path.isdir(args.input):
        image_files = sorted([f for f in os.listdir(args.input) if f.endswith('.jpg') and 'box' in f])
        print(f"找到 {len(image_files)} 张检测框图像")
        for image_file in image_files[:10]:
            image_path = os.path.join(args.input, image_file)
            analyze_image(image_path, ocr, args.output)
    else:
        analyze_image(args.input, ocr, args.output)
    
    print(f"\n增强后的图像保存到: {args.output}")

if __name__ == '__main__':
    main()
