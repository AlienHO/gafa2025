"""
图像处理工具模块 - 提供图像变换、编码和绘制功能
"""
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from io import BytesIO
import base64
from modules.config import CHINESE_FONT, VISION_API_IMAGE_QUALITY, VISION_API_MAX_SIZE


def setup_camera():
    """
    设置并初始化摄像头
    
    返回:
        tuple: (cap, frame_width, frame_height) 如果成功
        None: 如果初始化失败
    """
    from modules.config import CAMERA_ID
    
    # 设定摄像头ID
    cap = cv2.VideoCapture(CAMERA_ID)
    
    if not cap.isOpened():
        print("错误：无法打开摄像头")
        return None
    
    # 获取摄像头尺寸用于归一化 - 通过读取一帧来获取真实分辨率
    ret, test_frame = cap.read()
    if ret:
        frame_height, frame_width = test_frame.shape[:2]
        print(f"[已获取] 摄像头尺寸: {frame_width}x{frame_height}")
    else:
        # 如果读取失败，使用默认值
        frame_width = 640
        frame_height = 480
        print(f"警告：无法读取摄像头帧，使用默认尺寸: {frame_width}x{frame_height}")
    
    return cap, frame_width, frame_height


def crop_frame_to_pil(frame, bbox, resize=None):
    """
    将视频帧裁剪为PIL图像
    
    参数:
        frame: OpenCV格式的帧
        bbox: 边界框 (x1, y1, x2, y2)
        resize: 可选的调整大小元组 (width, height)
        
    返回:
        PIL.Image: 裁剪（和调整大小）后的PIL图像
    """
    x1, y1, x2, y2 = [int(v) for v in bbox]
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    cropped = rgb_frame[y1:y2, x1:x2]
    
    # 创建PIL图像
    pil_image = Image.fromarray(cropped)
    
    # 如果需要调整大小
    if resize:
        pil_image = pil_image.resize(resize, Image.LANCZOS)
        
    return pil_image


def encode_image_to_base64(pil_image, format='JPEG', quality=VISION_API_IMAGE_QUALITY, max_size=VISION_API_MAX_SIZE):
    """
    将PIL图像编码为base64字符串，使用内存缓冲区而不是临时文件
    
    参数:
        pil_image: PIL图像对象
        format: 图像格式，默认JPEG
        quality: JPEG压缩质量(1-100)，仅用于JPEG
        max_size: 图像的最大边长，保持宽高比调整大小
        
    返回:
        str: base64编码的图像数据
    """
    # 检查是否需要调整大小
    width, height = pil_image.size
    if max_size and (width > max_size or height > max_size):
        # 计算调整后的大小，保持宽高比
        if width > height:
            new_width = max_size
            new_height = int(height * (max_size / width))
        else:
            new_height = max_size
            new_width = int(width * (max_size / height))
        
        pil_image = pil_image.resize((new_width, new_height), Image.LANCZOS)
    
    # 使用内存缓冲区编码图像
    buffer = BytesIO()
    
    # 根据格式和质量设置
    if format.upper() == 'JPEG':
        pil_image.save(buffer, format=format, quality=quality)
    else:
        pil_image.save(buffer, format=format)
    
    return base64.b64encode(buffer.getvalue()).decode('utf-8')


def draw_anything_boxes(frame, anything_boxes):
    """
    在图像上绘制OpenAI Vision的框和文本
    使用优化的绘制方法，减少OpenCV和PIL之间的转换
    
    参数:
        frame: OpenCV格式的帧（将被原地修改）
        anything_boxes: 框列表 [(box, text, timestamp), ...]
    """
    if not anything_boxes:
        return
        
    # 一次性转换为PIL图像
    pil_img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil_img)
    font = CHINESE_FONT
    
    # 绘制所有框和文本
    for box, text, _ in anything_boxes:
        x1, y1, x2, y2 = [int(v) for v in box]
        
        # 在PIL图像上绘制矩形 - 黄色细线
        draw.rectangle([(x1, y1), (x2, y2)], outline=(255, 255, 0), width=1)
        
        # 计算文本边界框
        text_bbox = draw.textbbox((0, 0), text, font=font)
        text_width = text_bbox[2] - text_bbox[0]
        text_height = text_bbox[3] - text_bbox[1]
        
        # 文本位置: 居中于框的顶部
        text_x = x1 + (x2 - x1 - text_width) // 2
        text_y = y1 - text_height - 5
        
        # 如果顶部空间不够，则放在框内顶部
        if text_y < 0:
            text_y = y1 + 5
            
        # 绘制文本背景
        draw.rectangle(
            [(text_x - 2, text_y - 2), (text_x + text_width + 2, text_y + text_height + 2)],
            fill=(0, 0, 0))
            
        # 绘制文本
        draw.text((text_x, text_y), text, font=font, fill=(255, 255, 0))
    
    # 转换回OpenCV格式并更新原始帧
    cv2_img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    frame[:] = cv2_img[:]



