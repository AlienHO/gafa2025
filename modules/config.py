"""
配置模块 - 存储系统中使用的全局配置参数
"""
import time
import os
from PIL import ImageFont

# 全局变量用于记录上次数据清理的时间
LAST_DATA_CLEANUP_TIME = 0

# 全局预加载中文字体，程序启动时只加载一次
try:
    CHINESE_FONT = ImageFont.truetype("simhei.ttf", 24)
    print("成功加载中文字体 'simhei.ttf'")
except IOError:
    CHINESE_FONT = ImageFont.load_default()
    print("警告: 找不到中文字体文件 'simhei.ttf'，使用默认字体。文本可能无法正确显示中文。")

# ==================== 配置参数 ====================
# 模型配置
# GAF姿势识别模型切换
GAF_MODEL_TYPE = "32"  # 可选: "0", "72", "63", "32"
GAF_MODEL_PATHS = {
    "0": "models/GAFA.pt",
    "32": "models/GAFA32.pt",
    "63": "models/GAFA63.pt",
    "72": "models/GAFA72.pt"
}
LETTER_MODEL_PATH = GAF_MODEL_PATHS.get(GAF_MODEL_TYPE, "models/GAFA.pt")  # GAF字母检测模型路径
PERSON_MODEL_PATH = "models/yolov8n.pt"      # 人物检测模型路径
CONFIDENCE_THRESHOLD = 0.2              # 检测置信度阈值
IOU_THRESHOLD = 0.8                     # 非极大值抑制IoU阈值
MAX_MISSED_FRAMES = 30                  # 目标跟踪最大丢失帧数

# OpenAI API配置
OPENAI_API_KEY = "sk-AAcR9Ipm05waI9qOxFSj0IsErNFLrGcQcLvLKMHHvNYHDHi8"  # OpenAI API密钥
OPENAI_BASE_URL = "https://api.41box.com/v1/chat/completions"  # 使用兼容API的基础URL
OPENAI_MODEL = "claude-3-5-sonnet-20240620"  # 使用的模型
VISION_API_INTERVAL = 1.0               # OpenAI API请求间隔(秒)
VISION_API_ENABLED = True               # 是否启用OpenAI Vision API
VISION_API_IMAGE_QUALITY = 75           # 图像JPEG压缩质量
VISION_API_MAX_SIZE = 600               # 最大图像尺寸
VISION_API_BOX_DURATION = 5.0          # 框持续显示时间(秒)
VISION_API_MAX_ONSCREEN = 6             # 屏幕上最多同时显示多少个框
VISION_API_MAX_HEIGHT_RATIO = 0.25      # 框最大高度占图像高度的比例

# 摄像头设置
CAMERA_ID = 1                           # 摄像头设备ID
FPS = 15                                # 每秒帧数

# OSC通信配置
OSC_IP = "127.0.0.1"                   # 字母检测OSC服务器IP
OSC_PORT = 8000                        # 字母检测OSC端口
OSC_ADDRESS = "/camera/detect"          # 字母检测OSC地址
PERSON_OSC_IP = "127.0.0.1"            # 人物检测OSC服务器IP
PERSON_OSC_PORT = 10000                # 人物检测OSC端口
PERSON_OSC_ADDRESS = "/person"         # 人物检测OSC地址


# OpenAI兼容API OSC配置
OPENAI_OSC_IP = "127.0.0.1"    # 图像生成API发送OSC地址
OPENAI_OSC_PORT = 7000         # 图像生成API发送OSC端口
OPENAI_OSC_ADDRESS = "/anything"  # 图像生成API OSC地址

# 运动目标跟踪OSC配置
RUN_OSC_IP = "127.0.0.1"         # 运动目标跟踪OSC地址
RUN_OSC_PORT = 5000             # 运动目标跟踪OSC端口
RUN_OSC_ADDRESS = "/run"          # 运动目标跟踪OSC地址

# 稳定性阈值配置
STABLE_CONF_THRESHOLD = 0.35   # 稳定GAF姿势的置信度阈值
STABLE_TIME_THRESHOLD = 0.5   # 检测持续多少秒后认为是稳定的
RESEND_COOLDOWN = 5.0         # 同一目标多少秒内不重复发送
SEND_ONLY_STABLE = True       # 是否只发送稳定的目标
SEND_ONLY_CHANGES = True      # 是否只在首次检测或状态变化时发送

# 类别名称定义
CLASS_NAME_MAP = {
    "G": 0,
    "A": 1,
    "F": 2
}

# 人物检测类别ID（在YOLO模型中person的类别ID是0）
PERSON_CLASS_ID = 0

# 颜色定义 (BGR格式)
LETTER_COLORS = {
    0: (0, 255, 0),    # G: 绿色
    1: (0, 0, 255),    # A: 红色
    2: (255, 0, 0)     # F: 蓝色
}

# 调试配置
DEBUG = False  # 是否打印调试信息
DISPLAY_SCALE = 1.0  # 显示窗口缩放比例

# 数据清理配置
DATA_CLEANUP_INTERVAL = 30.0  # 数据清理的时间间隔（秒）
DATA_MAX_AGE = 60.0  # 数据过期时间（秒）
MAX_BOXES_BEFORE_CLEANUP = 1000  # 清理发送框记录的阈值
