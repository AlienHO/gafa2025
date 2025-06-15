"""
OSC通信工具模块 - 提供OSC消息构建与发送功能
"""
import time
from pythonosc import udp_client, osc_bundle_builder, osc_message_builder
from modules.config import *

def setup_network():
    """
    设置OSC网络通信客户端
    
    返回:
        tuple: (osc_client, person_osc_client, vision_api_osc_client, run_osc_client) 客户端对象
    """
    # 初始化OSC客户端
    osc_client = udp_client.SimpleUDPClient(OSC_IP, OSC_PORT)
    print(f"[已初始化] OSC客户端连接到 {OSC_IP}:{OSC_PORT}")
    
    # 初始化人物检测的OSC客户端
    person_osc_client = udp_client.SimpleUDPClient(PERSON_OSC_IP, PERSON_OSC_PORT)
    print(f"[已初始化] 人物检测OSC客户端连接到 {PERSON_OSC_IP}:{PERSON_OSC_PORT}")
    
    # 初始化OpenAI Vision的OSC客户端
    vision_api_osc_client = udp_client.SimpleUDPClient(OPENAI_OSC_IP, OPENAI_OSC_PORT)
    print(f"[已初始化] 视觉API OSC客户端连接到 {OPENAI_OSC_IP}:{OPENAI_OSC_PORT}")
    
    # 初始化运动目标跟踪的OSC客户端
    run_osc_client = udp_client.SimpleUDPClient(RUN_OSC_IP, RUN_OSC_PORT)
    print(f"[已初始化] 运动目标跟踪OSC客户端连接到 {RUN_OSC_IP}:{RUN_OSC_PORT}")
    
    return osc_client, person_osc_client, vision_api_osc_client, run_osc_client


def build_osc_message(address, args):
    """
    构建OSC消息的通用函数
    
    参数:
        address: OSC地址
        args: 消息参数列表
    
    返回:
        构建好的OSC消息
    """
    msg = osc_message_builder.OscMessageBuilder(address=address)
    
    # 添加所有参数
    for arg in args:
        msg.add_arg(arg)
        
    return msg.build()


def build_osc_bundle(messages):
    """
    构建OSC消息包的通用函数
    
    参数:
        messages: 消息列表
    
    返回:
        构建好的OSC消息包
    """
    bundle = osc_bundle_builder.OscBundleBuilder(osc_bundle_builder.IMMEDIATELY)
    
    # 添加所有消息
    for msg in messages:
        bundle.add_content(msg)
    
    return bundle.build()


def send_osc_messages(client, address, args_list, port=None, bundle=True):
    """
    发送一组OSC消息的通用函数
    
    参数:
        client: OSC客户端
        address: OSC地址
        args_list: 列表，每个元素是一组参数
        port: 端口号（用于日志显示）
        bundle: 是否使用bundle发送（多条消息才需要）
    
    返回:
        发送的消息数量
    """
    if not args_list:
        return 0
    
    if len(args_list) == 1 and not bundle:
        # 单条消息且不需要bundle
        msg = build_osc_message(address, args_list[0])
        client.send(msg)
    else:
        # 多条消息或强制使用bundle
        messages = [build_osc_message(address, args) for args in args_list]
        bundle_data = build_osc_bundle(messages)
        client.send(bundle_data)
    
    if port:
        print(f"[发送OSC] 已发送 {len(args_list)} 条数据到端口 {port}")
    
    return len(args_list)


def send_vision_api_osc(vision_api_osc_client, boxes, frame_width, frame_height, sent_anything_boxes):
    """
    将OpenAI Vision生成的框和文本通过OSC发送
    
    参数:
        vision_api_osc_client: OSC客户端
        boxes: OpenAI Vision框列表 [(box, text, timestamp), ...]
        frame_width: 帧宽度
        frame_height: 帧高度
        sent_anything_boxes: 已发送的框集合
    """
    # 准备消息参数列表
    message_args_list = []
    
    # 遍历所有框
    for box, text, _ in boxes:
        # 转换为内部使用的坐标元组格式
        box_tuple = tuple(box)  # 转为元组使其可哈希化
        
        # 如果框已发送过，则跳过
        if box_tuple in sent_anything_boxes:
            continue
        
        # 计算归一化坐标
        x1, y1, x2, y2 = box
        # 计算中心点和宽高
        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2
        w = x2 - x1
        h = y2 - y1
        
        # 归一化并反转Y坐标(TouchDesigner兼容性)
        norm_cx = cx / frame_width
        norm_cy = 1.0 - (cy / frame_height)  # Y坐标反转
        norm_w = w / frame_width
        norm_h = h / frame_height
        
        # 准备消息参数
        args = [float(norm_cx), float(norm_cy), float(norm_w), float(norm_h), text]
        message_args_list.append(args)
        
        # 标记为已发送
        sent_anything_boxes.add(box_tuple)
    
    # 使用通用函数发送OSC消息
    return send_osc_messages(
        client=vision_api_osc_client,
        address=OPENAI_OSC_ADDRESS,
        args_list=message_args_list,
        port=OPENAI_OSC_PORT,
        bundle=True
    )
