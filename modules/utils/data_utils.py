"""
数据处理工具模块 - 提供数据管理和清理功能
"""
import time
from modules.config import LAST_DATA_CLEANUP_TIME, DATA_CLEANUP_INTERVAL, DATA_MAX_AGE, MAX_BOXES_BEFORE_CLEANUP


def cleanup_historical_data(target_positions, sent_anything_boxes, max_age_seconds=DATA_MAX_AGE):
    """
    清理过期的历史数据记录
    
    参数:
        target_positions: 目标位置历史字典 {id: {"last_x": x, "last_time": time, ...}}
        sent_anything_boxes: 已发送的框集合
        max_age_seconds: 数据过期时间（秒）
        
    返回:
        (cleaned_positions, cleaned_boxes): 清理的目标数量和框数量
    """
    current_time = time.time()
    positions_to_remove = []
    
    # 检查并标记过期的目标位置记录
    for track_id, data in target_positions.items():
        if "last_time" in data and current_time - data["last_time"] > max_age_seconds:
            positions_to_remove.append(track_id)
    
    # 清理过期目标
    for track_id in positions_to_remove:
        del target_positions[track_id]
    
    # 注意：sent_anything_boxes是一个集合，我们无法跟踪当前框的创建时间
    # 为了避免内存无限增长，我们定期完全清空它
    # 这意味着可能会重新发送一些框，但这是可接受的
    boxes_count = len(sent_anything_boxes)
    if boxes_count > MAX_BOXES_BEFORE_CLEANUP:  # 设置一个阈值，避免过多框积累
        sent_anything_boxes.clear()
        
    return len(positions_to_remove), boxes_count if boxes_count > MAX_BOXES_BEFORE_CLEANUP else 0


def should_run_cleanup(interval_seconds=DATA_CLEANUP_INTERVAL):
    """
    判断是否应该运行数据清理
    
    参数:
        interval_seconds: 清理间隔时间（秒）
        
    返回:
        True如果应该运行清理，否则False
    """
    global LAST_DATA_CLEANUP_TIME
    current_time = time.time()
    
    if current_time - LAST_DATA_CLEANUP_TIME > interval_seconds:
        LAST_DATA_CLEANUP_TIME = current_time
        return True
    return False


def get_letter_color(class_id):
    """
    根据类别ID获取字母颜色
    
    参数:
        class_id: 类别ID
        
    返回:
        tuple: BGR颜色元组
    """
    from modules.config import LETTER_COLORS
    
    return LETTER_COLORS.get(class_id, (255, 255, 255))  # 默认白色


def get_letter_name_by_id(class_id):
    """
    根据类别ID获取字母名称
    
    参数:
        class_id: 类别ID
        
    返回:
        str: 字母名称
    """
    letter_map = {0: "G", 1: "A", 2: "F"}
    return letter_map.get(class_id, "未知")
