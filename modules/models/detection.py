#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
物体检测模型初始化与处理模块

功能:
- 初始化YOLOv8模型
- 处理模型预测结果
- 管理字母、人物等检测功能

作者: Ho Alien
版本: 1.0.0
日期: 2025-06-14
"""

import torch
import cv2
import numpy as np
from ultralytics import YOLO
import time

# 从配置模块导入模型路径
from modules.config import LETTER_MODEL_PATH, PERSON_MODEL_PATH


def initialize_models():
    """
    初始化字母检测和人物检测模型
    
    返回:
        tuple: (letter_model, person_model, letter_names, person_names) 两个模型和对应的类别名称
    """
    try:
        print(f"初始化字母检测模型: {LETTER_MODEL_PATH}")
        model = YOLO(LETTER_MODEL_PATH)
        class_names = model.names
        print(f"字母检测模型加载完成，类别: {class_names}")
        
        print(f"初始化人物检测模型: {PERSON_MODEL_PATH}")
        person_model = YOLO(PERSON_MODEL_PATH)
        person_names = person_model.names
        print(f"人物检测模型加载完成，类别: {person_names}")
        
        return model, person_model, class_names, person_names
        
    except Exception as e:
        print(f"模型初始化失败: {e}")
        import traceback
        print(traceback.format_exc())
        return None


def detect_with_model(model, frame, conf_threshold=0.5, class_filter=None):
    """
    使用YOLOv8模型进行物体检测
    
    参数:
        model: YOLOv8模型
        frame: 图像帧
        conf_threshold: 置信度阈值
        class_filter: 可选的类别过滤列表
        
    返回:
        tuple: (boxes, class_ids, confidences) 检测结果
    """
    try:
        # 使用模型进行预测
        results = model.predict(source=frame, conf=conf_threshold, verbose=False)
        result = results[0]  # 只处理第一个结果
        
        # 提取预测结果
        boxes = []
        class_ids = []
        confidences = []
        
        if hasattr(result, 'boxes') and len(result.boxes) > 0:
            for box in result.boxes:
                # 检查类别过滤
                cls_id = int(box.cls.item())
                if class_filter is not None and cls_id not in class_filter:
                    continue
                
                # 获取边界框坐标 (x1, y1, x2, y2)
                x1, y1, x2, y2 = [int(i) for i in box.xyxy.squeeze().tolist()]
                conf = float(box.conf.item())
                
                boxes.append([x1, y1, x2, y2])
                class_ids.append(cls_id)
                confidences.append(conf)
        
        return boxes, class_ids, confidences
        
    except Exception as e:
        print(f"检测过程中发生错误: {e}")
        import traceback
        print(traceback.format_exc())
        return [], [], []


def filter_detections_by_class(detections, class_ids, confs, target_class_ids):
    """
    根据类别ID过滤检测结果
    
    参数:
        detections: 检测框列表
        class_ids: 类别ID列表
        confs: 置信度列表
        target_class_ids: 目标类别ID列表
        
    返回:
        tuple: (filtered_detections, filtered_class_ids, filtered_confs)
    """
    if not target_class_ids or not isinstance(target_class_ids, (list, tuple, set)):
        return detections, class_ids, confs
    
    filtered_detections = []
    filtered_class_ids = []
    filtered_confs = []
    
    for i, class_id in enumerate(class_ids):
        if class_id in target_class_ids:
            filtered_detections.append(detections[i])
            filtered_class_ids.append(class_id)
            filtered_confs.append(confs[i])
            
    return filtered_detections, filtered_class_ids, filtered_confs
