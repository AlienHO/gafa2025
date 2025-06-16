#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GAF字母检测与追踪系统主程序入口
(Camera-based GAF Letter Detection and Tracking System)

功能特点：
- 基于YOLOv8实时检测G、A、F字母并进行追踪
- ID持久化跟踪与稳定性判断 
- 通过OSC协议发送检测结果到其他软件（如TouchDesigner）
- 集成人体检测功能，同时支持字母和人物的检测与跟踪
- 整合OpenAI Vision功能，通过大模型随机生成画面区域描述
- 模块化架构，易于维护和扩展

作者: Ho Alien，
版本: 2.0.0
日期: 2025-06-14
"""

import os
import cv2
import time
import threading
import signal
import sys

# 导入配置模块
from modules.config import *

# 导入模型相关模块
from modules.models.detection import initialize_models, detect_with_model, filter_detections_by_class
from modules.models.tracker import IoUTracker

# 导入工具模块
from modules.utils.osc_utils import setup_network, send_osc_messages, send_vision_api_osc
from modules.utils.image_utils import setup_camera, draw_anything_boxes
from modules.utils.data_utils import cleanup_historical_data, should_run_cleanup

# 导入Vision API模块
from modules.vision_api.api import OpenAIVisionAPI
from modules.vision_api.worker import AnythingWorker


class GAFDetectionSystem:
    """GAF字母检测系统主类"""
    
    def __init__(self):
        """初始化系统"""
        print("正在初始化GAF字母检测系统...")
        
        # 系统运行状态
        self.running = True
        
        # 初始化模型
        print("加载检测模型...")
        model_result = initialize_models()
        if model_result is None:
            raise RuntimeError("模型初始化失败")
        self.letter_model, self.person_model, self.letter_names, self.person_names = model_result
        
        # 初始化摄像头
        print("初始化摄像头...")
        camera_result = setup_camera()
        if camera_result is None:
            raise RuntimeError("摄像头初始化失败")
        self.cap, self.frame_width, self.frame_height = camera_result
        
        # 初始化OSC客户端
        print("设置网络通信...")
        osc_client, person_osc_client, vision_api_osc_client, run_osc_client = setup_network()
        self.clients = {
            'letter': osc_client,
            'person': person_osc_client,
            'vision_api': vision_api_osc_client,
            'run': run_osc_client
        }
        
        # 初始化追踪器
        self.tracker = IoUTracker(iou_threshold=IOU_THRESHOLD, max_missed_frames=MAX_MISSED_FRAMES)
        
        # 初始化数据存储
        self.target_positions = {}  # 目标位置历史
        self.sent_anything_boxes = set()  # 已发送的Vision API框
        
        # 跟踪目标状态的数据结构
        self.first_detection_time = {}  # 目标首次检测时间
        self.last_sent_time = {}        # 目标上次发送时间
        self.last_sent_state = {}       # 目标上次发送的状态
        
        # 初始化OpenAI Vision功能
        if VISION_API_ENABLED:
            print("初始化OpenAI Vision功能...")
            self.vision_api = OpenAIVisionAPI(
                api_key=OPENAI_API_KEY,
                base_url=OPENAI_BASE_URL,
                model=OPENAI_MODEL
            )
            self.vision_worker = AnythingWorker(interval=VISION_API_INTERVAL)
            self.vision_worker.start()
        else:
            self.vision_api = None
            self.vision_worker = None
        
        print("系统初始化完成!")
    
    def signal_handler(self, sig, frame):
        """处理系统信号，优雅退出"""
        print("\n正在退出系统...")
        self.running = False
        if self.vision_worker:
            self.vision_worker.running = False
        sys.exit(0)
    
    def process_frame(self, frame):
        """处理单帧图像"""
        current_time = time.time()
        
        # 更新Vision Worker的帧
        if self.vision_worker:
            self.vision_worker.update_frame(frame)
        
        # 字母检测
        letter_detections, letter_class_ids, letter_confs = detect_with_model(
            self.letter_model, frame, CONFIDENCE_THRESHOLD, list(CLASS_NAME_MAP.values())
        )
        
        # 人物检测
        person_detections, person_class_ids, person_confs = detect_with_model(
            self.person_model, frame, CONFIDENCE_THRESHOLD, [PERSON_CLASS_ID]
        )
        
        # 更新追踪器
        tracked_objects = self.tracker.update(letter_detections, letter_class_ids, letter_confs)
        
        # 发送字母检测结果
        if tracked_objects:
            self.send_letter_detection_results(tracked_objects)
        
        # 发送人物检测结果
        if person_detections:
            self.send_person_detection_results(person_detections, person_confs)
        
        # 处理Vision API结果
        vision_boxes = []
        if self.vision_worker:
            vision_boxes = self.vision_worker.get_current_boxes()
            if vision_boxes:
                # 发送Vision API结果
                send_vision_api_osc(
                    self.clients['vision_api'], vision_boxes,
                    self.frame_width, self.frame_height, self.sent_anything_boxes
                )
        
        # 绘制检测结果
        display_frame = frame.copy()
        
        # 绘制字母检测框
        self.draw_detection_boxes(display_frame, tracked_objects, self.letter_names, list(CLASS_NAME_MAP.keys()))
        
        # 绘制人物检测框
        self.draw_person_boxes(display_frame, person_detections)
        
        # 绘制Vision API框
        if vision_boxes:
            draw_anything_boxes(display_frame, vision_boxes)
        
        # 定期清理历史数据
        if should_run_cleanup():
            cleanup_historical_data(self.target_positions, self.sent_anything_boxes)
        
        return display_frame
    

    
    def send_letter_detection_results(self, tracked_objects):
        """发送字母检测结果 - 只发送类别ID和坐标信息，只发送稳定的目标"""
        current_time = time.time()
        
        # 检查tracked_objects的形状
        for obj in tracked_objects:
            # 根据tracker返回的7元组：(x1,y1,x2,y2,track_id,class_id,conf)
            if len(obj) == 7:
                x1, y1, x2, y2, track_id, class_id, conf = obj
                
                # 计算字母中心点和宽高
                w = x2 - x1
                h = y2 - y1
                cx = x1 + w / 2  # 中心点X坐标
                cy = y1 + h / 2  # 中心点Y坐标
                
                # 与人物检测相同的归一化方式
                norm_x = float(cx / self.frame_width)
                norm_y = float(1.0 - (cy / self.frame_height))  # Y坐标反转
                norm_w = float(w / self.frame_width)
                norm_h = float(h / self.frame_height)
                
                # 当前状态
                current_state = (class_id, norm_x, norm_y, norm_w, norm_h)
                
                # 记录首次检测时间
                if track_id not in self.first_detection_time:
                    self.first_detection_time[track_id] = current_time
                
                # 新的稳定性判断逻辑：检测到同一ID持续 STABLE_TIME_THRESHOLD 秒
                detection_duration = current_time - self.first_detection_time[track_id]
                stable = (detection_duration >= STABLE_TIME_THRESHOLD) and (conf >= STABLE_CONF_THRESHOLD)
                
                # 判断是否需要发送
                should_send = False
                
                # 如果配置为只发送稳定目标，检查稳定性
                if SEND_ONLY_STABLE and not stable:
                    continue  # 跳过不稳定的目标
                
                # 检查是否首次检测
                is_first_detection = track_id not in self.last_sent_time
                
                # 检查已经过去了RESEND_COOLDOWN时间
                cooldown_passed = True
                if track_id in self.last_sent_time:
                    time_since_last_sent = current_time - self.last_sent_time[track_id]
                    cooldown_passed = time_since_last_sent >= RESEND_COOLDOWN
                
                # 检查状态是否发生变化
                state_changed = True
                if track_id in self.last_sent_state:
                    # 比较类别及位置是否变化
                    last_state = self.last_sent_state[track_id]
                    # 如果坐标差异非常小，可以认为位置未变
                    state_changed = (last_state[0] != current_state[0]) or \
                                  (abs(last_state[1] - current_state[1]) > 0.01) or \
                                  (abs(last_state[2] - current_state[2]) > 0.01) or \
                                  (abs(last_state[3] - current_state[3]) > 0.01) or \
                                  (abs(last_state[4] - current_state[4]) > 0.01)
                
                # 根据配置和状态决定是否发送
                if is_first_detection or (cooldown_passed and (not SEND_ONLY_CHANGES or state_changed)):
                    # 发送OSC消息
                    self.clients['letter'].send_message(OSC_ADDRESS, 
                        [class_id, norm_x, norm_y, norm_w, norm_h])
                    
                    # 更新发送记录
                    self.last_sent_time[track_id] = current_time
                    self.last_sent_state[track_id] = current_state
    
    def send_person_detection_results(self, person_detections, person_confs=None):
        """发送人物检测结果"""
        for i, (x1, y1, x2, y2) in enumerate(person_detections):
            # 获取置信度，如果available
            conf = person_confs[i] if person_confs and i < len(person_confs) else 1.0
            
            # 计算物体中心点和宽高
            w = x2 - x1
            h = y2 - y1
            cx = x1 + w / 2  # 中心点X坐标
            cy = y1 + h / 2  # 中心点Y坐标
            
            # 归一化坐标 - 与old.py保持一致
            norm_x = float(cx / self.frame_width)
            norm_y = float(1.0 - (cy / self.frame_height))  # Y坐标反转
            norm_w = float(w / self.frame_width)
            norm_h = float(h / self.frame_height)
            
            # 发送OSC消息: x, y, width, height, confidence (与old.py一致)
            self.clients['person'].send_message(PERSON_OSC_ADDRESS, 
                [norm_x, norm_y, norm_w, norm_h, float(conf)])
    

    
    def draw_detection_boxes(self, frame, tracked_objects, class_names, class_ids):
        """绘制字母检测框"""
        for obj in tracked_objects:
            # 如果是tracker返回的7元组 (x1, y1, x2, y2, track_id, class_id, conf)
            if len(obj) == 7:
                x1, y1, x2, y2, track_id, class_id, conf = obj
                
                # 判断目标是否稳定
                current_time = time.time()
                detection_duration = 0
                if track_id in self.first_detection_time:
                    detection_duration = current_time - self.first_detection_time[track_id]
                stable = (detection_duration >= STABLE_TIME_THRESHOLD) and (conf >= STABLE_CONF_THRESHOLD)
                
                # 获取颜色
                color = LETTER_COLORS.get(class_id, (0, 255, 0))  # 默认绿色
                
                # 根据稳定性决定线条粗细和颜色深浅
                if not stable:
                    # 不稳定目标：暗色细线
                    # 将颜色减淡到40%
                    darkened_color = tuple(int(c * 0.4) for c in color)
                    thickness = 1
                else:
                    # 稳定目标：正常颜色粗线
                    darkened_color = color
                    thickness = 2
                
                if True:  # 所有目标都显示
                    class_index = class_ids.index(class_id) if class_id in class_ids else 0
                    class_name = class_names[class_index]
                    cv2.rectangle(frame, (x1, y1), (x2, y2), darkened_color, thickness)
                    cv2.putText(frame, f"{class_name} ({track_id})", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, darkened_color, thickness)
    
    def draw_person_boxes(self, frame, person_detections):
        """绘制人物检测框"""
        # 人物检测框使用暗紫色细线 (128, 0, 128)
        person_color = (128, 0, 128)  # 暗紫色 (BGR)
        thickness = 1  # 细线
        
        for i, (x1, y1, x2, y2) in enumerate(person_detections):
            cv2.rectangle(frame, (x1, y1), (x2, y2), person_color, thickness)
            cv2.putText(frame, f"Person {i}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, person_color, thickness)
    
    def run(self):
        """运行主循环"""
        print("开始运行检测系统...")
        print("按 'q' 键退出程序")
        
        # 注册信号处理器
        signal.signal(signal.SIGINT, self.signal_handler)
        
        frame_count = 0
        start_time = time.time()
        
        try:
            while self.running:
                ret, frame = self.cap.read()
                if not ret:
                    print("无法读取摄像头画面")
                    break
                
                # 处理帧
                display_frame = self.process_frame(frame)
                
                # 显示结果
                cv2.imshow('GAF Letter Detection System', display_frame)
                
                # 计算并显示FPS
                frame_count += 1
                if frame_count % 30 == 0:
                    elapsed = time.time() - start_time
                    fps = frame_count / elapsed
                    print(f"FPS: {fps:.1f}")
                
                # 检查退出条件
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                    
        except KeyboardInterrupt:
            print("\n用户中断程序")
        except Exception as e:
            print(f"程序运行出错: {e}")
            import traceback
            traceback.print_exc()
        finally:
            self.cleanup()
    
    def cleanup(self):
        """清理资源"""
        print("正在清理资源...")
        
        if hasattr(self, 'cap') and self.cap:
            self.cap.release()
        
        if self.vision_worker:
            self.vision_worker.running = False
            self.vision_worker.join(timeout=2.0)
        
        cv2.destroyAllWindows()
        print("资源清理完成")


def main():
    """主程序入口"""
    try:
        # 创建并运行检测系统
        system = GAFDetectionSystem()
        system.run()
        
    except RuntimeError as e:
        print(f"系统初始化失败: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"程序异常退出: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()