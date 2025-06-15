#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
OpenAI Vision API 后台工作线程

功能:
- 处理随机图像区域分析的后台线程
- 生成随机边界框并调用大模型API进行分析
- 线程安全的结果管理

作者: Ho Alien，Leonardo Li
版本: 1.0.0
日期: 2025-06-14
"""

import time
import random
import threading
import copy
import traceback
from PIL import Image

from ..utils.image_utils import crop_frame_to_pil, encode_image_to_base64
from ..vision_api.api import send_vision_query
from ..config import (
    VISION_API_INTERVAL, VISION_API_BOX_DURATION, 
    VISION_API_MAX_ONSCREEN, VISION_API_MAX_HEIGHT_RATIO,
    OPENAI_API_KEY, OPENAI_BASE_URL, OPENAI_MODEL
)


class AnythingWorker(threading.Thread):
    """
    OpenAI Vision后台工作线程，负责定时生成随机框并调用大模型API
    """
    def __init__(self, interval=5.0, stop_event=None):
        super().__init__(daemon=True)
        self.interval = interval  # 每隔多少秒执行一次
        self.latest_frame = None  # 最新的图像帧
        self.boxes = []  # 当前有效的框 [(box, text, timestamp), ...]
        self.running = True
        self.lock = threading.Lock()  # 用于线程安全
        self.stop_event = stop_event
        
    def update_frame(self, frame):
        """
        更新最新帧
        """
        with self.lock:
            self.latest_frame = frame.copy()
            
    def get_current_boxes(self):
        """
        获取当前有效的框和描述
        """
        with self.lock:
            now = time.time()
            # 移除过期的框
            active_boxes = []
            for box, text, timestamp in self.boxes:
                if now - timestamp <= VISION_API_BOX_DURATION:  # 保留未过期的框
                    active_boxes.append((box, text, timestamp))
            
            # 更新框列表，只保留活跃的框
            self.boxes = active_boxes
            
            # 如果框数量超过最大显示数量，移除最旧的框
            while len(self.boxes) > VISION_API_MAX_ONSCREEN:
                oldest_box = min(self.boxes, key=lambda x: x[2])
                self.boxes.remove(oldest_box)
                print(f"[OpenAI Vision] 强制移除最早添加的框，保持数量在 {VISION_API_MAX_ONSCREEN} 以内")
                
            return copy.deepcopy(self.boxes)  # 返回一个深拷贝，避免线程冲突
            
    def generate_random_box(self, frame):
        """
        在图像上生成一个合理的随机边界框
        """
        h, w = frame.shape[:2]
        # 随机框的最小尺寸
        min_size = min(w, h) // 10
        
        # 随机框的中心点
        cx = random.randint(min_size, w - min_size)
        cy = random.randint(min_size, h - min_size)
        
        # 随机大小，但要保证不会太大或太小
        # 宽度在图像宽度的10%~40%之间
        box_w = random.randint(int(w * 0.1), int(w * 0.4))
        # 高度在框宽度的0.5~1.5倍之间，且不超过图像高度的1/4
        box_h = min(random.randint(int(box_w * 0.5), int(box_w * 1.5)), int(h * VISION_API_MAX_HEIGHT_RATIO))
        
        # 确保框不超出图像边界
        x1 = max(0, cx - box_w // 2)
        y1 = max(0, cy - box_h // 2)
        x2 = min(w, x1 + box_w)
        y2 = min(h, y1 + box_h)
        
        return [x1, y1, x2, y2]
    
    def process_frame_with_llm(self, frame):
        """
        随机生成边界框并使用LLM处理，保存结果
        """
        print("[OpenAI Vision] 调用process_frame_with_llm处理新帧")
        print(f"[OpenAI Vision] 配置: API URL={OPENAI_BASE_URL}, 有效期={VISION_API_BOX_DURATION}s")
        
        try:
            # 检查帧是否有效
            if frame is None:
                print("[OpenAI Vision] 帧为None，跳过处理")
                return
            
            # 当前时间
            now = time.time()
            
            # 如果达到最大显示数量，检查是否有框已接近其有效期
            with self.lock:
                # 检查当前有效的框
                current_boxes_with_time = [(b, t) for b, _, t in self.boxes if now - t <= VISION_API_BOX_DURATION]
                
                # 如果已经达到最大显示数量
                if len(current_boxes_with_time) >= VISION_API_MAX_ONSCREEN:
                    # 找到最早添加的框及其时间
                    if current_boxes_with_time:
                        oldest_box_time = min(current_boxes_with_time, key=lambda x: x[1])[1]
                        time_since_oldest = now - oldest_box_time
                        
                        # 如果最旧的框距离有效期的一半还很远，则跳过生成
                        if time_since_oldest < VISION_API_BOX_DURATION / 2:
                            print(f"[OpenAI Vision] 已达到最大显示数量 ({VISION_API_MAX_ONSCREEN})，最旧框年龄: {time_since_oldest:.1f}秒，跳过生成")
                            return
                        else:
                            print(f"[OpenAI Vision] 已达到最大显示数量，但最旧框已近有效期一半 ({time_since_oldest:.1f}秒)，继续生成")
                    else:
                        print(f"[OpenAI Vision] 强制更新框")
                
            # 随机生成1个框
            box_count = 1  # 简化为每次只生成一个框
            print("[OpenAI Vision] 生成随机边界框")
            
            # 生成随机框
            box = self.generate_random_box(frame)
            if box is None:
                print("[OpenAI Vision] 无法生成有效的随机边界框")
                return
    
            print(f"[OpenAI Vision] 生成了有效边界框: {box}")
    
            # 裁剪图像并转换为 PIL Image
            try:
                x1, y1, x2, y2 = box
                print(f"[OpenAI Vision] 尝试裁剪图像区域: ({x1}, {y1}, {x2}, {y2})")
                
                # 裁剪区域并转为PIL格式
                pil_image = crop_frame_to_pil(frame, box)
                print("[OpenAI Vision] 裁剪成功，将图像转换为Base64")
                
                # 转换为base64编码
                img_base64 = encode_image_to_base64(pil_image)
                print(f"[OpenAI Vision] Base64转换成功，长度: {len(img_base64) if img_base64 else 0}")
                
                # 生成描述的提示词 - 采用与成功示例相同的更丰富提示词
                prompt = "请用精简且可爱的超短句（15字以内）根据框内事物体现你的日常哲学迷思；或者表现你对框选内容的好奇和猜测；哲思属性或憨憨可爱属性二选一，只输出话语不要写哲思版什么的，去对框内的东西发问，或者发呆，或者一起玩。注意，哲思和可爱的人格尽量不要在同一个描述中出现"
                print(f"[OpenAI Vision] 使用提示词: {prompt[:20]}...")
                
                # 创建API调用线程
                def api_call_thread(box_coords, img_base64):
                    try:
                        print("[OpenAI Vision] 准备调用API...")
                        # 检查API配置
                        print(f"[OpenAI] API配置: URL={OPENAI_BASE_URL}, 模型={OPENAI_MODEL}")
                        
                        # 调用OpenAI兼容API
                        print(f"[OpenAI Vision] 发送API请求: 模型={OPENAI_MODEL}, 图像大小={len(img_base64)}字节")
                        print(f"[OpenAI Vision] 开始时间: {time.strftime('%H:%M:%S')}")
                        response_text = send_vision_query(prompt, img_base64, api_key=OPENAI_API_KEY, base_url=OPENAI_BASE_URL, model=OPENAI_MODEL)
                        print(f"[OpenAI Vision] 完成时间: {time.strftime('%H:%M:%S')}")
                        print(f"[OpenAI Vision] 收到API响应: \"{response_text}\"")
                        print(f"[OpenAI Vision API] ★★★ 成功获取模型回复! ★★★")
                        
                        # 保存结果
                        with self.lock:
                            current_time = time.time()
                            # 清理过期的框
                            self.boxes = [(b, t, ts) for b, t, ts in self.boxes
                                          if current_time - ts <= VISION_API_BOX_DURATION]
                            
                            # 如果当前框数量已达到或超过最大显示数量，移除最早的框
                            while len(self.boxes) >= VISION_API_MAX_ONSCREEN:
                                oldest_box = min(self.boxes, key=lambda x: x[2])
                                self.boxes.remove(oldest_box)
                                print(f"[OpenAI Vision] 移除最早添加的框，保持最大数量为 {VISION_API_MAX_ONSCREEN}")
                                
                            # 添加新框
                            self.boxes.append((box_coords, response_text, current_time))
                            print(f"[OpenAI Vision] 成功添加新框，当前框数量: {len(self.boxes)}")
                    except Exception as e:
                        print(f"[OpenAI Vision] API调用失败: {e}")
                        import traceback
                        traceback.print_exc()
                
                # 启动线程执行 API 调用
                print("[OpenAI Vision] 启动API调用线程")
                api_thread = threading.Thread(target=api_call_thread, args=(box, img_base64))
                api_thread.daemon = True
                api_thread.start()
                print(f"[OpenAI Vision] API调用线程已启动: {api_thread.name}")
                
            except Exception as e:
                print(f"[OpenAI Vision] 图像处理失败: {str(e)}")
                import traceback
                print(traceback.format_exc())
        except Exception as e:
            print(f"[OpenAI Vision] process_frame_with_llm 方法异常: {str(e)}")
            import traceback
            print(traceback.format_exc())
    
    def run(self):
        """
        线程主循环
        """
        try:
            print("[OpenAI Vision] 工作线程已启动，开始运行...")
            print(f"[OpenAI Vision] 配置: 处理间隔={self.interval}秒, API={OPENAI_BASE_URL}")
            print(f"[OpenAI Vision] self.running = {self.running}, stop_event.is_set() = {self.stop_event.is_set() if self.stop_event else False}")
            
            last_process_time = 0
            loop_count = 0
            
            while self.running and not (self.stop_event and self.stop_event.is_set()):
                now = time.time()
                loop_count += 1
                
                if loop_count % 50 == 0:  # 每50次循环输出一次状态信息
                    print(f"[OpenAI Vision] 线程运行中... 已循环 {loop_count} 次")
                
                # 检查是否到达处理间隔时间
                if now - last_process_time >= self.interval:
                    with self.lock:  # 线程安全地访问 latest_frame
                        current_frame = self.latest_frame
                    
                    # 如果有有效的帧，进行处理
                    if current_frame is not None:
                        print(f"[OpenAI Vision] 开始处理帧 - 时间间隔: {now - last_process_time:.2f}s")
                        try:
                            self.process_frame_with_llm(current_frame)
                        except Exception as e:
                            print(f"[OpenAI Vision] 处理帧时发生异常: {e}")
                    else:
                        print("[OpenAI Vision] 无有效帧可处理")
                        
                    # 更新上次处理时间
                    last_process_time = now
                    
                # 小睡等待一下，减少CPU使用
                time.sleep(0.1)
        except Exception as e:
            print(f"[OpenAI Vision] 线程运行异常终止: {e}")
            import traceback
            print(traceback.format_exc())
