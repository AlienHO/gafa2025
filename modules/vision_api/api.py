#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
OpenAI兼容的视觉API接口封装

功能:
- 封装对OpenAI兼容的API调用，特别是包含图像的请求
- 支持自定义模型、API密钥和基础URL
- 提供统一的错误处理和日志输出

作者: Ho Alien
版本: 1.0.0
日期: 2025-06-13
"""

import requests
import json
import base64
import time
import logging
import traceback

# API配置常量，可被其他模块导入
DEFAULT_API_KEY = "sk-AAcR9Ipm05waI9qOxFSj0IsErNFLrGcQcLvLKMHHvNYHDHi8"
DEFAULT_BASE_URL = "https://api.41box.com/v1/chat/completions"
DEFAULT_MODEL = "claude-3-5-haiku-20241022"

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("OpenAIVision")

class OpenAIVisionAPI:
    """OpenAI兼容的视觉API接口封装类"""
    
    def __init__(self, api_key="sk-AAcR9Ipm05waI9qOxFSj0IsErNFLrGcQcLvLKMHHvNYHDHi8", base_url="https://api.41box.com/v1/chat/completions", model="claude-3-5-haiku-20241022"):
        """
        初始化OpenAI API客户端
        
        参数:
            api_key: API密钥，可以为空
            base_url: API基础URL
            model: 使用的模型名称
        """
        self.api_key = api_key
        self.base_url = base_url
        self.model = model
        logger.info(f"初始化OpenAI视觉API客户端: URL={base_url}, 模型={model}")
    
    def send_image_query(self, prompt, image_base64=None, max_tokens=300):
        """
        发送包含图像的查询到OpenAI兼容API
        
        参数:
            prompt: 发送给模型的提示文本
            image_base64: Base64编码的图像，可选
            max_tokens: 返回的最大令牌数
            
        返回:
            str: 模型的文本响应
        """
        # 准备请求头
        headers = {
            "Content-Type": "application/json"
        }
        
        # 如果提供API密钥，添加到请求头
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        
        # 构建消息内容
        messages = [
            {"role": "system", "content": "你是一个善于描述图像场景的AI助手，请用精简、富有情感的语言回复。"}, 
            {"role": "user", "content": []}
        ]
        
        # 添加文本提示
        messages[-1]["content"].append({"type": "text", "text": prompt})
        
        # 如果有图像，将其添加到消息中
        if image_base64:
            # 确保base64字符串没有前缀
            if image_base64.startswith("data:image"):
                # 删除MIME前缀，只保留base64部分
                image_base64 = image_base64.split(",", 1)[1]
            
            # 添加图像到消息
            messages[-1]["content"].append({
                "type": "image_url", 
                "image_url": {
                    "url": f"data:image/png;base64,{image_base64}"
                }
            })
        
        # 准备请求体
        payload = {
            "model": self.model,
            "messages": messages,
            "max_tokens": max_tokens
        }
        
        logger.info(f"发送请求到API: {self.base_url}")
        if image_base64:
            logger.info(f"包含图像数据，图像数据长度: {len(image_base64)}字符")
        
        try:
            # 发送请求
            logging.info(f"发送API请求: 模型={self.model}, 图像={'包含图像' if image_base64 else '无图像'}, 提示词={prompt[:30]}...")
            
            response = requests.post(
                self.base_url,
                headers=headers,
                json=payload,
                timeout=30
            )
            
            # 记录API响应状态码
            logging.info(f"成功获取API响应: HTTP状态码={response.status_code}")
            
            response.raise_for_status()
            response_data = response.json()
            
            logger.info("成功获取API响应")
            
            # 处理响应
            if 'choices' in response_data and len(response_data['choices']) > 0:
                content = response_data['choices'][0]['message']['content']
                # 记录接收到的回复（截断过长的内容）
                logging.info(f"[OpenAI Vision API] 成功接收回复: {content[:100]}...")
                # 记录使用的token数量
                if 'usage' in response_data:
                    usage = response_data['usage']
                    logging.info(f"[OpenAI Vision API] Token使用情况: 提示词={usage.get('prompt_tokens', 0)}, 补全={usage.get('completion_tokens', 0)}, 总计={usage.get('total_tokens', 0)}")
                return content
            else:
                logging.warning(f"API响应缺少预期的内容结构: {response_data}")
                return "API响应格式异常..."
        
        except requests.exceptions.RequestException as e:
            error_msg = f"API连接失败: {str(e)}"
            logger.error(error_msg)
            return "连接失败..."
        
        except ValueError as e:
            error_msg = f"API响应格式不正确: {str(e)}"
            logger.error(error_msg)
            return "处理出错..."
        
        except Exception as e:
            error_msg = f"发生未知错误: {str(e)}"
            logger.error(error_msg)
            return "处理遇到问题..."


# 提供一个便捷的全局函数用于直接调用
def send_vision_query(prompt, image_base64=None, api_key="sk-AAcR9Ipm05waI9qOxFSj0IsErNFLrGcQcLvLKMHHvNYHDHi8", base_url="https://api.41box.com/v1/chat/completions", model="claude-3-5-haiku-20241022", max_tokens=300):
    """
    发送包含图像的查询到OpenAI兼容API的便捷函数
    
    参数:
        prompt: 发送给模型的提示文本
        image_base64: Base64编码的图像，可选
        api_key: API密钥，可选
        base_url: API基础URL，可选
        model: 使用的模型名称，可选
        max_tokens: 返回的最大令牌数，可选
        
    返回:
        str: 模型的文本响应
    """
    api_client = OpenAIVisionAPI(api_key=api_key, base_url=base_url, model=model)
    return api_client.send_image_query(prompt, image_base64, max_tokens)
