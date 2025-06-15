"""
目标跟踪模块 - 提供IoU跟踪器和相关追踪逻辑
"""
import time
import numpy as np


class IoUTracker:
    """
    基于IoU的简单目标跟踪器
    """
    def __init__(self, iou_threshold=0.3, max_missed_frames=30):
        """
        初始化跟踪器
        
        参数:
            iou_threshold: IoU阈值，大于此阈值的视为同一目标
            max_missed_frames: 最大允许丢失帧数
        """
        self.iou_threshold = iou_threshold
        self.max_missed_frames = max_missed_frames
        self.next_id = 1
        self.tracks = {}  # {id: {"bbox": bbox, "missed_count": count, "class_id": class_id, "conf": conf}}
        self.track_history = {}  # {id: [{"bbox": bbox, "time": time, "class_id": class_id, "conf": conf}, ...]}
        
    def update(self, detections, class_ids, confs):
        """
        更新跟踪器状态
        
        参数:
            detections: 检测结果列表 [[x1, y1, x2, y2], ...]
            class_ids: 类别ID列表
            confs: 置信度列表
            
        返回:
            list: 跟踪结果 [(x1,y1,x2,y2,track_id,class_id,conf), ...]
        """
        # 如果无检测结果，更新所有跟踪对象的丢失帧数
        if len(detections) == 0:
            for track_id in list(self.tracks.keys()):
                self.tracks[track_id]["missed_count"] += 1
                if self.tracks[track_id]["missed_count"] > self.max_missed_frames:
                    del self.tracks[track_id]
                    # 保留轨迹历史供后续分析
            return []
        
        # 计算当前检测与现有跟踪对象的IoU矩阵
        iou_matrix = np.zeros((len(detections), len(self.tracks)))
        track_ids = list(self.tracks.keys())
        
        for d_idx, detection in enumerate(detections):
            for t_idx, track_id in enumerate(track_ids):
                iou_matrix[d_idx, t_idx] = self._calculate_iou(detection, self.tracks[track_id]["bbox"])
        
        # 检测结果匹配现有跟踪
        result = []
        tracked_dets = set()
        tracked_tracks = set()
        
        # 查找最佳匹配
        if len(self.tracks) > 0:
            for d_idx in range(len(detections)):
                # 对每个检测，找到最佳匹配的轨迹
                max_iou = self.iou_threshold
                max_track_idx = -1
                
                for t_idx, track_id in enumerate(track_ids):
                    if t_idx in tracked_tracks:
                        continue
                        
                    if iou_matrix[d_idx, t_idx] > max_iou:
                        max_iou = iou_matrix[d_idx, t_idx]
                        max_track_idx = t_idx
                
                # 如果找到匹配
                if max_track_idx != -1:
                    track_id = track_ids[max_track_idx]
                    self.tracks[track_id]["bbox"] = detections[d_idx]
                    self.tracks[track_id]["missed_count"] = 0
                    self.tracks[track_id]["class_id"] = class_ids[d_idx]
                    self.tracks[track_id]["conf"] = confs[d_idx]
                    
                    # 添加到轨迹历史
                    if track_id not in self.track_history:
                        self.track_history[track_id] = []
                    
                    # 记录当前时间和位置，便于后续分析运动方向
                    self.track_history[track_id].append({
                        "bbox": detections[d_idx],
                        "time": time.time(),
                        "class_id": class_ids[d_idx],
                        "conf": confs[d_idx]
                    })
                    
                    # 限制历史记录长度，避免内存无限增长
                    if len(self.track_history[track_id]) > 100:
                        self.track_history[track_id] = self.track_history[track_id][-100:]
                    
                    result.append((*detections[d_idx], track_id, class_ids[d_idx], confs[d_idx]))
                    tracked_dets.add(d_idx)
                    tracked_tracks.add(max_track_idx)
        
        # 创建未匹配的检测对应的新跟踪
        for d_idx in range(len(detections)):
            if d_idx not in tracked_dets:
                track_id = self.next_id
                self.next_id += 1
                
                self.tracks[track_id] = {
                    "bbox": detections[d_idx],
                    "missed_count": 0,
                    "class_id": class_ids[d_idx],
                    "conf": confs[d_idx]
                }
                
                # 创建新的轨迹历史
                self.track_history[track_id] = [{
                    "bbox": detections[d_idx],
                    "time": time.time(),
                    "class_id": class_ids[d_idx],
                    "conf": confs[d_idx]
                }]
                
                result.append((*detections[d_idx], track_id, class_ids[d_idx], confs[d_idx]))
        
        # 更新未匹配跟踪的丢失帧数
        for t_idx, track_id in enumerate(track_ids):
            if t_idx not in tracked_tracks:
                self.tracks[track_id]["missed_count"] += 1
                if self.tracks[track_id]["missed_count"] > self.max_missed_frames:
                    del self.tracks[track_id]
                    # 保留轨迹历史
        
        return result
    
    def _calculate_iou(self, box1, box2):
        """
        计算两个边界框的IoU
        
        参数:
            box1: 第一个框 [x1, y1, x2, y2]
            box2: 第二个框 [x1, y1, x2, y2]
            
        返回:
            float: IoU值
        """
        x1_1, y1_1, x2_1, y2_1 = box1
        x1_2, y1_2, x2_2, y2_2 = box2
        
        # 计算交集区域
        x1_i = max(x1_1, x1_2)
        y1_i = max(y1_1, y1_2)
        x2_i = min(x2_1, x2_2)
        y2_i = min(y2_1, y2_2)
        
        # 无交集情况
        if x2_i < x1_i or y2_i < y1_i:
            return 0.0
            
        # 计算交集面积
        intersection = (x2_i - x1_i) * (y2_i - y1_i)
        
        # 计算并集面积
        box1_area = (x2_1 - x1_1) * (y2_1 - y1_1)
        box2_area = (x2_2 - x1_2) * (y2_2 - y1_2)
        union = box1_area + box2_area - intersection
        
        # 返回IoU
        return intersection / union if union > 0 else 0
        
    def get_active_tracks(self):
        """
        获取所有当前活跃的跟踪
        
        返回:
            dict: 活跃跟踪字典 {id: {"bbox": bbox, "class_id": class_id, "conf": conf}}
        """
        return {
            track_id: {
                "bbox": track["bbox"],
                "class_id": track["class_id"],
                "conf": track["conf"]
            }
            for track_id, track in self.tracks.items()
            if track["missed_count"] == 0
        }
