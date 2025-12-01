#!/usr/bin/env python3
"""
YOLOv11x 绊线人数统计算法服务
专门用于绊线检测和跨线计数
符合EasyDarwin智能分析插件规范
"""
import os
import argparse
import json
import time
import threading
import signal
import sys
import tempfile
import socketserver
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse, parse_qs
import requests
import urllib.request
import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont
from predict import init_acl_resource, load_om_model, om_infer, release_acl_resource
from collections import defaultdict
from pathlib import Path
from datetime import datetime
import uuid
import atexit
try:
    from scipy.optimize import linear_sum_assignment as _hungarian_assignment
except ImportError:
    _hungarian_assignment = None

# 尝试导入 ThreadingHTTPServer，如果不存在则创建
try:
    from http.server import ThreadingHTTPServer
except ImportError:
    # Python < 3.7 的兼容方案
    class ThreadingHTTPServer(socketserver.ThreadingMixIn, HTTPServer):
        daemon_threads = True

# 全局配置
CONFIG = {
    'service_id': 'yolo11x_line_crossing',
    'name': 'YOLOv11x绊线人数统计算法',
    'version': '1.0.0',
    'model_path': './weight/best.om',
    'task_types': ['绊线人数统计'],
    'port': 7903,  # 使用不同的端口
    'host': '0.0.0.0',
    'easydarwin_url': '172.16.5.207:5066',
    'heartbeat_interval': 30,  # 秒
    'device_id': 0,  # Ascend NPU设备ID
    # 批处理配置
    'batch_size': 8,
    'batch_timeout': 0.1,
    'enable_batching': True,
    'max_queue_size': 100,
    # 视频保存配置
    'enable_video_save': False,  # 是否保存过程视频（默认关闭）
    'video_save_dir': './videos',  # 视频保存目录
    'video_fps': 25,  # 视频帧率
    'video_segment_duration': 60,  # 每个视频片段时长（秒），默认1分钟
    'video_segment_max_size_mb': 500,  # 每个视频片段最大大小（MB），默认500MB
    # 视频绘制配置（默认都开启，可通过algo_config覆盖）
    'video_draw_trajectory': True,  # 是否绘制跟踪轨迹
    'video_draw_line_config': True,  # 是否绘制绊线配置
    'video_draw_stats': True,  # 是否绘制统计信息
}

# 全局变量
MODEL = None
OM_LOADED = False
CLASS_NAMES = ['head']  # OM模型类别名称
RUNNING = True
HEARTBEAT_THREAD = None
REGISTER_THREAD = None
REGISTERED = False  # 注册状态标志
TRACKER_MANAGER = None
TRACKER_LOCK = threading.Lock()
VIDEO_WRITERS = {}  # {task_id: {'writer': cv2.VideoWriter, 'path': Path, 'start_time': float, 'frame_count': int}}
VIDEO_WRITERS_LOCK = threading.Lock()

# 统计信息（改为与实时算法一致的格式）
STATS = {
    'total_requests': 0,
    'total_inference_time': 0.0,
    'last_inference_time': 0.0,  # 最近一次推理时间（ms）
    'last_total_time': 0.0,      # 最近一次总耗时（ms）
}

# 绊线告警相关（增量告警机制）
LAST_CROSSING_COUNTS = {}
LAST_CROSSING_COUNTS_LOCK = threading.Lock()


def point_in_polygon(point, polygon):
    """
    判断点是否在多边形内（射线法）
    point: (x, y)
    polygon: [(x1, y1), (x2, y2), ...]
    """
    x, y = point
    n = len(polygon)
    inside = False
    
    p1x, p1y = polygon[0]
    for i in range(1, n + 1):
        p2x, p2y = polygon[i % n]
        if y > min(p1y, p2y):
            if y <= max(p1y, p2y):
                if x <= max(p1x, p2x):
                    if p1y != p2y:
                        xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                    if p1x == p2x or x <= xinters:
                        inside = not inside
        p1x, p1y = p2x, p2y
    
    return inside


def filter_objects_by_region(objects, regions_or_config, image_size):
    """
    根据区域过滤检测对象（支持矩形、多边形，支持归一化/画布坐标）
    objects:    检测到的对象列表
    regions_or_config: 区域配置列表或完整算法配置
    image_size: (width, height)
    返回:       过滤后的对象列表
    """
    if not regions_or_config:
        return objects
    
    algo_config = regions_or_config if isinstance(regions_or_config, dict) else None
    regions = algo_config.get('regions', []) if algo_config else regions_or_config
    
    if not regions:
        return objects
    
    # 只考虑启用的非绊线区域
    enabled_regions = [
        r for r in regions
        if r.get('enabled', True) and r.get('type') not in ['line']
    ]
    
    if not enabled_regions:
        # 没有启用的检测区域，返回所有对象
        return objects
    
    width, height = image_size
    filtered_objects = []
    
    default_coordinate_type = ''
    canvas_size = {}
    if algo_config:
        default_coordinate_type = (algo_config.get('coordinate_type') or '').lower()
        canvas_size = algo_config.get('canvas_size') or {}
    
    canvas_width = canvas_size.get('width') or width
    canvas_height = canvas_size.get('height') or height

    def convert_point(point, coordinate_type_override=None):
        if point is None or len(point) < 2:
            return None
        
        x, y = point[0], point[1]
        coord_type = (coordinate_type_override or '').lower()
        if not coord_type:
            coord_type = default_coordinate_type
        
        if coord_type in ('normalized', 'relative'):
            return x * width, y * height
        
        if coord_type in ('canvas', 'design', 'ui'):
            if canvas_width and canvas_height:
                scale_x = width / canvas_width
                scale_y = height / canvas_height
                return x * scale_x, y * scale_y
            return x, y
        
        if coord_type in ('pixel', 'pixels', 'absolute'):
            return x, y
        
        if isinstance(x, (int, float)) and isinstance(y, (int, float)):
            if 0.0 <= x <= 1.0 and 0.0 <= y <= 1.0:
                return x * width, y * height
        return x, y

    for obj in objects:
        bbox = obj['bbox']
        # 计算物体中心点（原始坐标）
        center_x_raw = (bbox[0] + bbox[2]) / 2
        center_y_raw = (bbox[1] + bbox[3]) / 2
        
        # 判断bbox是否为归一化坐标，并转换为像素坐标
        if all(0 <= coord <= 1 for coord in bbox):
            # 归一化坐标，转换为像素坐标
            center_x = center_x_raw * width
            center_y = center_y_raw * height
        else:
            # 已经是像素坐标
            center_x = center_x_raw
            center_y = center_y_raw
        
        # 检查是否在任何一个区域内
        in_any_region = False
        for region in enabled_regions:
            region_type = region.get('type')
            points = region.get('points', [])
            region_coord_type = (region.get('coordinate_type') or '').lower()
            region_threshold = None
            properties = region.get('properties') or {}
            if isinstance(properties, dict):
                region_threshold = properties.get('threshold')
            if region_threshold is not None:
                try:
                    if float(obj.get('confidence', 0.0)) < float(region_threshold):
                        continue
                except Exception:
                    continue
            
            if region_type == 'rectangle' and len(points) >= 2:
                # 矩形区域：points[0] 是左上角，points[1] 是右下角
                p1, p2 = points[0], points[1]
                
                converted_p1 = convert_point(p1, region_coord_type)
                converted_p2 = convert_point(p2, region_coord_type)
                if not converted_p1 or not converted_p2:
                    continue
                x1, y1 = converted_p1
                x2, y2 = converted_p2
                
                # 确保 x1 < x2, y1 < y2
                x1, x2 = min(x1, x2), max(x1, x2)
                y1, y2 = min(y1, y2), max(y1, y2)
                
                # 判断中心点是否在矩形内
                if x1 <= center_x <= x2 and y1 <= center_y <= y2:
                    in_any_region = True
                    break
                    
            elif region_type == 'polygon' and len(points) >= 3:
                # 多边形区域
                polygon = []
                for point in points:
                    converted = convert_point(point, region_coord_type)
                    if converted is not None:
                        polygon.append(tuple(converted))
                
                if len(polygon) < 3:
                    continue
                
                # 判断中心点是否在多边形内
                if point_in_polygon((center_x, center_y), polygon):
                    in_any_region = True
                    break
        
        if in_any_region:
            filtered_objects.append(obj)
    
    return filtered_objects


class InferenceRequest:
    """推理请求对象"""
    
    def __init__(self, request_id, image, request_data):
        self.request_id = request_id
        self.image = image
        self.request_data = request_data
        self.result = None
        self.error = None
        self.event = threading.Event()
        self.submit_time = time.time()


def _linear_sum_assignment(cost_matrix):
    """
    使用SciPy的匈牙利算法；若不可用则采用贪心匹配回退，保证在线服务不依赖额外包。
    """
    if _hungarian_assignment is not None:
        return _hungarian_assignment(cost_matrix)
    
    rows, cols = cost_matrix.shape
    flat_indices = np.argsort(cost_matrix, axis=None)
    used_rows = set()
    used_cols = set()
    row_ind = []
    col_ind = []
    
    for flat_idx in flat_indices:
        r = flat_idx // cols
        c = flat_idx % cols
        if r in used_rows or c in used_cols:
            continue
        row_ind.append(r)
        col_ind.append(c)
        used_rows.add(r)
        used_cols.add(c)
    
    return np.array(row_ind, dtype=int), np.array(col_ind, dtype=int)


class ImprovedKalmanFilter2D:
    """更平滑的常速度Kalman滤波器（优化用于人员密集场景）"""
    
    def __init__(self, dt=1.0):
        self.dt = dt
        self.F = np.array([
            [1, 0, dt, 0],
            [0, 1, 0, dt],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ], dtype=float)
        self.H = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0]
        ], dtype=float)
        # 大幅降低过程噪声，显著提高轨迹平滑度（减少抖动）
        self.Q = np.diag([0.08, 0.08, 0.8, 0.8])  # 进一步降低，从[0.15, 0.15, 1.5, 1.5]
        # 大幅降低观测噪声，提高平滑度
        self.R = np.eye(2, dtype=float) * 1.2  # 从1.8进一步降低
        self.P = np.eye(4, dtype=float) * 5.0  # 从10.0进一步降低
        self.state = np.zeros((4, 1), dtype=float)
    
    def initialize(self, x, y):
        self.state = np.array([[x], [y], [0.0], [0.0]], dtype=float)
    
    def predict(self):
        self.state = self.F @ self.state
        self.P = self.F @ self.P @ self.F.T + self.Q
        return self.state[:2].flatten()
    
    def update(self, x, y):
        z = np.array([[x], [y]], dtype=float)
        y_residual = z - (self.H @ self.state)
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.state = self.state + K @ y_residual
        self.P = (np.eye(4) - K @ self.H) @ self.P
        return self.state[:2].flatten()


class ImprovedObjectTracker:
    """参考测试脚本的目标跟踪实体（带Kalman滤波与轨迹历史，优化用于人员密集场景）"""
    
    def __init__(self, track_id, bbox, confidence, class_name):
        self.track_id = track_id
        self.bbox = bbox
        self.confidence = confidence
        self.class_name = class_name
        self.center_history = []
        self.last_update = time.time()
        self.missed = 0
        self.crossed_lines = {}  # 改为字典：{line_key: (last_cross_segment, last_cross_direction)}
        self.is_crossed = False
        self.hits = 1
        # 添加速度历史，用于改进匹配
        self.velocity_history = []
        # 添加指数移动平均（EMA）用于轨迹平滑
        self.ema_cx = None
        self.ema_cy = None
        self.ema_alpha = 0.55  # EMA平滑系数（越小越平滑，但响应越慢）
        
        cx, cy = self.get_center(bbox)
        self.width = bbox[2] - bbox[0]
        self.height = bbox[3] - bbox[1]
        self.kf = ImprovedKalmanFilter2D()
        self.kf.initialize(cx, cy)
        # 初始化EMA
        self.ema_cx = cx
        self.ema_cy = cy
        self.center_history.append((cx, cy))
    
    @staticmethod
    def get_center(bbox):
        x1, y1, x2, y2 = bbox
        return (x1 + x2) / 2.0, (y1 + y2) / 2.0
    
    def predict(self):
        cx, cy = self.kf.predict()
        
        # 使用EMA进行平滑
        if self.ema_cx is not None and self.ema_cy is not None:
            self.ema_cx = self.ema_alpha * cx + (1 - self.ema_alpha) * self.ema_cx
            self.ema_cy = self.ema_alpha * cy + (1 - self.ema_alpha) * self.ema_cy
            cx, cy = self.ema_cx, self.ema_cy
        else:
            self.ema_cx = cx
            self.ema_cy = cy
        
        # 如果历史轨迹足够长，使用短期移动平均适度平滑
        if len(self.center_history) >= 3:
            # 对最近3个点进行移动平均
            last_points = self.center_history[-3:]
            avg_cx = sum(p[0] for p in last_points) / len(last_points)
            avg_cy = sum(p[1] for p in last_points) / len(last_points)
            # 使用加权平均：EMA值权重0.8，历史平均值权重0.2，减小滞后
            cx = 0.8 * cx + 0.2 * avg_cx
            cy = 0.8 * cy + 0.2 * avg_cy
        
        self._update_bbox_from_center(cx, cy)
        self.center_history.append((cx, cy))
        if len(self.center_history) > 100:  # 增加历史长度，用于更好的轨迹预测
            self.center_history.pop(0)
        self.missed += 1
        return self.bbox
    
    def update(self, bbox, confidence):
        cx, cy = self.get_center(bbox)
        # 计算速度（用于改进匹配）
        if len(self.center_history) > 0:
            last_cx, last_cy = self.center_history[-1]
            vx = cx - last_cx
            vy = cy - last_cy
            self.velocity_history.append((vx, vy))
            if len(self.velocity_history) > 10:
                self.velocity_history.pop(0)
        
        # 在更新卡尔曼滤波器之前，先对检测值进行平滑
        if len(self.center_history) >= 1:
            last_cx, last_cy = self.center_history[-1]
            # 对检测值进行轻度平滑：85%检测值 + 15%历史值，减少延迟
            cx = 0.85 * cx + 0.15 * last_cx
            cy = 0.85 * cy + 0.15 * last_cy
        
        self.kf.update(cx, cy)
        cx_smoothed, cy_smoothed = self.kf.state[0, 0], self.kf.state[1, 0]
        
        # 使用EMA进一步平滑卡尔曼滤波器的输出
        if self.ema_cx is not None and self.ema_cy is not None:
            self.ema_cx = self.ema_alpha * cx_smoothed + (1 - self.ema_alpha) * self.ema_cx
            self.ema_cy = self.ema_alpha * cy_smoothed + (1 - self.ema_alpha) * self.ema_cy
            cx_smoothed, cy_smoothed = self.ema_cx, self.ema_cy
        else:
            self.ema_cx = cx_smoothed
            self.ema_cy = cy_smoothed
        
        # 如果历史轨迹足够长，使用短期移动平均进一步平滑
        if len(self.center_history) >= 3:
            last_points = self.center_history[-3:]
            avg_cx = sum(p[0] for p in last_points) / len(last_points)
            avg_cy = sum(p[1] for p in last_points) / len(last_points)
            # 使用加权平均：EMA值权重0.85，历史平均值权重0.15，提升响应速度
            cx_smoothed = 0.85 * cx_smoothed + 0.15 * avg_cx
            cy_smoothed = 0.85 * cy_smoothed + 0.15 * avg_cy
        
        # 增加bbox尺寸更新的平滑系数，减少尺寸抖动
        self.width = 0.9 * self.width + 0.1 * (bbox[2] - bbox[0])  # 从0.85/0.15改为0.9/0.1
        self.height = 0.9 * self.height + 0.1 * (bbox[3] - bbox[1])
        
        self._update_bbox_from_center(cx_smoothed, cy_smoothed)
        self.confidence = confidence
        self.last_update = time.time()
        self.missed = 0
        self.hits += 1
    
    def get_velocity(self):
        """获取平均速度"""
        if len(self.velocity_history) == 0:
            return (0.0, 0.0)
        vx_avg = sum(v[0] for v in self.velocity_history) / len(self.velocity_history)
        vy_avg = sum(v[1] for v in self.velocity_history) / len(self.velocity_history)
        return (vx_avg, vy_avg)
    
    def _update_bbox_from_center(self, cx, cy):
        half_w = max(self.width / 2.0, 1.0)
        half_h = max(self.height / 2.0, 1.0)
        self.bbox = [
            cx - half_w,
            cy - half_h,
            cx + half_w,
            cy + half_h
        ]
    
    def get_trajectory(self):
        """获取最近两点轨迹（保持向后兼容）"""
        if len(self.center_history) >= 2:
            return self.center_history[-2], self.center_history[-1]
        return None
    
    def get_recent_trajectory(self, num_points=10):
        """获取最近的轨迹点（用于更鲁棒的穿越检测）"""
        if len(self.center_history) < 2:
            return []
        # 返回最近的num_points个点，但至少返回2个点
        return self.center_history[-min(num_points, len(self.center_history)):]
    
    def get_smoothed_trajectory(self, num_points=10, window_size=5):
        """获取平滑后的轨迹点（使用更大的移动平均窗口减少抖动）"""
        if len(self.center_history) < 2:
            return []
        
        recent_points = self.center_history[-min(num_points, len(self.center_history)):]
        if len(recent_points) < window_size:
            return recent_points
        
        # 对轨迹进行移动平均平滑（使用更大的窗口）
        smoothed = []
        for i in range(len(recent_points)):
            # 计算窗口内的平均值（使用更大的窗口）
            start_idx = max(0, i - window_size // 2)
            end_idx = min(len(recent_points), i + window_size // 2 + 1)
            window_points = recent_points[start_idx:end_idx]
            
            # 使用加权平均，中心点权重更大
            weights = []
            center_idx = i
            for j in range(start_idx, end_idx):
                # 距离中心越近，权重越大
                dist = abs(j - center_idx)
                weight = max(0, 1.0 - dist / (window_size / 2.0))
                weights.append(weight)
            
            total_weight = sum(weights)
            if total_weight > 0:
                avg_x = sum(p[0] * w for p, w in zip(window_points, weights)) / total_weight
                avg_y = sum(p[1] * w for p, w in zip(window_points, weights)) / total_weight
            else:
                avg_x = recent_points[i][0]
                avg_y = recent_points[i][1]
            
            smoothed.append((avg_x, avg_y))
        
        return smoothed


class TrackerManager:
    """
    稳定性更强的匈牙利+卡尔曼跟踪器，跨线累计逻辑对齐 test_line_crossing_video.py
    优化用于人员密集场景：降低IOU阈值、增加距离容忍度、改进匹配算法
    """
    
    def __init__(self, iou_threshold=0.05, max_missed=10, center_distance=200, frame_step=1):
        # 进一步降低IOU阈值，在人员密集时更容忍遮挡（从0.10降到0.05）
        self.iou_threshold = iou_threshold
        # 控制最大丢失帧数，默认10帧，快速清理丢失轨迹
        self.base_max_missed = max_missed
        # 适度降低中心距离阈值，避免误匹配（从250降低到200）
        self.base_center_distance = center_distance
        self.frame_step = frame_step
        self.task_trackers = defaultdict(dict)              # task_id -> {track_id: tracker}
        self.task_next_ids = defaultdict(lambda: 1)         # task_id -> 下一个track id
        self.line_crossing_counts = defaultdict(lambda: defaultdict(int))
        self.last_reset_time = time.time()
        self.reset_interval = 24 * 60 * 60
    
    def _predict_all(self, task_id, frame_step=1):
        trackers_dict = self.task_trackers[task_id]
        effective_max_missed = self.base_max_missed * max(1, frame_step)
        for tracker in trackers_dict.values():
            tracker.predict()
        to_delete = [tid for tid, tracker in trackers_dict.items() if tracker.missed > effective_max_missed]
        for tid in to_delete:
            del trackers_dict[tid]
        return trackers_dict
    
    def _ensure_task_state(self, task_id):
        _ = self.task_trackers[task_id]
        _ = self.task_next_ids[task_id]
        _ = self.line_crossing_counts[task_id]
    
    def update(self, task_id, detections, frame_step=1):
        """更新指定task的跟踪状态"""
        self._ensure_task_state(task_id)
        
        if not detections:
            return list(self._predict_all(task_id, frame_step).values())
        
        trackers_dict = self._predict_all(task_id, frame_step)
        trackers_list = list(trackers_dict.items())
        n_trk = len(trackers_list)
        n_det = len(detections)
        
        effective_center_distance = self.base_center_distance * max(1, frame_step)
        
        if n_trk == 0:
            for det in detections:
                track_id = self.task_next_ids[task_id]
                trackers_dict[track_id] = ImprovedObjectTracker(track_id, det['bbox'], det['confidence'], det['class'])
                self.task_next_ids[task_id] += 1
            return list(trackers_dict.values())
        
        # 两阶段匹配策略：先高IOU匹配，再低IOU/距离匹配
        assigned_tracks = set()
        assigned_dets = set()
        
        # 第一阶段：高IOU匹配（IOU > 0.25，提高阈值确保高质量匹配，避免一个目标多条轨迹）
        # 使用贪心策略，但确保每个检测只匹配一次
        stage1_candidates = []  # [(iou, tid, det_idx, tracker, det, score), ...]
        for i, (tid, tracker) in enumerate(trackers_list):
            for j, det in enumerate(detections):
                iou = self._iou(det['bbox'], tracker.bbox)
                if iou > 0.25:  # 提高阈值到0.25，确保高质量匹配
                    # 计算综合得分：IOU + 跟踪器稳定性（hits越多越稳定）
                    stability_score = min(tracker.hits / 10.0, 1.0)  # 归一化到[0, 1]
                    combined_score = iou * 0.8 + stability_score * 0.2
                    stage1_candidates.append((combined_score, iou, tid, j, tracker, det))
        
        # 按综合得分降序排序，优先匹配高质量和稳定的
        stage1_candidates.sort(reverse=True, key=lambda x: x[0])
        
        # 执行第一阶段匹配（每个tracker和detection只匹配一次）
        for combined_score, iou, tid, det_idx, tracker, det in stage1_candidates:
            if tid in assigned_tracks or det_idx in assigned_dets:
                continue
            # 额外检查：如果跟踪器已经丢失很多帧，要求更高的IOU
            if tracker.missed > 5 and iou < 0.35:
                continue
            tracker.update(det['bbox'], det['confidence'])
            tracker.class_name = det.get('class', tracker.class_name)
            assigned_tracks.add(tid)
            assigned_dets.add(det_idx)
        
        # 第二阶段：中等IOU或距离匹配（使用匈牙利算法）
        remaining_trackers = [(i, tid, tracker) for i, (tid, tracker) in enumerate(trackers_list) if tid not in assigned_tracks]
        remaining_dets = [(j, det) for j, det in enumerate(detections) if j not in assigned_dets]
        
        if remaining_trackers and remaining_dets:
            n_rem_trk = len(remaining_trackers)
            n_rem_det = len(remaining_dets)
            cost_matrix = np.full((n_rem_trk, n_rem_det), 999.0, dtype=np.float32)
            
            for i, (orig_i, tid, tracker) in enumerate(remaining_trackers):
                for j, (orig_j, det) in enumerate(remaining_dets):
                    iou = self._iou(det['bbox'], tracker.bbox)
                    center_dist = self._center_distance(det['bbox'], tracker.bbox)
                    
                    # 计算轨迹预测位置（使用卡尔曼滤波器的预测）
                    pred_cx, pred_cy = tracker.kf.state[0, 0], tracker.kf.state[1, 0]
                    det_cx, det_cy = ImprovedObjectTracker.get_center(det['bbox'])
                    pred_dist = np.hypot(det_cx - pred_cx, det_cy - pred_cy)
                    
                    # 速度一致性（如果可用）
                    velocity_bonus = 0.0
                    if len(tracker.velocity_history) > 0:
                        vx, vy = tracker.get_velocity()
                        if abs(vx) > 0.1 or abs(vy) > 0.1:
                            # 基于速度的预测位置
                            pred_cx_v = pred_cx + vx
                            pred_cy_v = pred_cy + vy
                            vel_dist = np.hypot(det_cx - pred_cx_v, det_cy - pred_cy_v)
                            # 如果速度预测更准确，给予奖励
                            if vel_dist < pred_dist:
                                velocity_bonus = -0.2
                    
                    # 成本计算：综合考虑IOU、距离、预测距离、跟踪器稳定性
                    # 跟踪器稳定性惩罚：丢失帧数越多，成本越高
                    missed_penalty = min(tracker.missed / 10.0, 0.5)  # 最多增加0.5的成本
                    # 跟踪器稳定性奖励：hits越多，成本越低
                    stability_bonus = -min(tracker.hits / 20.0, 0.2)  # 最多减少0.2的成本
                    
                    if iou > 0.15:
                        # 中等IOU：主要考虑IOU和距离
                        cost = (1 - iou) * 0.6 + min(center_dist / effective_center_distance, 1.0) * 0.4 + velocity_bonus + missed_penalty + stability_bonus
                    elif iou > 0.08:
                        # 低IOU：主要考虑距离和预测距离
                        cost = (1 - iou) * 0.3 + min(center_dist / effective_center_distance, 1.0) * 0.5 + min(pred_dist / effective_center_distance, 1.0) * 0.2 + velocity_bonus + missed_penalty + stability_bonus
                    elif iou > 0.05:
                        # 极低IOU：主要考虑距离，但增加成本
                        cost = 0.5 + min(center_dist / effective_center_distance, 1.0) * 0.4 + min(pred_dist / effective_center_distance, 1.0) * 0.1 + velocity_bonus + missed_penalty + stability_bonus
                    elif center_dist < effective_center_distance * 0.7:  # 降低距离限制，避免误匹配
                        # 极低IOU但距离合理：纯距离匹配，但成本较高
                        cost = 0.8 + min(center_dist / effective_center_distance, 1.0) * 0.2 + min(pred_dist / effective_center_distance, 1.0) * 0.1 + missed_penalty + stability_bonus
                    else:
                        # 距离较远，成本很高
                        cost = 2.0 + min(center_dist / effective_center_distance, 2.0) + missed_penalty
                    
                    cost_matrix[i, j] = cost
            
            # 使用匈牙利算法进行匹配
            row_ind, col_ind = _linear_sum_assignment(cost_matrix)
            
            # 执行第二阶段匹配（使用更严格的成本阈值，避免误匹配）
            for i, j in zip(row_ind, col_ind):
                if i >= n_rem_trk or j >= n_rem_det:
                    continue
                # 根据IOU动态调整成本阈值：IOU越高，允许的成本越高
                orig_i, tid, tracker = remaining_trackers[i]
                orig_j, det = remaining_dets[j]
                iou = self._iou(det['bbox'], tracker.bbox)
                
                # 动态成本阈值：IOU越高，阈值越宽松
                if iou > 0.15:
                    max_cost = 3.0
                elif iou > 0.08:
                    max_cost = 2.5
                elif iou > 0.05:
                    max_cost = 2.0
                else:
                    max_cost = 1.5  # 极低IOU要求更严格
                
                if cost_matrix[i, j] > max_cost:
                    continue
                
                # 额外检查：如果跟踪器丢失很多帧，要求更高的IOU
                if tracker.missed > 10 and iou < 0.12:
                    continue
                
                if tid in assigned_tracks or orig_j in assigned_dets:
                    continue
                tracker.update(det['bbox'], det['confidence'])
                tracker.class_name = det.get('class', tracker.class_name)
                assigned_tracks.add(tid)
                assigned_dets.add(orig_j)
        
        # 第三阶段：对仍未匹配的tracker，尝试更宽松的距离匹配（但更严格，避免误匹配）
        still_unmatched_trackers = [(tid, tracker) for tid, tracker in trackers_dict.items() if tid not in assigned_tracks]
        still_unmatched_dets = [(j, det) for j, det in enumerate(detections) if j not in assigned_dets]
        
        if still_unmatched_trackers and still_unmatched_dets:
            # 构建距离矩阵，找到最佳匹配对
            stage3_candidates = []  # [(score, dist, iou, tid, det_idx, tracker, det), ...]
            max_dist = effective_center_distance * 1.0  # 降低距离限制，避免误匹配
            
            for tid, tracker in still_unmatched_trackers:
                pred_cx, pred_cy = tracker.kf.state[0, 0], tracker.kf.state[1, 0]
                
                for det_idx, det in still_unmatched_dets:
                    det_cx, det_cy = ImprovedObjectTracker.get_center(det['bbox'])
                    dist = np.hypot(det_cx - pred_cx, det_cy - pred_cy)
                    iou = self._iou(det['bbox'], tracker.bbox)
                    
                    # 如果距离在合理范围内，且IOU不是太低，加入候选
                    if dist < max_dist and iou > 0.03:  # 要求最小IOU
                        # 综合得分：距离越近、IOU越高、跟踪器越稳定，得分越高
                        stability_score = min(tracker.hits / 10.0, 1.0)
                        combined_score = (1.0 - dist / max_dist) * 0.5 + iou * 0.3 + stability_score * 0.2
                        stage3_candidates.append((combined_score, dist, iou, tid, det_idx, tracker, det))
            
            # 按综合得分降序排序，优先匹配高质量匹配
            stage3_candidates.sort(reverse=True, key=lambda x: x[0])
            
            # 执行第三阶段匹配（每个tracker和detection只匹配一次）
            for combined_score, dist, iou, tid, det_idx, tracker, det in stage3_candidates:
                if tid in assigned_tracks or det_idx in assigned_dets:
                    continue
                # 额外检查：如果跟踪器丢失很多帧，要求更高的IOU或更近的距离
                if tracker.missed > 15:
                    if iou < 0.05 and dist > effective_center_distance * 0.6:
                        continue
                tracker.update(det['bbox'], det['confidence'])
                tracker.class_name = det.get('class', tracker.class_name)
                assigned_tracks.add(tid)
                assigned_dets.add(det_idx)
        
        # 未匹配的检测创建新跟踪器（确保所有检测都被跟踪）
        for det_idx, det in enumerate(detections):
            if det_idx in assigned_dets:
                continue
            track_id = self.task_next_ids[task_id]
            trackers_dict[track_id] = ImprovedObjectTracker(track_id, det['bbox'], det['confidence'], det['class'])
            self.task_next_ids[task_id] += 1
        
        return list(trackers_dict.values())
    
    def _reset_line_counts_if_needed(self):
        current_time = time.time()
        if current_time - self.last_reset_time >= self.reset_interval:
            print("  🔄 跨线累计已超过24小时，自动清零")
            self.line_crossing_counts = defaultdict(lambda: defaultdict(int))
            self.last_reset_time = current_time
    
    def check_line_crossing(self, task_id, regions, image_size=None):
        """检查跨线并累加（与测试脚本保持一致逻辑）"""
        self._reset_line_counts_if_needed()
        trackers_dict = self.task_trackers.get(task_id, {})
        if not trackers_dict:
            return {}
        
        width, height = image_size if image_size else (1920, 1080)
        crossing_results = {}
        
        for region in regions:
            if not region.get('enabled', True):
                continue
            if region.get('type') != 'line':
                continue
            points = region.get('points', [])
            if len(points) < 2:
                continue
            
            region_id = region.get('id', 'line_unknown')
            region_name = region.get('name', region_id)
            direction = region.get('properties', {}).get('direction', 'both')
            p1_raw = points[0]
            p2_raw = points[1]
            coord_type = (region.get('coordinate_type') or '').lower()
            if not coord_type and any(0 <= coord <= 1 for point in points for coord in point):
                coord_type = 'normalized'
            
            if coord_type in ('normalized', 'relative') or all(0 <= coord <= 1 for coord in p1_raw + p2_raw):
                p1 = (int(p1_raw[0] * width), int(p1_raw[1] * height))
                p2 = (int(p2_raw[0] * width), int(p2_raw[1] * height))
            else:
                p1 = tuple(map(int, p1_raw))
                p2 = tuple(map(int, p2_raw))
            
            for tracker in trackers_dict.values():
                # 使用平滑后的轨迹历史进行穿越检测（最多检查最近20个点）
                # 使用平滑轨迹可以减少抖动对穿越检测的影响
                recent_trajectory = tracker.get_smoothed_trajectory(num_points=20, window_size=3)
                if len(recent_trajectory) < 2:
                    # 如果平滑轨迹不够，使用原始轨迹
                    recent_trajectory = tracker.get_recent_trajectory(num_points=20)
                if len(recent_trajectory) < 2:
                    continue
                
                # 检查轨迹的所有连续线段，找到穿越点
                crossed = False
                cross_direction = None
                crossing_segment = None
                
                # 从后往前检查（最近的轨迹优先）
                for i in range(len(recent_trajectory) - 1, 0, -1):
                    start_point = recent_trajectory[i-1]
                    end_point = recent_trajectory[i]
                    
                    # 检查线段是否与绊线相交
                    if self._segments_intersect(start_point, end_point, p1, p2):
                        crossed = True
                        crossing_segment = (start_point, end_point)
                        # 使用穿越线段判断方向
                        cross_direction = self._get_cross_direction(start_point, end_point, p1, p2)
                        break  # 找到穿越就停止
                
                # 如果没有找到直接相交，尝试使用更宽松的方法：检查轨迹是否跨越绊线
                if not crossed and len(recent_trajectory) >= 3:
                    # 检查轨迹起点和终点是否在绊线两侧
                    first_point = recent_trajectory[0]
                    last_point = recent_trajectory[-1]
                    
                    # 计算点到直线的有向距离
                    def point_to_line_side(point, line_p1, line_p2):
                        """判断点在直线的哪一侧，返回>0或<0"""
                        x, y = point
                        x1, y1 = line_p1
                        x2, y2 = line_p2
                        # 使用叉积判断
                        return (x2 - x1) * (y - y1) - (y2 - y1) * (x - x1)
                    
                    side_first = point_to_line_side(first_point, p1, p2)
                    side_last = point_to_line_side(last_point, p1, p2)
                    
                    # 如果起点和终点在绊线两侧，且距离绊线足够近，认为发生了穿越
                    if side_first * side_last < 0:  # 异号表示在两侧
                        # 计算到绊线的最短距离
                        def point_to_line_distance(point, line_p1, line_p2):
                            """计算点到线段的最短距离"""
                            x, y = point
                            x1, y1 = line_p1
                            x2, y2 = line_p2
                            
                            # 线段向量
                            dx = x2 - x1
                            dy = y2 - y1
                            line_len_sq = dx * dx + dy * dy
                            
                            if line_len_sq < 1e-6:
                                # 线段退化为点
                                return np.hypot(x - x1, y - y1)
                            
                            # 计算投影参数t
                            t = max(0, min(1, ((x - x1) * dx + (y - y1) * dy) / line_len_sq))
                            
                            # 投影点
                            proj_x = x1 + t * dx
                            proj_y = y1 + t * dy
                            
                            # 返回距离
                            return np.hypot(x - proj_x, y - proj_y)
                        
                        # 检查轨迹中是否有足够近的点
                        min_dist = float('inf')
                        for point in recent_trajectory:
                            dist = point_to_line_distance(point, p1, p2)
                            min_dist = min(min_dist, dist)
                        
                        # 如果最近距离小于阈值（绊线长度的10%或50像素，取较大值）
                        line_length = np.hypot(p2[0] - p1[0], p2[1] - p1[1])
                        threshold = max(line_length * 0.1, 50.0)
                        
                        if min_dist < threshold:
                            crossed = True
                            # 使用起点和终点判断方向
                            cross_direction = self._get_cross_direction(first_point, last_point, p1, p2)
                            if cross_direction == 'unknown':
                                # 如果无法判断，根据位置判断
                                if side_first > 0 and side_last < 0:
                                    cross_direction = 'in'
                                elif side_first < 0 and side_last > 0:
                                    cross_direction = 'out'
                
                if not crossed:
                    continue
                
                # 检查方向是否符合要求
                should_count = (
                    direction == 'both' or
                    (direction == 'in' and cross_direction == 'in') or
                    (direction == 'out' and cross_direction == 'out')
                )
                if not should_count:
                    continue
                
                cross_key = f"{task_id}_{region_id}_{tracker.track_id}"
                
                # 检查是否是新的穿越（避免同一帧内重复计数）
                # 策略：记录最后一次穿越的轨迹段和方向
                # - 如果方向改变（in<->out），允许计数（来回穿越）
                # - 如果方向相同但穿越段距离较远，允许计数（可能是新的穿越）
                # - 如果方向相同且穿越段很近，不重复计数（同一帧内重复检测）
                is_new_crossing = True
                if cross_key in tracker.crossed_lines:
                    last_segment, last_direction = tracker.crossed_lines[cross_key]
                    
                    # 如果方向改变，允许计数（来回穿越）
                    if cross_direction != last_direction:
                        is_new_crossing = True
                    else:
                        # 方向相同，检查穿越段是否不同
                        if crossing_segment:
                            seg_start, seg_end = crossing_segment
                            if last_segment and len(last_segment) == 2:
                                last_seg_start, last_seg_end = last_segment
                                # 计算穿越段的终点距离
                                dist = np.hypot(seg_end[0] - last_seg_end[0], seg_end[1] - last_seg_end[1])
                                # 如果距离很近（小于20像素），认为是同一穿越，不重复计数
                                if dist < 20.0:
                                    is_new_crossing = False
                        else:
                            # 如果没有crossing_segment，使用轨迹的最近点
                            if len(recent_trajectory) >= 2:
                                last_point = recent_trajectory[-1]
                                if last_segment and len(last_segment) == 2:
                                    last_seg_end = last_segment[1]
                                    dist = np.hypot(last_point[0] - last_seg_end[0], last_point[1] - last_seg_end[1])
                                    # 如果距离很近，认为是同一穿越
                                    if dist < 20.0:
                                        is_new_crossing = False
                
                if is_new_crossing:
                    self.line_crossing_counts[task_id][region_id] += 1
                    # 更新最后一次穿越记录
                    if crossing_segment:
                        tracker.crossed_lines[cross_key] = (crossing_segment, cross_direction)
                    else:
                        # 如果没有crossing_segment，使用轨迹的最近两点
                        if len(recent_trajectory) >= 2:
                            tracker.crossed_lines[cross_key] = ((recent_trajectory[-2], recent_trajectory[-1]), cross_direction)
                    tracker.is_crossed = True
                    print(f"    [绊线穿越] task={task_id} track={tracker.track_id} line={region_id} dir={cross_direction} -> 累计 {self.line_crossing_counts[task_id][region_id]}")
            
            crossing_results[region_id] = {
                'region_name': region_name,
                'count': self.line_crossing_counts[task_id][region_id],
                'direction': direction,
                'points': [p1, p2]
            }
        
        return crossing_results
    
    @staticmethod
    def _iou(bbox1, bbox2):
        x1 = max(bbox1[0], bbox2[0])
        y1 = max(bbox1[1], bbox2[1])
        x2 = min(bbox1[2], bbox2[2])
        y2 = min(bbox1[3], bbox2[3])
        if x2 < x1 or y2 < y1:
            return 0.0
        inter = (x2 - x1) * (y2 - y1)
        area1 = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
        area2 = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])
        union = area1 + area2 - inter
        return inter / union if union > 0 else 0.0
    
    @staticmethod
    def _center_distance(bbox1, bbox2):
        c1 = ImprovedObjectTracker.get_center(bbox1)
        c2 = ImprovedObjectTracker.get_center(bbox2)
        return np.hypot(c1[0] - c2[0], c1[1] - c2[1])
    
    @staticmethod
    def _segments_intersect(p1, p2, p3, p4):
        def ccw(A, B, C):
            return (C[1] - A[1]) * (B[0] - A[0]) > (B[1] - A[1]) * (C[0] - A[0])
        return ccw(p1, p3, p4) != ccw(p2, p3, p4) and ccw(p1, p2, p3) != ccw(p1, p2, p4)
    
    @staticmethod
    def _get_cross_direction(start, end, line_p1, line_p2):
        def cross_product(o, a, b):
            return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])
        cp_start = cross_product(line_p1, line_p2, start)
        cp_end = cross_product(line_p1, line_p2, end)
        if cp_start > 0 and cp_end < 0:
            return 'in'
        if cp_start < 0 and cp_end > 0:
            return 'out'
        return 'unknown'


class BatchInferenceProcessor:
    """批处理推理处理器（使用OM模型）"""
    
    def __init__(self, model_path, batch_size=8, batch_timeout=0.1):
        self.model_path = model_path  # OM模型路径
        self.batch_size = batch_size
        self.batch_timeout = batch_timeout
        self.request_queue = queue.Queue(maxsize=CONFIG['max_queue_size'])
        self.running = True
        self.stats_lock = threading.Lock()
        self.post_process_pool = ThreadPoolExecutor(max_workers=16, thread_name_prefix="PostProcess")
        
    def submit_request(self, image, request_data):
        """提交推理请求"""
        request_id = str(uuid.uuid4())
        request = InferenceRequest(request_id, image, request_data)
        
        try:
            self.request_queue.put(request, block=True, timeout=5.0)
            INFERENCE_EVENTS[request_id] = request.event
            
            with self.stats_lock:
                BATCH_STATS['total_requests'] += 1
            
            return request_id, request
        except queue.Full:
            raise Exception("推理队列已满，请稍后重试")
    
    def process_loop(self):
        """批处理循环"""
        print("批处理推理线程已启动")
        
        while self.running:
            try:
                batch_requests = []
                deadline = time.time() + self.batch_timeout
                
                try:
                    first_request = self.request_queue.get(timeout=1.0)
                    batch_requests.append(first_request)
                except queue.Empty:
                    continue
                
                while len(batch_requests) < self.batch_size and time.time() < deadline:
                    try:
                        remaining_time = deadline - time.time()
                        if remaining_time <= 0:
                            break
                        request = self.request_queue.get(timeout=remaining_time)
                        batch_requests.append(request)
                    except queue.Empty:
                        break
                
                if not batch_requests:
                    continue
                
                self._process_batch(batch_requests)
                
            except Exception as e:
                print(f"批处理循环错误: {str(e)}")
                import traceback
                traceback.print_exc()
    
    def _process_batch(self, batch_requests):
        """处理一批请求"""
        batch_size = len(batch_requests)
        
        try:
            print(f"\n{'='*60}")
            print(f"开始批处理推理 [{time.strftime('%H:%M:%S')}]")
            print(f"  批大小: {batch_size}")
            
            images = [req.image for req in batch_requests]
            
            inference_start = time.time()
            # 使用OM模型进行批量推理（逐个推理，因为om_infer不支持批量）
            # 注意：ACL不支持多线程并发调用，必须加锁保护
            results = []
            for image in images:
                with ACL_INFERENCE_LOCK:  # 保护ACL推理调用
                    boxes_out = om_infer(self.model_path, image, debug=False)
                results.append(boxes_out)
            inference_time = (time.time() - inference_start) * 1000
            
            print(f"  ✓ 批量推理完成: {inference_time:.0f}ms (平均 {inference_time/batch_size:.0f}ms/张)")
            
            post_process_start = time.time()
            
            futures = []
            for idx, (request, boxes_out) in enumerate(zip(batch_requests, results)):
                future = self.post_process_pool.submit(
                    self._process_single_result_wrapper,
                    request, boxes_out, inference_time / batch_size, idx, batch_size
                )
                futures.append((future, request))
            
            for future, request in futures:
                try:
                    future.result()
                except Exception as e:
                    request.error = str(e)
                    print(f"  ⚠️  后处理失败: {str(e)}")
            
            post_process_time = (time.time() - post_process_start) * 1000
            print(f"  ✓ 并行后处理完成: {post_process_time:.0f}ms")
            
            with self.stats_lock:
                BATCH_STATS['total_batches'] += 1
                BATCH_STATS['total_inference_time'] += inference_time
                BATCH_STATS['avg_batch_size'] = (
                    (BATCH_STATS['avg_batch_size'] * (BATCH_STATS['total_batches'] - 1) + batch_size) 
                    / BATCH_STATS['total_batches']
                )
                BATCH_STATS['max_batch_size'] = max(BATCH_STATS['max_batch_size'], batch_size)
            
            print(f"  批处理完成: {batch_size} 个请求")
            print(f"{'='*60}")
            
        except Exception as e:
            print(f"批处理失败: {str(e)}")
            import traceback
            traceback.print_exc()
            
            for request in batch_requests:
                request.error = f"批处理失败: {str(e)}"
                INFERENCE_RESULTS[request.request_id] = {
                    'result': None,
                    'error': request.error
                }
                request.event.set()
    
    def _process_single_result_wrapper(self, request, boxes_out, inference_time_per_image, idx, batch_size):
        """后处理包装器"""
        try:
            self._process_single_result(request, boxes_out, inference_time_per_image)
        except Exception as e:
            request.error = str(e)
            print(f"  请求 {idx+1}/{batch_size} 后处理失败: {str(e)}")
        finally:
            INFERENCE_RESULTS[request.request_id] = {
                'result': request.result,
                'error': request.error
            }
            request.event.set()
    
    def _process_single_result(self, request, boxes_out, inference_time_per_image):
        """处理单个推理结果（绊线专用版本，使用OM模型）"""
        global TRACKER_MANAGER, TRACKER_LOCK, LAST_CROSSING_COUNTS, LAST_CROSSING_COUNTS_LOCK, CLASS_NAMES
        
        request_data = request.request_data
        image = request.image
        task_id = request_data.get('task_id', 'unknown')
        algo_config = request_data.get('algo_config')
        if not algo_config:
            algo_config = load_algo_config(request_data.get('image_url', ''))
        
        # 获取算法参数
        confidence_threshold = 0.5
        if algo_config:
            algo_params = algo_config.get('algorithm_params', {})
            confidence_threshold = algo_params.get('confidence_threshold', 0.5)
        
        # 解析OM模型推理结果
        objects = []
        detections = []
        
        if boxes_out is not None and len(boxes_out) > 0:
            for b in boxes_out:
                x1, y1, x2, y2, conf, cls_id = b
                if float(conf) < confidence_threshold:
                    continue
                cls_id = int(cls_id)
                class_name = CLASS_NAMES[cls_id] if 0 <= cls_id < len(CLASS_NAMES) else str(cls_id)
                
                obj = {
                    'class': class_name,
                    'confidence': float(conf),
                    'bbox': [float(x1), float(y1), float(x2), float(y2)]
                }
                objects.append(obj)
                detections.append(obj)
        
        # 【区域过滤】如果配置了检测区域，只保留区域内的物体
        original_count = len(objects)
        regions = []
        if algo_config:
            regions = algo_config.get('regions', [])
            if regions:
                image_size = (image.shape[1], image.shape[0])
                objects = filter_objects_by_region(objects, algo_config, image_size)
                detections = filter_objects_by_region(detections, algo_config, image_size)
                filtered_count = original_count - len(objects)
                if filtered_count > 0:
                    print(f"  ℹ️  区域过滤: 原始 {original_count} 个 → 区域内 {len(objects)} 个 (过滤掉 {filtered_count} 个)")
        
        # 构建结果
        result_data = {
            'objects': objects,
            'total_count': len(objects),
        }
        
        # 跟踪和绊线检测
        line_crossing_results = None
        trackers = []
        
        if TRACKER_MANAGER:
            with TRACKER_LOCK:
                trackers = TRACKER_MANAGER.update(task_id, detections)
        
        if algo_config and trackers:
            regions = algo_config.get('regions', [])
            
            if regions:
                image_size = (image.shape[1], image.shape[0])
                with TRACKER_LOCK:
                    line_crossing_results = TRACKER_MANAGER.check_line_crossing(task_id, regions, image_size)
                
                if line_crossing_results:
                    # 【绊线增量告警】只有发生新穿越时才返回告警
                    total_crossed = sum(info['count'] for info in line_crossing_results.values())
                    
                    with LAST_CROSSING_COUNTS_LOCK:
                        last_count = LAST_CROSSING_COUNTS.get(task_id, 0)
                        
                        if total_crossed > last_count:
                            # 有新穿越 → 返回完整结果（触发告警）
                            new_crossings = total_crossed - last_count
                            result_data['person_count'] = new_crossings
                            result_data['line_crossing'] = line_crossing_results
                            LAST_CROSSING_COUNTS[task_id] = total_crossed
                            print(f"  ✅ 检测到新穿越: {last_count} → {total_crossed} (+{new_crossings})，上传告警")
                            print(f"     返回: total_count={result_data['total_count']}, person_count={new_crossings}, objects={len(result_data['objects'])}")
                        else:
                            # 无新穿越 → 返回空结果（不触发告警）
                            result_data['total_count'] = 0
                            result_data['objects'] = []
                            print(f"  ℹ️  无新穿越（累计={total_crossed}），返回空结果（不上传告警）")
                            print(f"     返回: total_count=0, objects=[], 无person_count")
                else:
                    # 无有效跨线检测结果 → 返回空结果
                    result_data['total_count'] = 0
                    result_data['objects'] = []
                    print(f"  ℹ️  绊线人数统计但无有效跨线结果，返回空结果")
        
        # 计算平均置信度（注意：无新穿越时 objects 会被清空）
        avg_confidence = 0.0
        if result_data.get('objects') and len(result_data['objects']) > 0:
            avg_confidence = sum(obj['confidence'] for obj in result_data['objects']) / len(result_data['objects'])
        
        # 计算总处理时间（从提交到现在）
        total_time = (time.time() - request.submit_time) * 1000
        
        # 更新最近一次时间统计
        global BATCH_STATS
        BATCH_STATS['last_inference_time'] = inference_time_per_image
        BATCH_STATS['last_total_time'] = total_time
        
        # 保存结果
        request.result = {
            'success': True,
            'result': result_data,
            'confidence': avg_confidence,
            'inference_time_ms': round(inference_time_per_image, 2),  # 模型推理时间
            'total_time_ms': round(total_time, 2),  # 全部处理时间（包含等待、推理、后处理）
            'image_url': request_data.get('image_url', ''),  # 请求的图片URL
            'task_id': task_id  # 任务ID
        }
    
    def stop(self):
        """停止处理器"""
        self.running = False


def draw_trajectory(image, tracker, color=None, is_crossed=False):
    """在图像上绘制跟踪轨迹，穿越后变色（参考test_line_crossing_video.py）"""
    if color is None:
        # 根据track_id生成颜色
        np.random.seed(tracker.track_id)
        color = tuple(map(int, np.random.randint(0, 255, 3)))
    
    # 如果已穿越绊线，使用红色
    if tracker.is_crossed or is_crossed:
        color = (0, 0, 255)  # 红色 (BGR格式)
    
    # 绘制当前边界框（穿越后加粗）
    x1, y1, x2, y2 = map(int, tracker.bbox)
    thickness = 3 if tracker.is_crossed else 2
    cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness)
    
    # 绘制轨迹（中心点连线）
    if len(tracker.center_history) >= 2:
        points = []
        for center in tracker.center_history:
            cx, cy = map(int, center)
            points.append((cx, cy))
        
        # 绘制轨迹线（穿越后加粗）
        line_thickness = 3 if tracker.is_crossed else 2
        for i in range(len(points) - 1):
            cv2.line(image, points[i], points[i + 1], color, line_thickness)
        
        # 绘制轨迹点
        point_radius = 4 if tracker.is_crossed else 3
        for point in points:
            cv2.circle(image, point, point_radius, color, -1)
    
    # 绘制track_id和置信度
    label = f"ID:{tracker.track_id} {tracker.class_name} {tracker.confidence:.2f}"
    if tracker.is_crossed:
        label += " [CROSSED]"
    label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    label_y = max(y1 - 5, label_size[1])
    cv2.rectangle(image, (x1, label_y - label_size[1] - 5), 
                  (x1 + label_size[0], label_y + 5), color, -1)
    cv2.putText(image, label, (x1, label_y), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    return image


def draw_line_config(image, line_regions, image_size=None):
    """绘制绊线配置（参考test_line_crossing_video.py）"""
    if not line_regions:
        return
    
    width, height = image_size if image_size else (image.shape[1], image.shape[0])
    
    for region in line_regions:
        if not region.get('enabled', True):
            continue
        
        if region.get('type') != 'line':
            continue
        
        points = region.get('points', [])
        if len(points) < 2:
            continue
        
        region_id = region.get('id', 'line_unknown')
        region_name = region.get('name', region_id)
        
        # 转换坐标
        p1_raw = points[0]
        p2_raw = points[1]
        
        coord_type = (region.get('coordinate_type') or '').lower()
        if not coord_type and any(0 <= coord <= 1 for point in points for coord in point):
            coord_type = 'normalized'
        
        if coord_type == 'normalized' or all(0 <= coord <= 1 for coord in p1_raw + p2_raw):
            p1 = (int(p1_raw[0] * width), int(p1_raw[1] * height))
            p2 = (int(p2_raw[0] * width), int(p2_raw[1] * height))
        else:
            p1 = tuple(map(int, p1_raw))
            p2 = tuple(map(int, p2_raw))
        
        # 绘制绊线（黄色粗线）
        cv2.line(image, p1, p2, (0, 255, 255), 3)
        
        # 在线段中点绘制名称和方向
        mid_x = (p1[0] + p2[0]) // 2
        mid_y = (p1[1] + p2[1]) // 2
        
        direction = region.get('properties', {}).get('direction', 'both')
        direction_text = {'in': '入', 'out': '出', 'both': '双向'}.get(direction, direction)
        
        label = f"{region_name} [{direction_text}]"
        label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        
        # 绘制文字背景
        cv2.rectangle(image, 
                     (mid_x - label_size[0] // 2 - 5, mid_y - label_size[1] - 5),
                     (mid_x + label_size[0] // 2 + 5, mid_y + 5),
                     (0, 255, 255), -1)
        
        # 绘制文字
        cv2.putText(image, label, 
                   (mid_x - label_size[0] // 2, mid_y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
        
        # 在线段端点绘制箭头指示方向
        if direction != 'both':
            # 计算箭头方向
            dx = p2[0] - p1[0]
            dy = p2[1] - p1[1]
            length = np.sqrt(dx*dx + dy*dy)
            if length > 0:
                dx /= length
                dy /= length
                
                # 在p1或p2处绘制箭头
                arrow_point = p2 if direction == 'out' else p1
                arrow_tip = (int(arrow_point[0] + dx * 15), int(arrow_point[1] + dy * 15))
                
                # 绘制箭头
                cv2.arrowedLine(image, arrow_point, arrow_tip, (0, 255, 255), 3, tipLength=0.3)


def draw_stats(image, line_crossing_results, inference_time=0, total_time=0, track_count=0):
    """在图像上绘制统计信息"""
    info_text = [
        f"Inference: {inference_time:.1f}ms",
        f"Total: {total_time:.1f}ms",
        f"Tracks: {track_count}"
    ]
    
    # 添加绊线统计信息（总是显示，即使没有结果）
    # 计算总穿越次数
    total_crossing_count = 0
    if line_crossing_results:
        total_crossing_count = sum(info['count'] for info in line_crossing_results.values())
    
    if total_crossing_count > 0:
        info_text.append("")
        info_text.append("Line Crossing:")
        info_text.append(f"  Total: {total_crossing_count}")  # 显示总穿越次数（青色高亮）
        
        # 显示每条线的穿越次数（仅显示 >0 的项目）
        if line_crossing_results:
            for line_id, line_info in line_crossing_results.items():
                if line_info['count'] > 0:
                    info_text.append(f"  {line_info['region_name']}: {line_info['count']}")
    
    y_offset = 20
    for text in info_text:
        if text == "":
            y_offset += 5
            continue
        # 总穿越次数使用更醒目的颜色（黄色，BGR格式）
        if "Total:" in text and text.startswith("  ") and "Line Crossing:" not in text:
            color = (0, 255, 255)  # 黄色 (BGR格式：B=0, G=255, R=255)
            font_scale = 0.7
            thickness = 3  # 加粗显示
        elif text.startswith("  ") and "Total:" not in text:
            color = (0, 255, 255)  # 黄色，显示单条线的统计
            font_scale = 0.5
            thickness = 2
        elif "Line Crossing:" in text:
            color = (0, 255, 255)  # 黄色标题
            font_scale = 0.6
            thickness = 2
        else:
            color = (0, 255, 0)  # 绿色，基本信息
            font_scale = 0.6
            thickness = 2
        cv2.putText(image, text, (10, y_offset), 
                   cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness)
        y_offset += 25


def get_or_create_video_writer(task_id, image_shape, force_new=False):
    """获取或创建视频写入器（按task_id管理，支持分段保存）"""
    global VIDEO_WRITERS, VIDEO_WRITERS_LOCK
    
    # 从配置中获取分段参数
    segment_duration = CONFIG.get('video_segment_duration', 60)  # 默认1分钟
    segment_max_size_mb = CONFIG.get('video_segment_max_size_mb', 500)  # 默认500MB
    
    if not CONFIG.get('enable_video_save', False):
        return None
    
    with VIDEO_WRITERS_LOCK:
        current_time = time.time()
        
        # 检查是否需要创建新段
        if task_id in VIDEO_WRITERS and not force_new:
            video_info = VIDEO_WRITERS[task_id]
            writer = video_info['writer']
            video_path = video_info['path']
            start_time = video_info['start_time']
            frame_count = video_info.get('frame_count', 0)
            
            # 检查时长是否超过限制
            duration = current_time - start_time
            if duration >= segment_duration:
                # 关闭当前写入器
                writer.release()
                print(f"  📹 视频片段时长达到限制，关闭: {video_path} (时长: {duration:.1f}秒)")
                force_new = True
            else:
                # 检查文件大小是否超过限制
                try:
                    if video_path.exists():
                        size_mb = video_path.stat().st_size / (1024 * 1024)
                        if size_mb >= segment_max_size_mb:
                            writer.release()
                            print(f"  📹 视频片段大小达到限制，关闭: {video_path} (大小: {size_mb:.1f}MB)")
                            force_new = True
                except Exception as e:
                    print(f"  ⚠️  检查视频文件大小失败: {e}")
            
            if not force_new:
                # 更新帧计数
                video_info['frame_count'] = frame_count + 1
                return writer
        
        # 创建新的视频写入器
        video_dir = Path(CONFIG.get('video_save_dir', './videos'))
        video_dir.mkdir(parents=True, exist_ok=True)
        
        # 生成视频文件名（包含task_id和时间戳，支持分段）
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        segment_index = 0
        if task_id in VIDEO_WRITERS:
            # 如果已有视频，查找下一个段索引
            existing_files = list(video_dir.glob(f"line_crossing_{task_id}_*.mp4"))
            if existing_files:
                # 从文件名中提取段索引
                indices = []
                for f in existing_files:
                    parts = f.stem.split('_')
                    if len(parts) >= 4:
                        try:
                            idx = int(parts[-1])
                            indices.append(idx)
                        except:
                            pass
                segment_index = max(indices) + 1 if indices else 0
        
        video_filename = f"line_crossing_{task_id}_{timestamp}_seg{segment_index:03d}.mp4"
        video_path = video_dir / video_filename
        
        height, width = image_shape[:2]
        fps = CONFIG.get('video_fps', 25)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer = cv2.VideoWriter(str(video_path), fourcc, fps, (width, height))
        
        if video_writer.isOpened():
            VIDEO_WRITERS[task_id] = {
                'writer': video_writer,
                'path': video_path,
                'start_time': current_time,
                'frame_count': 0
            }
            print(f"  📹 创建视频写入器: {video_path} (task_id={task_id}, 段索引: {segment_index})")
            return video_writer
        else:
            print(f"  ⚠️  无法创建视频写入器: {video_path}")
            return None


def load_algo_config(image_url):
    """
    加载算法配置文件
    从图片URL推断配置文件路径（同一路径下的algo_config.json）
    同时将远程配置文件保存到本地
    """
    try:
        parsed = urlparse(image_url)
        
        path_parts = parsed.path.rsplit('/', 1)
        if len(path_parts) == 2:
            config_url = f"{parsed.scheme}://{parsed.netloc}{path_parts[0]}/algo_config.json"
            
            print(f"  🔍 尝试加载配置文件: {config_url}")
            
            response = requests.get(config_url, timeout=5)
            if response.status_code == 200:
                config = response.json()
                print(f"  ✓ 成功加载配置文件")
                print(f"  📋 配置内容: task_id={config.get('task_id')}, regions={len(config.get('regions', []))}")
                
                # 保存配置文件到本地
                try:
                    config_dir = Path("/cv_space/predict/configs")
                    config_dir.mkdir(parents=True, exist_ok=True)
                    
                    task_id = config.get('task_id', 'unknown')
                    general_config_path = config_dir / f"{task_id}_algo_config.json"
                    
                    should_save = True
                    if general_config_path.exists():
                        try:
                            with open(general_config_path, 'r', encoding='utf-8') as f:
                                existing_config = json.load(f)
                                if existing_config == config:
                                    should_save = False
                                    print(f"  ℹ️  配置文件未改变，跳过保存")
                        except:
                            pass
                    
                    if should_save:
                        with open(general_config_path, 'w', encoding='utf-8') as f:
                            json.dump(config, f, ensure_ascii=False, indent=2)
                        print(f"  💾 配置文件已保存: {general_config_path}")
                    
                except Exception as save_error:
                    print(f"  ⚠️  保存配置文件失败: {str(save_error)}")
                
                return config
            else:
                print(f"  ℹ️  配置文件不存在 (状态码: {response.status_code})")
        
    except Exception as e:
        print(f"  ℹ️  加载配置文件失败: {str(e)}")
    
    return None


class YOLOInferenceHandler(BaseHTTPRequestHandler):
    """HTTP推理请求处理器"""
    
    def log_message(self, format, *args):
        """自定义日志格式"""
        print(f"[{self.log_date_time_string()}] {format % args}")
    
    def do_POST(self):
        """处理POST请求"""
        if self.path == '/infer':
            self.handle_inference()
        elif self.path == '/health':
            self.handle_health()
        elif self.path == '/reset_stats':
            self.handle_reset_stats()
        else:
            self.send_error(404, "Not Found")
    
    def do_GET(self):
        """处理GET请求"""
        # 调试信息：打印请求路径
        print(f"[DEBUG] GET请求路径: {self.path}")
        
        if self.path == '/health':
            self.handle_health()
        elif self.path == '/':
            self.handle_index()
        elif self.path == '/stats':
            self.handle_stats()
        elif self.path == '/videos' or self.path == '/videos/':
            self.handle_video_list()
        elif self.path.startswith('/videos/'):
            self.handle_video_download()
        elif self.path == '/api/writing-videos':
            self.handle_writing_videos()
        else:
            self.send_error(404, "Not Found")
    
    def handle_index(self):
        """首页"""
        self.send_response(200)
        self.send_header('Content-type', 'text/html; charset=utf-8')
        self.end_headers()
        
        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>{CONFIG['name']}</title>
            <meta charset="utf-8">
            <style>
                body {{
                    font-family: Arial, sans-serif;
                    max-width: 900px;
                    margin: 50px auto;
                    padding: 20px;
                    background: #f5f5f5;
                }}
                .container {{
                    background: white;
                    padding: 30px;
                    border-radius: 10px;
                    box-shadow: 0 2px 10px rgba(0,0,0,0.1);
                }}
                h1 {{
                    color: #333;
                    border-bottom: 3px solid #2196F3;
                    padding-bottom: 10px;
                }}
                .info-grid {{
                    display: grid;
                    grid-template-columns: repeat(2, 1fr);
                    gap: 15px;
                    margin: 20px 0;
                }}
                .info-item {{
                    padding: 15px;
                    background: #f9f9f9;
                    border-radius: 5px;
                    border-left: 4px solid #2196F3;
                }}
                .info-item strong {{
                    color: #666;
                    display: block;
                    margin-bottom: 5px;
                    font-size: 14px;
                }}
                .info-item span {{
                    color: #333;
                    font-size: 18px;
                    font-weight: bold;
                }}
                .stats-section {{
                    margin: 30px 0;
                    padding: 20px;
                    background: #e3f2fd;
                    border-radius: 5px;
                }}
                .stats-section h2 {{
                    margin-top: 0;
                    color: #1565c0;
                }}
                .stat-value {{
                    font-size: 32px;
                    font-weight: bold;
                    color: #0d47a1;
                    margin: 10px 0;
                }}
                .btn {{
                    background: #f44336;
                    color: white;
                    border: none;
                    padding: 12px 30px;
                    font-size: 16px;
                    border-radius: 5px;
                    cursor: pointer;
                    transition: background 0.3s;
                }}
                .btn:hover {{
                    background: #d32f2f;
                }}
                .btn:active {{
                    transform: scale(0.98);
                }}
                .message {{
                    padding: 15px;
                    margin: 15px 0;
                    border-radius: 5px;
                    display: none;
                }}
                .message.success {{
                    background: #4CAF50;
                    color: white;
                }}
                .endpoints {{
                    margin: 20px 0;
                    padding: 15px;
                    background: #fff3cd;
                    border-radius: 5px;
                    border-left: 4px solid #ffc107;
                }}
                .endpoints code {{
                    background: #fff;
                    padding: 2px 6px;
                    border-radius: 3px;
                    font-family: monospace;
                }}
            </style>
        </head>
        <body>
            <div class="container">
                <h1>🚶 {CONFIG['name']}</h1>
                
                <div class="info-grid">
                    <div class="info-item">
                        <strong>服务ID</strong>
                        <span>{CONFIG['service_id']}</span>
                    </div>
                    <div class="info-item">
                        <strong>版本</strong>
                        <span>{CONFIG['version']}</span>
                    </div>
                    <div class="info-item">
                        <strong>支持任务类型</strong>
                        <span>{', '.join(CONFIG['task_types'])}</span>
                    </div>
                    <div class="info-item">
                        <strong>推理模式</strong>
                        <span>单线程直接推理</span>
                    </div>
                </div>

                <div class="stats-section">
                    <h2>📊 实时统计</h2>
                    <div class="info-item">
                        <strong>累积推理次数</strong>
                        <div class="stat-value" id="total-requests">加载中...</div>
                    </div>
                    <div class="info-item" style="margin-top: 15px;">
                        <strong>平均推理时间</strong>
                        <div class="stat-value" id="avg-time">加载中...</div>
                    </div>
                    <button class="btn" onclick="resetStats()">🔄 清零统计数据</button>
                    <div id="message" class="message"></div>
                </div>

                <div class="endpoints">
                    <h3>🔌 API 端点</h3>
                    <p><strong>推理:</strong> <code>POST /infer</code></p>
                    <p><strong>健康检查:</strong> <code>GET /health</code></p>
                    <p><strong>统计信息:</strong> <code>GET /stats</code></p>
                    <p><strong>清零统计:</strong> <code>POST /reset_stats</code></p>
                    <p><strong>视频列表:</strong> <code>GET /videos</code></p>
                    <p><strong>视频下载:</strong> <code>GET /videos/文件名.mp4</code></p>
                </div>
                
                <div class="stats-section" id="video-section" style="background: #e8f5e9; border: 2px solid #4CAF50;">
                    <h2>📹 视频下载</h2>
                    <div id="video-list-container">
                        <p style="color: #666;">正在加载视频列表...</p>
                    </div>
                    <p style="margin-top: 15px;">
                        <a href="/videos" style="color: #2196F3; text-decoration: none; font-weight: bold; font-size: 16px;">
                            📋 查看完整视频列表 →
                        </a>
                    </p>
                </div>
            </div>

            <script>
                // 加载统计数据
                function loadStats() {{
                    fetch('/stats')
                        .then(res => res.json())
                        .then(data => {{
                            const stats = data.statistics || {{}};
                            const totalRequests = stats.total_requests || 0;
                            const avgTime = data.avg_inference_time_per_request || 0;
                            
                            document.getElementById('total-requests').textContent = totalRequests.toLocaleString();
                            document.getElementById('avg-time').textContent = avgTime.toFixed(2) + ' ms';
                        }})
                        .catch(err => {{
                            console.error('加载统计失败:', err);
                            document.getElementById('total-requests').textContent = '加载失败';
                            document.getElementById('avg-time').textContent = '加载失败';
                        }});
                }}

                // 清零统计数据
                function resetStats() {{
                    if (!confirm('确定要清零所有统计数据吗？')) {{
                        return;
                    }}
                    
                    fetch('/reset_stats', {{ method: 'POST' }})
                        .then(res => res.json())
                        .then(data => {{
                            if (data.success) {{
                                showMessage('统计数据已清零', 'success');
                                loadStats();
                            }}
                        }})
                        .catch(err => {{
                            console.error('清零失败:', err);
                            alert('清零失败: ' + err);
                        }});
                }}

                // 显示消息
                function showMessage(msg, type) {{
                    const msgDiv = document.getElementById('message');
                    msgDiv.textContent = msg;
                    msgDiv.className = 'message ' + type;
                    msgDiv.style.display = 'block';
                    setTimeout(() => {{
                        msgDiv.style.display = 'none';
                    }}, 3000);
                }}

                // 加载视频列表
                function loadVideos() {{
                    fetch('/videos')
                        .then(res => res.text())
                        .then(html => {{
                            // 解析HTML获取视频列表
                            const parser = new DOMParser();
                            const doc = parser.parseFromString(html, 'text/html');
                            const table = doc.querySelector('table');
                            const container = document.getElementById('video-list-container');
                            
                            if (table) {{
                                // 提取视频信息
                                const rows = table.querySelectorAll('tbody tr');
                                if (rows.length > 0) {{
                                    let videoHtml = '<table style="width: 100%; border-collapse: collapse; margin: 10px 0;">';
                                    videoHtml += '<thead><tr><th style="padding: 8px; text-align: left; border-bottom: 1px solid #ddd;">文件名</th><th style="padding: 8px; text-align: left; border-bottom: 1px solid #ddd;">大小</th><th style="padding: 8px; text-align: left; border-bottom: 1px solid #ddd;">操作</th></tr></thead><tbody>';
                                    
                                    rows.forEach((row, index) => {{
                                        if (index < 5) {{ // 只显示最近5个视频
                                            const cells = row.querySelectorAll('td');
                                            const filename = cells[0].textContent;
                                            const size = cells[1].textContent;
                                            const downloadUrl = cells[3].querySelector('a').href;
                                            
                                            videoHtml += `<tr style="border-bottom: 1px solid #eee;">
                                                <td style="padding: 8px;">${{filename}}</td>
                                                <td style="padding: 8px;">${{size}}</td>
                                                <td style="padding: 8px;">
                                                    <a href="${{downloadUrl}}" style="background: #4CAF50; color: white; padding: 6px 12px; border-radius: 4px; text-decoration: none; display: inline-block;" download>⬇️ 下载</a>
                                                </td>
                                            </tr>`;
                                        }}
                                    }});
                                    
                                    videoHtml += '</tbody></table>';
                                    if (rows.length > 5) {{
                                        videoHtml += `<p style="color: #666; font-size: 14px; margin-top: 10px;">显示最近5个视频，共 ${{rows.length}} 个视频</p>`;
                                    }}
                                    container.innerHTML = videoHtml;
                                }} else {{
                                    container.innerHTML = '<p style="color: #666;">暂无视频文件</p>';
                                }}
                            }} else {{
                                const message = doc.querySelector('.message');
                                if (message && message.textContent.includes('未启用')) {{
                                    container.innerHTML = '<p style="color: #856404; padding: 10px; background: #fff3cd; border-radius: 5px;">⚠️ 视频保存功能未启用，请使用 <code>--enable-video-save</code> 参数启动服务</p>';
                                }} else {{
                                    container.innerHTML = '<p style="color: #666;">暂无视频文件</p>';
                                }}
                            }}
                        }})
                        .catch(err => {{
                            console.error('加载视频列表失败:', err);
                            document.getElementById('video-list-container').innerHTML = '<p style="color: #f44336;">加载视频列表失败</p>';
                        }});
                }}
                
                // 初始加载和定时刷新
                loadStats();
                loadVideos();
                setInterval(loadStats, 3000);  // 每3秒刷新一次统计
                setInterval(loadVideos, 10000);  // 每10秒刷新一次视频列表
            </script>
        </body>
        </html>
        """
        self.wfile.write(html.encode('utf-8'))
    
    def handle_health(self):
        """健康检查"""
        self.send_response(200)
        self.send_header('Content-type', 'application/json')
        self.end_headers()
        
        response = {
            'status': 'healthy',
            'service_id': CONFIG['service_id'],
            'version': CONFIG['version'],
            'model_loaded': OM_LOADED
        }
        self.wfile.write(json.dumps(response).encode('utf-8'))
    
    def handle_stats(self):
        """性能统计"""
        self.send_response(200)
        self.send_header('Content-type', 'application/json')
        self.end_headers()
        
        global STATS
        
        avg_inference_time = 0.0
        if STATS['total_requests'] > 0:
            avg_inference_time = STATS['total_inference_time'] / STATS['total_requests']
        
        stats = {
            'statistics': dict(STATS),
            'avg_inference_time_per_request': round(avg_inference_time, 2)
        }
        
        self.wfile.write(json.dumps(stats, indent=2).encode('utf-8'))
    
    def handle_reset_stats(self):
        """清零统计数据"""
        global STATS
        
        STATS['total_requests'] = 0
        STATS['total_inference_time'] = 0.0
        STATS['last_inference_time'] = 0.0
        STATS['last_total_time'] = 0.0
        
        print(f"\n[{time.strftime('%H:%M:%S')}] 统计数据已清零")
        
        self.send_response(200)
        self.send_header('Content-type', 'application/json')
        self.end_headers()
        
        response = {
            'success': True,
            'message': '统计数据已清零'
        }
        self.wfile.write(json.dumps(response).encode('utf-8'))
    
    def handle_video_list(self):
        """处理视频列表请求"""
        try:
            video_dir = Path(CONFIG.get('video_save_dir', './videos'))
            
            # 检查是否启用视频保存
            if not CONFIG.get('enable_video_save', False):
                self.send_response(200)
                self.send_header('Content-type', 'text/html; charset=utf-8')
                self.end_headers()
                html = """
            <!DOCTYPE html>
            <html>
            <head>
                <title>视频列表 - {}</title>
                <meta charset="utf-8">
                <style>
                    body {{ font-family: Arial, sans-serif; max-width: 900px; margin: 50px auto; padding: 20px; }}
                    .container {{ background: white; padding: 30px; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }}
                    .message {{ padding: 20px; background: #fff3cd; border-radius: 5px; color: #856404; }}
                </style>
            </head>
            <body>
                <div class="container">
                    <h1>📹 视频列表</h1>
                    <div class="message">
                        <p>⚠️ 视频保存功能未启用</p>
                        <p>请使用 <code>--enable-video-save</code> 参数启动服务以启用视频保存功能。</p>
                    </div>
                    <p><a href="/">← 返回首页</a></p>
                </div>
            </body>
            </html>
            """.format(CONFIG['name'])
                self.wfile.write(html.encode('utf-8'))
                return
            
            # 获取所有视频文件
            video_files = []
            if video_dir.exists():
                for video_file in sorted(video_dir.glob('*.mp4'), key=lambda x: x.stat().st_mtime, reverse=True):
                    try:
                        stat = video_file.stat()
                        size_mb = stat.st_size / (1024 * 1024)
                        mtime = datetime.fromtimestamp(stat.st_mtime).strftime('%Y-%m-%d %H:%M:%S')
                        video_files.append({
                            'filename': video_file.name,
                            'size_mb': round(size_mb, 2),
                            'modified_time': mtime
                        })
                    except Exception as e:
                        print(f"处理视频文件 {video_file} 时出错: {e}")
                        continue
            
            self.send_response(200)
            self.send_header('Content-type', 'text/html; charset=utf-8')
            self.end_headers()
            
            html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>视频列表 - {CONFIG['name']}</title>
            <meta charset="utf-8">
            <style>
                body {{
                    font-family: Arial, sans-serif;
                    max-width: 1000px;
                    margin: 50px auto;
                    padding: 20px;
                    background: #f5f5f5;
                }}
                .container {{
                    background: white;
                    padding: 30px;
                    border-radius: 10px;
                    box-shadow: 0 2px 10px rgba(0,0,0,0.1);
                }}
                h1 {{
                    color: #333;
                    border-bottom: 3px solid #2196F3;
                    padding-bottom: 10px;
                }}
                table {{
                    width: 100%;
                    border-collapse: collapse;
                    margin: 20px 0;
                }}
                th, td {{
                    padding: 12px;
                    text-align: left;
                    border-bottom: 1px solid #ddd;
                }}
                th {{
                    background: #f9f9f9;
                    font-weight: bold;
                    color: #333;
                }}
                tr:hover {{
                    background: #f5f5f5;
                }}
                .btn-download {{
                    background: #4CAF50;
                    color: white;
                    border: none;
                    padding: 8px 16px;
                    border-radius: 5px;
                    cursor: pointer;
                    text-decoration: none;
                    display: inline-block;
                }}
                .btn-download:hover {{
                    background: #45a049;
                }}
                .empty {{
                    padding: 20px;
                    text-align: center;
                    color: #666;
                }}
            </style>
        </head>
        <body>
            <div class="container">
                <h1>📹 视频列表</h1>
                <p><a href="/">← 返回首页</a></p>
        """
        
            if video_files:
                html += """
                    <table>
                        <thead>
                            <tr>
                                <th>文件名</th>
                                <th>大小</th>
                                <th>修改时间</th>
                                <th>操作</th>
                            </tr>
                        </thead>
                        <tbody>
                """
                for video in video_files:
                    html += f"""
                            <tr>
                                <td>{video['filename']}</td>
                                <td>{video['size_mb']} MB</td>
                                <td>{video['modified_time']}</td>
                                <td>
                                    <a href="/videos/{video['filename']}" class="btn-download" download>下载</a>
                                </td>
                            </tr>
                    """
                html += """
                        </tbody>
                    </table>
                """
            else:
                html += """
                    <div class="empty">
                        <p>暂无视频文件</p>
                    </div>
                """
            
            html += """
            </div>
        </body>
        </html>
            """
            self.wfile.write(html.encode('utf-8'))
        except Exception as e:
            print(f"处理视频列表请求时出错: {str(e)}")
            import traceback
            traceback.print_exc()
            self.send_error(500, f"Internal server error: {str(e)}")
    
    def handle_writing_videos(self):
        """返回正在写入的视频列表API"""
        global VIDEO_WRITERS, VIDEO_WRITERS_LOCK
        
        self.send_response(200)
        self.send_header('Content-type', 'application/json')
        self.end_headers()
        
        writing_videos = []
        with VIDEO_WRITERS_LOCK:
            for task_id, video_info in VIDEO_WRITERS.items():
                if isinstance(video_info, dict) and 'path' in video_info:
                    video_path = video_info['path']
                    if video_path.exists():
                        writing_videos.append(str(video_path.resolve()))
        
        response = {'videos': writing_videos}
        self.wfile.write(json.dumps(response).encode('utf-8'))
    
    def handle_video_download(self):
        """处理视频下载请求"""
        # 从路径中提取文件名
        filename = self.path.replace('/videos/', '')
        # 安全检查：防止路径遍历攻击
        if '..' in filename or '/' in filename or '\\' in filename:
            self.send_error(400, "Invalid filename")
            return
        
        video_dir = Path(CONFIG.get('video_save_dir', './videos'))
        video_path = video_dir / filename
        
        # 检查文件是否存在
        if not video_path.exists() or not video_path.is_file():
            self.send_error(404, "Video not found")
            return
        
        # 检查文件扩展名
        if not filename.lower().endswith('.mp4'):
            self.send_error(400, "Invalid file type")
            return
        
        # 检查是否正在写入
        global VIDEO_WRITERS, VIDEO_WRITERS_LOCK
        is_writing = False
        video_path_str = str(video_path.resolve())
        with VIDEO_WRITERS_LOCK:
            for task_id, video_info in VIDEO_WRITERS.items():
                if isinstance(video_info, dict) and 'path' in video_info:
                    if str(video_info['path'].resolve()) == video_path_str:
                        is_writing = True
                        break
        
        if is_writing:
            self.send_error(409, "Video is being written")
            return
        
        try:
            # 读取文件并发送
            file_size = video_path.stat().st_size
            
            self.send_response(200)
            self.send_header('Content-type', 'video/mp4')
            self.send_header('Content-Length', str(file_size))
            self.send_header('Content-Disposition', f'attachment; filename="{filename}"')
            self.end_headers()
            
            # 分块读取并发送文件（避免大文件占用内存）
            with open(video_path, 'rb') as f:
                while True:
                    chunk = f.read(8192)  # 8KB chunks
                    if not chunk:
                        break
                    self.wfile.write(chunk)
            
        except Exception as e:
            print(f"视频下载错误: {str(e)}")
            self.send_error(500, "Internal server error")
    
    def handle_inference(self):
        """处理推理请求（绊线检测专用，单线程直接推理，避免ACL跨线程问题）"""
        global TRACKER_MANAGER, TRACKER_LOCK, LAST_CROSSING_COUNTS, LAST_CROSSING_COUNTS_LOCK, CLASS_NAMES, STATS
        
        start_time = time.time()
        image_url = ''
        task_id = 'unknown'
        request_id = uuid.uuid4().hex
        
        try:
            content_length = int(self.headers['Content-Length'])
            post_data = self.rfile.read(content_length)
            request_data = json.loads(post_data.decode('utf-8'))
            
            image_url = request_data.get('image_url', '')
            task_id = request_data.get('task_id', 'unknown')
            task_type = request_data.get('task_type', 'unknown')
            
            if not image_url:
                raise ValueError("缺少image_url参数")
            
            # 加载算法配置文件（参考实时算法：先从请求数据中获取，如果没有再从URL加载）
            algo_config = request_data.get('algo_config')
            if not algo_config:
                algo_config = load_algo_config(image_url)
            
            if not algo_config:
                raise ValueError("绊线人数统计必须有配置文件")
            
            # 下载图片
            with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp_file:
                temp_image_path = tmp_file.name
            
            try:
                urllib.request.urlretrieve(image_url, temp_image_path)
                
                # 读取图片
                image = cv2.imread(temp_image_path)
                if image is None:
                    raise ValueError("无法读取图片")
            finally:
                # 清理临时文件
                try:
                    os.remove(temp_image_path)
                except:
                    pass
            
            # 直接推理（主线程执行，避免ACL跨线程问题）
            inference_start = time.time()
            boxes_out = om_infer(CONFIG['model_path'], image, debug=False)
            inference_time = (time.time() - inference_start) * 1000
            
            # 更新统计（只有推理成功才统计）
            STATS['total_requests'] += 1
            STATS['total_inference_time'] += inference_time
            STATS['last_inference_time'] = inference_time
            
            # 获取算法参数
            confidence_threshold = 0.5
            if algo_config:
                algo_params = algo_config.get('algorithm_params', {})
                confidence_threshold = algo_params.get('confidence_threshold', 0.5)
            
            # 解析OM模型推理结果
            objects = []
            detections = []
            
            if boxes_out is not None and len(boxes_out) > 0:
                for b in boxes_out:
                    x1, y1, x2, y2, conf, cls_id = b
                    if float(conf) < confidence_threshold:
                        continue
                    cls_id = int(cls_id)
                    class_name = CLASS_NAMES[cls_id] if 0 <= cls_id < len(CLASS_NAMES) else str(cls_id)
                    
                    obj = {
                        'class': class_name,
                        'confidence': float(conf),
                        'bbox': [float(x1), float(y1), float(x2), float(y2)]
                    }
                    objects.append(obj)
                    detections.append(obj)
            
            # 【区域过滤】如果配置了检测区域，只保留区域内的物体
            original_count = len(objects)
            if algo_config:
                regions = algo_config.get('regions', [])
                if regions:
                    image_size = (image.shape[1], image.shape[0])
                    objects = filter_objects_by_region(objects, algo_config, image_size)
                    detections = filter_objects_by_region(detections, algo_config, image_size)
                    filtered_count = original_count - len(objects)
                    if filtered_count > 0:
                        print(f"  ℹ️  区域过滤: 原始 {original_count} 个 → 区域内 {len(objects)} 个 (过滤掉 {filtered_count} 个)")
            
            # 构建结果
            result_data = {
                'objects': objects,
                'total_count': len(objects),
            }
            
            # 跟踪和绊线检测
            line_crossing_results = {}
            trackers = []
            
            if TRACKER_MANAGER:
                with TRACKER_LOCK:
                    trackers = TRACKER_MANAGER.update(task_id, detections)
            
            if algo_config and trackers:
                regions = algo_config.get('regions', [])
                
                if regions:
                    image_size = (image.shape[1], image.shape[0])
                    with TRACKER_LOCK:
                        line_crossing_results = TRACKER_MANAGER.check_line_crossing(task_id, regions, image_size)
                    
                    if line_crossing_results:
                        # 【绊线增量告警】只有发生新穿越时才返回告警
                        total_crossed = sum(info['count'] for info in line_crossing_results.values())
                        
                        with LAST_CROSSING_COUNTS_LOCK:
                            last_count = LAST_CROSSING_COUNTS.get(task_id, 0)
                            
                            if total_crossed > last_count:
                                # 有新穿越 → 返回完整结果（触发告警）
                                new_crossings = total_crossed - last_count
                                result_data['person_count'] = new_crossings
                                result_data['line_crossing'] = line_crossing_results
                                LAST_CROSSING_COUNTS[task_id] = total_crossed
                                print(f"  ✅ 检测到新穿越: {last_count} → {total_crossed} (+{new_crossings})，上传告警")
                            else:
                                # 无新穿越 → 返回空结果（不触发告警）
                                result_data['total_count'] = 0
                                result_data['objects'] = []
                                print(f"  ℹ️  无新穿越（累计={total_crossed}），返回空结果（不上传告警）")
                    else:
                        # 无有效跨线检测结果 → 返回空结果
                        result_data['total_count'] = 0
                        result_data['objects'] = []
                        print(f"  ℹ️  绊线人数统计但无有效跨线结果，返回空结果")
            
            # 计算平均置信度（注意：无新穿越时 objects 会被清空）
            avg_confidence = 0.0
            if result_data.get('objects') and len(result_data['objects']) > 0:
                avg_confidence = sum(obj['confidence'] for obj in result_data['objects']) / len(result_data['objects'])
            
            # 计算总处理时间
            total_time = (time.time() - start_time) * 1000
            STATS['last_total_time'] = total_time
            
            # 【视频保存】如果启用了视频保存，绘制并保存视频
            # 只有在enable_video_save为True时才执行绘制和保存操作，避免多余耗时
            if CONFIG.get('enable_video_save', False):
                try:
                    # 从algo_config或CONFIG中获取绘制配置（优先使用algo_config）
                    video_config = algo_config.get('video_config', {}) if algo_config else {}
                    draw_trajectory_enabled = video_config.get('draw_trajectory', CONFIG.get('video_draw_trajectory', True))
                    draw_line_config_enabled = video_config.get('draw_line_config', CONFIG.get('video_draw_line_config', True))
                    draw_stats_enabled = video_config.get('draw_stats', CONFIG.get('video_draw_stats', True))
                    
                    # 创建图像副本（避免修改原图）
                    video_frame = image.copy()
                    image_size = (image.shape[1], image.shape[0])
                    
                    # 获取或创建视频写入器（自动分段）
                    video_writer = get_or_create_video_writer(task_id, image.shape)
                    
                    if video_writer:
                        # 根据配置决定是否绘制绊线配置
                        if draw_line_config_enabled and algo_config:
                            regions = algo_config.get('regions', [])
                            if regions:
                                line_regions = [r for r in regions if r.get('type') == 'line' and r.get('enabled', True)]
                                if line_regions:
                                    draw_line_config(video_frame, line_regions, image_size)
                        
                        # 根据配置决定是否绘制跟踪轨迹
                        if draw_trajectory_enabled:
                            for tracker in trackers:
                                draw_trajectory(video_frame, tracker)
                        
                        # 根据配置决定是否绘制统计信息
                        if draw_stats_enabled:
                            draw_stats(video_frame, line_crossing_results, inference_time, total_time, len(trackers))
                        
                        # 写入视频
                        video_writer.write(video_frame)
                except Exception as e:
                    print(f"  ⚠️  视频保存失败: {str(e)}")
            
            response = {
                'success': True,
                'result': result_data,
                'confidence': avg_confidence,
                'inference_time_ms': round(inference_time, 2),  # 模型推理时间
                'total_time_ms': round(total_time, 2),  # 全部处理时间（包含下载、预处理、推理、后处理）
                'image_url': image_url,  # 请求的图片URL
                'task_id': task_id,  # 任务ID
                'request_id': request_id  # 用于日志关联
            }
            
            # 发送响应
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps(response).encode('utf-8'))
            
        except Exception as e:
            print(f"  推理失败: {str(e)}")
            import traceback
            traceback.print_exc()
            
            total_time = (time.time() - start_time) * 1000
            
            error_response = {
                'success': False,
                'error': str(e),
                'confidence': 0.0,
                'inference_time_ms': 0,
                'total_time_ms': round(total_time, 2),
                'image_url': image_url,  # 请求的图片URL
                'task_id': task_id,  # 任务ID
                'request_id': request_id  # 用于日志关联
            }
            
            self.send_response(500)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps(error_response).encode('utf-8'))


def load_model():
    """加载OM模型并初始化ACL环境"""
    global OM_LOADED
    
    print(f"正在初始化ACL并加载OM模型: {CONFIG['model_path']}")
    start_time = time.time()
    
    # 初始化ACL
    init_acl_resource(device_id=CONFIG.get('device_id', 0))
    # 加载OM模型
    load_om_model(CONFIG['model_path'])
    OM_LOADED = True
    
    # 注册退出时清理资源
    atexit.register(release_acl_resource)
    
    load_time = time.time() - start_time
    print(f"✓ OM模型加载成功 (耗时: {load_time:.2f}秒)")


def register_service(quiet=False):
    """注册到EasyDarwin"""
    global REGISTERED
    
    url = f"{CONFIG['easydarwin_url']}/api/v1/ai_analysis/register"
    
    # 优先使用手动指定的主机IP，然后自动检测
    endpoint = f"http://{CONFIG['host']}:{CONFIG['port']}/infer"
    if CONFIG['host'] == '0.0.0.0':
        # 如果手动指定了主机IP且不为空，直接使用
        host_ip = CONFIG.get('host_ip')
        if host_ip and host_ip.strip():
            endpoint = f"http://{host_ip.strip()}:{CONFIG['port']}/infer"
        else:
            # 尝试自动获取本地IP
            import socket
            try:
                # 尝试多种方法获取本地IP
                hostname = socket.gethostname()
                local_ip = socket.gethostbyname(hostname)
                
                # 如果获取到的是127.0.0.1或localhost，尝试其他方法
                if local_ip in ['127.0.0.1', '::1']:
                    # 通过连接外部地址来获取本机IP
                    try:
                        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
                        s.connect(('8.8.8.8', 80))
                        local_ip = s.getsockname()[0]
                        s.close()
                    except:
                        pass
                
                if local_ip and local_ip not in ['127.0.0.1', '::1']:
                    endpoint = f"http://{local_ip}:{CONFIG['port']}/infer"
                    if not quiet:
                        print(f"  检测到本地IP: {local_ip}")
                else:
                    # 默认使用172.16.5.207
                    endpoint = f"http://172.16.5.207:{CONFIG['port']}/infer"
            except Exception as e:
                # 默认使用172.16.5.207
                endpoint = f"http://172.16.5.207:{CONFIG['port']}/infer"
                if not quiet:
                    print(f"  警告: 无法自动获取本地IP ({str(e)}), 使用172.16.5.207")
    
    payload = {
        'service_id': CONFIG['service_id'],
        'name': CONFIG['name'],
        'task_types': CONFIG['task_types'],
        'endpoint': endpoint,
        'version': CONFIG['version']
    }
    
    if not quiet:
        print(f"\n正在注册到 {CONFIG['easydarwin_url']}...")
        print(f"  服务ID: {CONFIG['service_id']}")
        print(f"  服务名称: {CONFIG['name']}")
        print(f"  任务类型: {CONFIG['task_types']} (将被注册到EasyDarwin)")
        print(f"  推理端点: {endpoint}")
        print(f"  注册Payload: {json.dumps(payload, ensure_ascii=False)}")
    
    try:
        response = requests.post(url, json=payload, timeout=10)
        response.raise_for_status()
        
        result = response.json()
        if result.get('ok'):
            if not quiet:
                print(f"✓ 注册成功")
            REGISTERED = True
            return True
        else:
            if not quiet:
                print(f"✗ 注册失败: {result}")
            return False
    except requests.exceptions.ConnectionError:
        if not quiet:
            print(f"✗ 注册失败: 无法连接到 {CONFIG['easydarwin_url']}（平台可能未启动）")
        return False
    except Exception as e:
        if not quiet:
            print(f"✗ 注册失败: {str(e)}")
        return False


def unregister_service():
    """注销服务"""
    # 规范化EasyDarwin URL
    easydarwin_url = CONFIG['easydarwin_url'].strip()
    if not easydarwin_url.startswith('http://') and not easydarwin_url.startswith('https://'):
        easydarwin_url = f"http://{easydarwin_url}"
    if easydarwin_url.endswith('/'):
        easydarwin_url = easydarwin_url[:-1]
    
    url = f"{easydarwin_url}/api/v1/ai_analysis/unregister/{CONFIG['service_id']}"
    
    print(f"\n正在注销服务: {CONFIG['service_id']}")
    
    try:
        response = requests.delete(url, timeout=10)
        response.raise_for_status()
        print("✓ 注销成功")
    except Exception as e:
        print(f"✗ 注销失败: {str(e)}")


def heartbeat_loop():
    """心跳循环"""
    global RUNNING, REGISTERED
    
    url = f"{CONFIG['easydarwin_url']}/api/v1/ai_analysis/heartbeat/{CONFIG['service_id']}"
    
    print(f"心跳线程已启动（每{CONFIG['heartbeat_interval']}秒）")
    
    consecutive_failures = 0
    max_failures = 3  # 连续失败3次后重新尝试注册
    
    while RUNNING:
        time.sleep(CONFIG['heartbeat_interval'])
        
        if not RUNNING:
            break
        
        try:
            # 计算平均推理时间
            avg_inference_time = 0.0
            # 计算平均推理时间
            avg_inference_time = 0.0
            if STATS['total_requests'] > 0:
                avg_inference_time = STATS['total_inference_time'] / STATS['total_requests']
            
            # 携带统计信息
            payload = {
                'total_requests': STATS['total_requests'],
                'avg_inference_time_ms': round(avg_inference_time, 2),
                'last_inference_time_ms': round(STATS['last_inference_time'], 2),  # 最近一次推理时间
                'last_total_time_ms': round(STATS['last_total_time'], 2)  # 最近一次总耗时
            }
            
            response = requests.post(url, json=payload, timeout=5)
            if response.status_code == 200:
                if consecutive_failures > 0:
                    print(f"[{time.strftime('%H:%M:%S')}] 心跳发送成功（已恢复）")
                    consecutive_failures = 0
                else:
                    # 正常时不打印日志，避免刷屏
                    pass
            else:
                consecutive_failures += 1
                print(f"[{time.strftime('%H:%M:%S')}] 心跳发送失败: HTTP {response.status_code}")
        except Exception as e:
            consecutive_failures += 1
            print(f"[{time.strftime('%H:%M:%S')}] 心跳发送失败: {str(e)}")
        
        # 如果连续失败多次，可能平台重启了，需要重新注册
        if consecutive_failures >= max_failures:
            print(f"[{time.strftime('%H:%M:%S')}] 连续失败{max_failures}次，尝试重新注册...")
            REGISTERED = False
            if register_service(quiet=True):
                consecutive_failures = 0
                print(f"[{time.strftime('%H:%M:%S')}] ✓ 重新注册成功，心跳继续")
            else:
                print(f"[{time.strftime('%H:%M:%S')}] ✗ 重新注册失败，继续重试...")


def register_retry_loop():
    """注册重试循环（后台持续尝试注册，直到成功）"""
    global RUNNING, REGISTERED, HEARTBEAT_THREAD
    
    retry_interval = 30  # 每30秒重试一次
    print(f"注册重试线程已启动（每{retry_interval}秒尝试注册，直到平台启动）")
    
    while RUNNING and not REGISTERED:
        time.sleep(retry_interval)
        
        if not RUNNING:
            break
        
        if REGISTERED:
            break
        
        # 尝试注册（quiet模式，减少日志输出）
        timestamp = time.strftime('%H:%M:%S')
        print(f"[{timestamp}] 正在尝试注册到 {CONFIG['easydarwin_url']}...")
        if register_service(quiet=True):
            print(f"[{timestamp}] ✓ 注册成功！开始心跳...")
            # 注册成功后，启动心跳线程
            HEARTBEAT_THREAD = threading.Thread(target=heartbeat_loop, daemon=True)
            HEARTBEAT_THREAD.start()
            break
        else:
            print(f"[{timestamp}] ✗ 注册失败（平台可能未启动），{retry_interval}秒后重试...")


def signal_handler(sig, frame):
    """信号处理器（优雅退出）"""
    global RUNNING
    
    print("\n\n收到退出信号，正在关闭服务...")
    RUNNING = False


def main():
    """主函数"""
    global RUNNING, HEARTBEAT_THREAD, REGISTER_THREAD, TRACKER_MANAGER
    
    parser = argparse.ArgumentParser(description='YOLOv11x绊线人数统计算法服务')
    parser.add_argument('--service-id', default='yolo11x_line_crossing',
                        help='服务ID (默认: yolo11x_line_crossing)')
    parser.add_argument('--name', default='YOLOv11x绊线人数统计算法',
                        help='服务名称')
    parser.add_argument('--port', type=int, default=7903,
                        help='监听端口 (默认: 7903)')
    parser.add_argument('--host', default='0.0.0.0',
                        help='监听地址 (默认: 0.0.0.0)')
    parser.add_argument('--easydarwin', default='172.16.5.207:5066',
                        help='EasyDarwin地址')
    parser.add_argument('--model', default='./weight/best.om',
                        help='OM模型路径 (默认: ./weight/best.om)')
    parser.add_argument('--device-id', type=int, default=0,
                        help='Ascend NPU设备ID (默认: 0)')
    parser.add_argument('--host-ip', type=str, default=None,
                        help='主机IP地址 (用于注册到EasyDarwin，默认自动检测)')
    parser.add_argument('--task-types', nargs='+', default=['绊线人数统计'],
                        help='支持的任务类型 (默认: 绊线人数统计)')
    parser.add_argument('--no-register', action='store_true',
                        help='不注册到EasyDarwin')
    parser.add_argument('--batch-size', type=int, default=8,
                        help='批处理大小 (默认: 8)')
    parser.add_argument('--batch-timeout', type=float, default=0.1,
                        help='批处理超时（秒）')
    parser.add_argument('--no-batching', action='store_true',
                        help='禁用批处理')
    parser.add_argument('--enable-video-save', action='store_true',
                        help='启用视频保存功能（默认关闭）')
    parser.add_argument('--video-save-dir', default='./videos',
                        help='视频保存目录 (默认: ./videos)')
    parser.add_argument('--video-fps', type=int, default=25,
                        help='视频帧率 (默认: 25)')
    parser.add_argument('--video-segment-duration', type=int, default=60,
                        help='每个视频片段时长（秒）(默认: 60秒，即1分钟)')
    parser.add_argument('--video-segment-max-size-mb', type=int, default=500,
                        help='每个视频片段最大大小（MB）(默认: 500MB)')
    
    args = parser.parse_args()
    
    # 更新配置
    CONFIG['service_id'] = args.service_id
    CONFIG['name'] = args.name
    CONFIG['task_types'] = args.task_types  # 确保任务类型可以被命令行参数覆盖
    CONFIG['port'] = args.port
    CONFIG['host'] = args.host
    CONFIG['device_id'] = args.device_id  # NPU设备ID
    CONFIG['host_ip'] = args.host_ip  # 添加主机IP配置
    CONFIG['easydarwin_url'] = args.easydarwin
    # 规范化 EasyDarwin 基地址，确保包含协议前缀
    if not (CONFIG['easydarwin_url'].startswith('http://') or CONFIG['easydarwin_url'].startswith('https://')):
        CONFIG['easydarwin_url'] = f"http://{CONFIG['easydarwin_url']}"
    
    CONFIG['model_path'] = args.model
    # 如果模型路径是相对路径，转换为绝对路径
    if not os.path.isabs(CONFIG['model_path']):
        CONFIG['model_path'] = os.path.abspath(CONFIG['model_path'])
    CONFIG['batch_size'] = args.batch_size
    CONFIG['batch_timeout'] = args.batch_timeout
    CONFIG['enable_batching'] = not args.no_batching
    CONFIG['enable_video_save'] = args.enable_video_save
    CONFIG['video_save_dir'] = args.video_save_dir
    CONFIG['video_fps'] = args.video_fps
    CONFIG['video_segment_duration'] = args.video_segment_duration
    CONFIG['video_segment_max_size_mb'] = args.video_segment_max_size_mb
    
    # 如果启用视频保存，确保目录存在
    if CONFIG['enable_video_save']:
        video_dir = Path(CONFIG['video_save_dir'])
        video_dir.mkdir(parents=True, exist_ok=True)
        print(f"✓ 视频保存已启用，保存目录: {video_dir.absolute()}")
    
    # 注册信号处理器
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    print("=" * 60)
    print(f"  {CONFIG['name']} v{CONFIG['version']}")
    print("=" * 60)
    
    # 加载模型
    load_model()
    
    # 初始化跟踪器管理器（优化参数：平衡跟踪稳定性和匹配准确性）
    TRACKER_MANAGER = TrackerManager(
        iou_threshold=0.05,      # 极低IOU阈值，最大容忍遮挡
        max_missed=10,           # 进一步增加最大丢失帧数，减少轨迹丢失
        center_distance=200,    # 适度降低中心距离阈值，避免误匹配
        frame_step=1
    )
    print("✓ 跟踪器管理器已初始化")
    
    # 不再使用批处理（改为单线程直接推理，避免ACL跨线程问题）
    print("✓ 单线程直接推理模式（已禁用批处理，避免ACL跨线程问题）")
    
    # 注册到EasyDarwin（优化：支持平台后启动）
    if not args.no_register:
        # 启动时先尝试注册一次
        if register_service():
            # 如果立即成功，启动心跳线程
            HEARTBEAT_THREAD = threading.Thread(target=heartbeat_loop, daemon=True)
            HEARTBEAT_THREAD.start()
        else:
            # 如果失败（平台未启动），启动注册重试线程
            print("\n⚠ 平台可能未启动，将在后台持续尝试注册...")
            REGISTER_THREAD = threading.Thread(target=register_retry_loop, daemon=True)
            REGISTER_THREAD.start()
    else:
        print("\n⚠ 跳过注册到EasyDarwin")
    
    # 启动HTTP服务器（单线程模式，避免ACL跨线程问题）
    server_address = (CONFIG['host'], CONFIG['port'])
    httpd = HTTPServer(server_address, YOLOInferenceHandler)
    print(f"✓ 单线程推理模式已启用（避免ACL跨线程问题）")
    
    print(f"\n绊线人数统计算法服务已启动")
    print(f"  服务ID: {CONFIG['service_id']}")
    print(f"  服务名称: {CONFIG['name']}")
    print(f"  任务类型: {CONFIG['task_types']}")
    print(f"  监听地址: {CONFIG['host']}:{CONFIG['port']}")
    print(f"  推理端点: http://{CONFIG['host']}:{CONFIG['port']}/infer")
    print(f"\n等待推理请求... (按Ctrl+C退出)")
    print("=" * 60)
    
    # 运行服务器
    try:
        while RUNNING:
            httpd.handle_request()
    except KeyboardInterrupt:
        pass
    finally:
        if REGISTERED:
            unregister_service()
        
        # 清理视频写入器
        try:
            global VIDEO_WRITERS, VIDEO_WRITERS_LOCK
            with VIDEO_WRITERS_LOCK:
                for task_id, video_info in VIDEO_WRITERS.items():
                    try:
                        if isinstance(video_info, dict) and 'writer' in video_info:
                            video_info['writer'].release()
                            print(f"  ✓ 关闭视频写入器: task_id={task_id}, path={video_info.get('path')}")
                        else:
                            # 兼容旧格式
                            video_info.release()
                            print(f"  ✓ 关闭视频写入器: task_id={task_id}")
                    except Exception as e:
                        print(f"  ⚠️  关闭视频写入器失败 (task_id={task_id}): {e}")
                VIDEO_WRITERS.clear()
        except Exception as e:
            print(f"清理视频写入器失败: {e}")
        
        # 清理ACL资源
        try:
            if OM_LOADED:
                release_acl_resource()
        except Exception as e:
            print(f"释放ACL资源失败: {e}")
        
        print("\n服务已关闭")
        sys.exit(0)


if __name__ == '__main__':
    main()

