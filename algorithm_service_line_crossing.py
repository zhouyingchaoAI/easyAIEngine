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
import queue
import socketserver
from concurrent.futures import ThreadPoolExecutor
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse, parse_qs
import requests
import urllib.request
import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont
from ultralytics import YOLO
from collections import defaultdict
from pathlib import Path
from datetime import datetime
import uuid

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
    'model_path': '/cv_space/NWPU-Crowd/runs/exp_yolo11x3/weights/best.pt',
    'task_types': ['绊线人数统计'],
    'port': 7903,  # 使用不同的端口
    'host': '0.0.0.0',
    'easydarwin_url': 'http://localhost:5066',
    'heartbeat_interval': 30,  # 秒
    # 批处理配置
    'batch_size': 8,
    'batch_timeout': 0.1,
    'enable_batching': True,
    'max_queue_size': 100,
    # 可视化配置
    'enable_video': False,
}

# 全局变量
MODEL = None
RUNNING = True
HEARTBEAT_THREAD = None
TRACKER_MANAGER = None
TRACKER_LOCK = threading.Lock()
VIDEO_WRITERS = {}
VIDEO_WRITERS_LOCK = threading.Lock()

# 批处理相关
BATCH_PROCESSOR = None
BATCH_PROCESSOR_THREAD = None
INFERENCE_RESULTS = {}
INFERENCE_EVENTS = {}
BATCH_STATS = {
    'total_requests': 0,
    'total_batches': 0,
    'total_inference_time': 0.0,
    'avg_batch_size': 0.0,
    'max_batch_size': 0,
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


class ObjectTracker:
    """简单的目标跟踪器（基于IOU匹配）"""
    
    def __init__(self, track_id, bbox, confidence, class_name):
        self.track_id = track_id
        self.bbox = bbox
        self.confidence = confidence
        self.class_name = class_name
        self.center_history = []
        self.last_update = time.time()
        self.crossed_lines = set()
        
        center = self.get_center(bbox)
        self.center_history.append(center)
    
    @staticmethod
    def get_center(bbox):
        """获取边界框中心点"""
        x1, y1, x2, y2 = bbox
        return ((x1 + x2) / 2, (y1 + y2) / 2)
    
    def update(self, bbox, confidence):
        """更新跟踪器"""
        self.bbox = bbox
        self.confidence = confidence
        self.last_update = time.time()
        
        center = self.get_center(bbox)
        self.center_history.append(center)
        
        if len(self.center_history) > 10:
            self.center_history.pop(0)
    
    def get_trajectory(self):
        """获取轨迹（最近两个点）"""
        if len(self.center_history) >= 2:
            return self.center_history[-2], self.center_history[-1]
        return None
    
    @staticmethod
    def iou(bbox1, bbox2):
        """计算两个边界框的IOU"""
        x1_1, y1_1, x2_1, y2_1 = bbox1
        x1_2, y1_2, x2_2, y2_2 = bbox2
        
        x1_i = max(x1_1, x1_2)
        y1_i = max(y1_1, y1_2)
        x2_i = min(x2_1, x2_2)
        y2_i = min(y2_1, y2_2)
        
        if x2_i < x1_i or y2_i < y1_i:
            return 0.0
        
        intersection = (x2_i - x1_i) * (y2_i - y1_i)
        area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
        area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0.0


class TrackerManager:
    """目标跟踪管理器"""
    
    def __init__(self, iou_threshold=0.3, max_age=30):
        self.trackers = {}
        self.next_id = 1
        self.iou_threshold = iou_threshold
        self.max_age = max_age
        self.task_accumulators = defaultdict(lambda: defaultdict(int))
        self.last_reset_time = time.time()
        self.reset_interval = 24 * 60 * 60
    
    def update(self, detections):
        """更新跟踪器"""
        current_time = time.time()
        
        matched_trackers = set()
        matched_detections = set()
        
        for det_idx, detection in enumerate(detections):
            best_iou = 0
            best_tracker_id = None
            
            for track_id, tracker in self.trackers.items():
                iou = ObjectTracker.iou(detection['bbox'], tracker.bbox)
                if iou > self.iou_threshold and iou > best_iou:
                    best_iou = iou
                    best_tracker_id = track_id
            
            if best_tracker_id is not None:
                self.trackers[best_tracker_id].update(detection['bbox'], detection['confidence'])
                matched_trackers.add(best_tracker_id)
                matched_detections.add(det_idx)
        
        for det_idx, detection in enumerate(detections):
            if det_idx not in matched_detections:
                tracker = ObjectTracker(
                    self.next_id,
                    detection['bbox'],
                    detection['confidence'],
                    detection['class']
                )
                self.trackers[self.next_id] = tracker
                self.next_id += 1
        
        to_remove = []
        for track_id, tracker in self.trackers.items():
            if current_time - tracker.last_update > self.max_age:
                to_remove.append(track_id)
        
        for track_id in to_remove:
            del self.trackers[track_id]
        
        return list(self.trackers.values())
    
    def check_and_reset_accumulators(self):
        """检查并重置累加器（每天自动清零）"""
        current_time = time.time()
        if current_time - self.last_reset_time >= self.reset_interval:
            print(f"  🔄 累加器自动清零（24小时间隔）")
            for task_id in self.task_accumulators:
                for region_id in self.task_accumulators[task_id]:
                    old_count = self.task_accumulators[task_id][region_id]
                    self.task_accumulators[task_id][region_id] = 0
                    print(f"    {task_id}.{region_id}: {old_count} -> 0")
            self.last_reset_time = current_time
    
    def check_line_crossing(self, task_id, regions, image_size=None):
        """检查跟踪目标是否跨越线段"""
        self.check_and_reset_accumulators()
        
        crossing_results = {}
        
        for region in regions:
            if not region.get('enabled', True):
                continue
            
            if region.get('type') != 'line':
                continue
            
            region_id = region.get('id')
            points = region.get('points', [])
            direction = region.get('properties', {}).get('direction', 'both')
            
            if len(points) < 2:
                continue
            
            p1 = tuple(points[0])
            p2 = tuple(points[1])
            
            if image_size and any(0 <= coord <= 1 for point in points for coord in point):
                width, height = image_size
                p1 = (int(points[0][0] * width), int(points[0][1] * height))
                p2 = (int(points[1][0] * width), int(points[1][1] * height))
            else:
                p1 = tuple(map(int, points[0]))
                p2 = tuple(map(int, points[1]))
            
            for tracker in self.trackers.values():
                trajectory = tracker.get_trajectory()
                if trajectory is None:
                    continue
                
                start_point, end_point = trajectory
                
                if self._segments_intersect(start_point, end_point, p1, p2):
                    cross_direction = self._get_cross_direction(start_point, end_point, p1, p2)
                    
                    should_count = False
                    if direction == 'both':
                        should_count = True
                    elif direction == 'in' and cross_direction == 'in':
                        should_count = True
                    elif direction == 'out' and cross_direction == 'out':
                        should_count = True
                    
                    if should_count:
                        current_time = time.time()
                        last_cross_time = getattr(tracker, f'last_cross_{region_id}', 0)
                        if current_time - last_cross_time > 0.5:
                            self.task_accumulators[task_id][region_id] += 1
                            print(f"    [绊线统计] ID:{tracker.track_id} 跨线 {region_id} ({cross_direction}) -> 累加: {self.task_accumulators[task_id][region_id]}")
                            
                            setattr(tracker, f'last_cross_{region_id}', current_time)
                            
                            cross_key = f"{task_id}_{region_id}_{tracker.track_id}"
                            if cross_key not in tracker.crossed_lines:
                                tracker.crossed_lines.add(cross_key)
            
            crossing_results[region_id] = {
                'region_name': region.get('name', region_id),
                'count': self.task_accumulators[task_id][region_id],
                'direction': direction
            }
        
        return crossing_results
    
    @staticmethod
    def _segments_intersect(p1, p2, p3, p4):
        """判断两条线段是否相交"""
        def ccw(A, B, C):
            return (C[1] - A[1]) * (B[0] - A[0]) > (B[1] - A[1]) * (C[0] - A[0])
        
        return ccw(p1, p3, p4) != ccw(p2, p3, p4) and ccw(p1, p2, p3) != ccw(p1, p2, p4)
    
    @staticmethod
    def _get_cross_direction(start, end, line_p1, line_p2):
        """判断跨越方向"""
        def cross_product(o, a, b):
            return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])
        
        cp_start = cross_product(line_p1, line_p2, start)
        cp_end = cross_product(line_p1, line_p2, end)
        
        if cp_start > 0 and cp_end < 0:
            return 'in'
        elif cp_start < 0 and cp_end > 0:
            return 'out'
        
        return 'unknown'


class BatchInferenceProcessor:
    """批处理推理处理器"""
    
    def __init__(self, model, batch_size=8, batch_timeout=0.1):
        self.model = model
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
            results = self.model(images)
            inference_time = (time.time() - inference_start) * 1000
            
            print(f"  ✓ 批量推理完成: {inference_time:.0f}ms")
            
            post_process_start = time.time()
            
            futures = []
            for idx, (request, result) in enumerate(zip(batch_requests, results)):
                future = self.post_process_pool.submit(
                    self._process_single_result_wrapper,
                    request, result, inference_time / batch_size, idx, batch_size
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
    
    def _process_single_result_wrapper(self, request, yolo_result, inference_time_per_image, idx, batch_size):
        """后处理包装器"""
        try:
            self._process_single_result(request, yolo_result, inference_time_per_image)
        except Exception as e:
            request.error = str(e)
            print(f"  请求 {idx+1}/{batch_size} 后处理失败: {str(e)}")
        finally:
            INFERENCE_RESULTS[request.request_id] = {
                'result': request.result,
                'error': request.error
            }
            request.event.set()
    
    def _process_single_result(self, request, yolo_result, inference_time_per_image):
        """处理单个推理结果（绊线专用版本）"""
        global TRACKER_MANAGER, TRACKER_LOCK, LAST_CROSSING_COUNTS, LAST_CROSSING_COUNTS_LOCK
        
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
        
        # 解析推理结果
        boxes = yolo_result.boxes
        objects = []
        detections = []
        
        if boxes is not None and len(boxes) > 0:
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().tolist()
                conf = float(box.conf[0].cpu().numpy())
                cls_id = int(box.cls[0].cpu().numpy())
                
                class_name = self.model.model.names[cls_id] if hasattr(self.model.model, 'names') else str(cls_id)
                
                if conf < confidence_threshold:
                    continue
                
                obj = {
                    'class': class_name,
                    'confidence': conf,
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
        
        if TRACKER_MANAGER and detections:
            with TRACKER_LOCK:
                trackers = TRACKER_MANAGER.update(detections)
        
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


def load_algo_config(image_url):
    """加载算法配置文件"""
    try:
        from urllib.parse import urlparse, unquote
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
                print(f"  ❌ 配置文件不存在或无法访问: {config_url}")
        else:
            print(f"  ❌ 无法从URL解析路径: {image_url}")
        
    except Exception as e:
        print(f"  ❌ 加载配置文件失败: {str(e)}")
    
    print(f"  ⚠️  未找到配置文件，绊线将不会工作")
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
        if self.path == '/health':
            self.handle_health()
        elif self.path == '/':
            self.handle_index()
        elif self.path == '/stats':
            self.handle_stats()
        else:
            self.send_error(404, "Not Found")
    
    def handle_index(self):
        """首页"""
        self.send_response(200)
        self.send_header('Content-type', 'text/html; charset=utf-8')
        self.end_headers()
        
        batching_status = "启用" if CONFIG['enable_batching'] else "禁用"
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
                        <strong>批处理模式</strong>
                        <span>{batching_status}</span>
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
                    <div class="info-item" style="margin-top: 15px;">
                        <strong>总批次数</strong>
                        <div class="stat-value" id="total-batches">加载中...</div>
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
                            const totalBatches = stats.total_batches || 0;
                            
                            // 计算平均推理时间
                            let avgTime = 0;
                            if (totalRequests > 0 && stats.total_inference_time) {{
                                avgTime = stats.total_inference_time / totalRequests;
                            }}
                            
                            document.getElementById('total-requests').textContent = totalRequests.toLocaleString();
                            document.getElementById('avg-time').textContent = avgTime.toFixed(2) + ' ms';
                            document.getElementById('total-batches').textContent = totalBatches.toLocaleString();
                        }})
                        .catch(err => {{
                            console.error('加载统计失败:', err);
                            document.getElementById('total-requests').textContent = '加载失败';
                            document.getElementById('avg-time').textContent = '加载失败';
                            document.getElementById('total-batches').textContent = '加载失败';
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

                // 初始加载和定时刷新
                loadStats();
                setInterval(loadStats, 3000);  // 每3秒刷新一次
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
            'model_loaded': MODEL is not None,
            'batching_enabled': CONFIG['enable_batching']
        }
        self.wfile.write(json.dumps(response).encode('utf-8'))
    
    def handle_stats(self):
        """性能统计"""
        self.send_response(200)
        self.send_header('Content-type', 'application/json')
        self.end_headers()
        
        global BATCH_STATS, BATCH_PROCESSOR
        
        stats = {
            'batching_enabled': CONFIG['enable_batching'],
            'batch_config': {
                'batch_size': CONFIG['batch_size'],
                'batch_timeout': CONFIG['batch_timeout'],
                'max_queue_size': CONFIG['max_queue_size']
            },
            'statistics': dict(BATCH_STATS) if CONFIG['enable_batching'] else {},
            'queue_size': BATCH_PROCESSOR.request_queue.qsize() if BATCH_PROCESSOR else 0
        }
        
        self.wfile.write(json.dumps(stats, indent=2).encode('utf-8'))
    
    def handle_reset_stats(self):
        """清零统计数据"""
        global BATCH_STATS
        
        BATCH_STATS['total_requests'] = 0
        BATCH_STATS['total_batches'] = 0
        BATCH_STATS['total_inference_time'] = 0.0
        BATCH_STATS['avg_batch_size'] = 0.0
        BATCH_STATS['max_batch_size'] = 0
        BATCH_STATS['last_inference_time'] = 0.0
        BATCH_STATS['last_total_time'] = 0.0
        
        print(f"\n[{time.strftime('%H:%M:%S')}] 统计数据已清零")
        
        self.send_response(200)
        self.send_header('Content-type', 'application/json')
        self.end_headers()
        
        response = {
            'success': True,
            'message': '统计数据已清零'
        }
        self.wfile.write(json.dumps(response).encode('utf-8'))
    
    def handle_inference(self):
        """处理推理请求"""
        global MODEL, TRACKER_MANAGER, BATCH_PROCESSOR
        
        start_time = time.time()
        image_url = ''
        task_id = 'unknown'
        
        try:
            content_length = int(self.headers['Content-Length'])
            post_data = self.rfile.read(content_length)
            request_data = json.loads(post_data.decode('utf-8'))
            
            image_url = request_data.get('image_url', '')
            task_id = request_data.get('task_id', 'unknown')
            
            if not image_url:
                raise ValueError("缺少image_url参数")
            
            # 加载算法配置文件
            algo_config = load_algo_config(image_url)
            if not algo_config:
                raise ValueError("绊线人数统计必须有配置文件")
            
            # 下载图片
            temp_image_path = f'/tmp/inference_{int(time.time()*1000)}.jpg'
            try:
                urllib.request.urlretrieve(image_url, temp_image_path)
            except Exception as e:
                raise ValueError(f"下载图片失败: {str(e)}")
            
            # 读取图片
            image = cv2.imread(temp_image_path)
            if image is None:
                raise ValueError("无法读取图片")
            
            # 清理临时文件
            try:
                os.remove(temp_image_path)
            except:
                pass
            
            print(f"\n" + "="*60)
            print(f"收到推理请求 [{time.strftime('%H:%M:%S')}]")
            print(f"  任务ID: {task_id}")
            print(f"  任务类型: 绊线人数统计")
            print(f"-"*60)
            
            # 批处理推理
            if CONFIG['enable_batching'] and BATCH_PROCESSOR:
                request_data['algo_config'] = algo_config
                request_id, request_obj = BATCH_PROCESSOR.submit_request(image, request_data)
                
                if not request_obj.event.wait(timeout=30.0):
                    raise Exception("推理超时")
                
                result_info = INFERENCE_RESULTS.get(request_id)
                if not result_info:
                    raise Exception("推理结果丢失")
                
                if result_info['error']:
                    raise Exception(result_info['error'])
                
                response = result_info['result']
                
                try:
                    del INFERENCE_RESULTS[request_id]
                    del INFERENCE_EVENTS[request_id]
                except:
                    pass
                
                total_time = (time.time() - start_time) * 1000
                inference_time = response.get('inference_time_ms', 0)
                print(f"  推理完成: 推理时间 {inference_time:.0f}ms, 总耗时 {total_time:.0f}ms")
                print(f"="*60)
            
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
                'task_id': task_id  # 任务ID
            }
            
            self.send_response(500)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps(error_response).encode('utf-8'))


def load_model():
    """加载YOLO模型"""
    global MODEL
    
    print(f"正在加载模型: {CONFIG['model_path']}")
    start_time = time.time()
    
    MODEL = YOLO(CONFIG['model_path'])
    
    load_time = time.time() - start_time
    print(f"✓ 模型加载成功 (耗时: {load_time:.2f}秒)")


def register_service():
    """注册到EasyDarwin"""
    url = f"{CONFIG['easydarwin_url']}/api/v1/ai_analysis/register"
    
    endpoint = f"http://{CONFIG['host']}:{CONFIG['port']}/infer"
    if CONFIG['host'] == '0.0.0.0':
        import socket
        try:
            hostname = socket.gethostname()
            local_ip = socket.gethostbyname(hostname)
            endpoint = f"http://{local_ip}:{CONFIG['port']}/infer"
        except:
            pass
    
    payload = {
        'service_id': CONFIG['service_id'],
        'name': CONFIG['name'],
        'task_types': CONFIG['task_types'],
        'endpoint': endpoint,
        'version': CONFIG['version']
    }
    
    print(f"\n正在注册到 {CONFIG['easydarwin_url']}...")
    print(f"  服务ID: {CONFIG['service_id']}")
    print(f"  服务名称: {CONFIG['name']}")
    print(f"  任务类型: {CONFIG['task_types']}")
    print(f"  推理端点: {endpoint}")
    
    try:
        response = requests.post(url, json=payload, timeout=10)
        response.raise_for_status()
        
        result = response.json()
        if result.get('ok'):
            print(f"✓ 注册成功: {result.get('service_id')}")
            return True
        else:
            print(f"✗ 注册失败: {result}")
            return False
    except requests.exceptions.ConnectionError:
        print(f"✗ 注册失败: 无法连接到 {CONFIG['easydarwin_url']}")
        return False
    except Exception as e:
        print(f"✗ 注册失败: {str(e)}")
        return False


def unregister_service():
    """注销服务"""
    url = f"{CONFIG['easydarwin_url']}/api/v1/ai_analysis/unregister/{CONFIG['service_id']}"
    
    print(f"\n正在注销服务: {CONFIG['service_id']}")
    
    try:
        response = requests.delete(url, timeout=10)
        response.raise_for_status()
        print("✓ 注销成功")
    except Exception as e:
        print(f"✗ 注销失败: {str(e)}")


def heartbeat_loop():
    """心跳循环"""
    global RUNNING
    
    url = f"{CONFIG['easydarwin_url']}/api/v1/ai_analysis/heartbeat/{CONFIG['service_id']}"
    
    print(f"心跳线程已启动（每{CONFIG['heartbeat_interval']}秒）")
    
    while RUNNING:
        time.sleep(CONFIG['heartbeat_interval'])
        
        if not RUNNING:
            break
        
        try:
            # 计算平均推理时间
            avg_inference_time = 0.0
            if BATCH_STATS['total_requests'] > 0:
                avg_inference_time = BATCH_STATS['total_inference_time'] / BATCH_STATS['total_requests']
            
            # 携带统计信息
            payload = {
                'total_requests': BATCH_STATS['total_requests'],
                'avg_inference_time_ms': round(avg_inference_time, 2),
                'last_inference_time_ms': round(BATCH_STATS['last_inference_time'], 2),  # 最近一次推理时间
                'last_total_time_ms': round(BATCH_STATS['last_total_time'], 2)  # 最近一次总耗时
            }
            
            response = requests.post(url, json=payload, timeout=5)
            if response.status_code == 200:
                print(f"[{time.strftime('%H:%M:%S')}] 心跳发送成功")
            else:
                print(f"[{time.strftime('%H:%M:%S')}] 心跳发送失败: {response.status_code}")
        except Exception as e:
            print(f"[{time.strftime('%H:%M:%S')}] 心跳发送失败: {str(e)}")


def signal_handler(sig, frame):
    """信号处理器（优雅退出）"""
    global RUNNING
    
    print("\n\n收到退出信号，正在关闭服务...")
    RUNNING = False


def main():
    """主函数"""
    global RUNNING, HEARTBEAT_THREAD, TRACKER_MANAGER, BATCH_PROCESSOR, BATCH_PROCESSOR_THREAD
    
    parser = argparse.ArgumentParser(description='YOLOv11x绊线人数统计算法服务')
    parser.add_argument('--service-id', default='yolo11x_line_crossing',
                        help='服务ID (默认: yolo11x_line_crossing)')
    parser.add_argument('--name', default='YOLOv11x绊线人数统计算法',
                        help='服务名称')
    parser.add_argument('--port', type=int, default=7903,
                        help='监听端口 (默认: 7903)')
    parser.add_argument('--host', default='0.0.0.0',
                        help='监听地址 (默认: 0.0.0.0)')
    parser.add_argument('--easydarwin', default='http://localhost:5066',
                        help='EasyDarwin地址')
    parser.add_argument('--model', default='/cv_space/NWPU-Crowd/runs/exp_yolo11x3/weights/best.pt',
                        help='模型路径')
    parser.add_argument('--gpu-id', type=str, default='3',
                        help='GPU设备ID (默认: 3, 可设置多个如 "0,1")')
    parser.add_argument('--no-register', action='store_true',
                        help='不注册到EasyDarwin')
    parser.add_argument('--batch-size', type=int, default=8,
                        help='批处理大小 (默认: 8)')
    parser.add_argument('--batch-timeout', type=float, default=0.1,
                        help='批处理超时（秒）')
    parser.add_argument('--no-batching', action='store_true',
                        help='禁用批处理')
    
    args = parser.parse_args()
    
    # 设置GPU设备
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    print(f"设置GPU设备: CUDA_VISIBLE_DEVICES={args.gpu_id}")
    
    # 更新配置
    CONFIG['service_id'] = args.service_id
    CONFIG['name'] = args.name
    CONFIG['port'] = args.port
    CONFIG['host'] = args.host
    CONFIG['easydarwin_url'] = args.easydarwin
    CONFIG['model_path'] = args.model
    CONFIG['batch_size'] = args.batch_size
    CONFIG['batch_timeout'] = args.batch_timeout
    CONFIG['enable_batching'] = not args.no_batching
    
    # 注册信号处理器
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    print("=" * 60)
    print(f"  {CONFIG['name']} v{CONFIG['version']}")
    print("=" * 60)
    
    # 加载模型
    load_model()
    
    # 初始化跟踪器管理器
    TRACKER_MANAGER = TrackerManager(iou_threshold=0.3, max_age=30)
    print("✓ 跟踪器管理器已初始化")
    
    # 初始化批处理器
    if CONFIG['enable_batching']:
        BATCH_PROCESSOR = BatchInferenceProcessor(
            MODEL, 
            batch_size=CONFIG['batch_size'],
            batch_timeout=CONFIG['batch_timeout']
        )
        BATCH_PROCESSOR_THREAD = threading.Thread(target=BATCH_PROCESSOR.process_loop, daemon=True)
        BATCH_PROCESSOR_THREAD.start()
        print(f"✓ 批处理推理已启动")
    else:
        print("⚠️  批处理已禁用")
    
    # 注册到EasyDarwin
    registered = False
    if not args.no_register:
        registered = register_service()
        
        if registered:
            HEARTBEAT_THREAD = threading.Thread(target=heartbeat_loop, daemon=True)
            HEARTBEAT_THREAD.start()
    else:
        print("\n⚠ 跳过注册到EasyDarwin")
    
    # 启动HTTP服务器
    server_address = (CONFIG['host'], CONFIG['port'])
    httpd = ThreadingHTTPServer(server_address, YOLOInferenceHandler)
    print(f"✓ 使用多线程HTTP服务器")
    
    print(f"\n绊线人数统计算法服务已启动")
    print(f"  服务ID: {CONFIG['service_id']}")
    print(f"  服务名称: {CONFIG['name']}")
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
        if registered:
            unregister_service()
        
        print("\n服务已关闭")
        sys.exit(0)


if __name__ == '__main__':
    main()

