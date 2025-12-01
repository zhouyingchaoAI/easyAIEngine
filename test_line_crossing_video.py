#!/usr/bin/env python3
"""
绊线算法视频测试Demo
读取mp4视频，进行目标跟踪，绘制轨迹并保存
支持绊线配置绘制、跟踪轨迹、穿越变色等功能
"""
import os
import sys
import cv2
import numpy as np
import argparse
from pathlib import Path
import time
import json

# 导入ACL推理函数
from predict import init_acl_resource, load_om_model, om_infer, release_acl_resource

# 尝试导入类别名称（若文件不存在则退回默认）
try:
    from algorithm_service_line_crossing import CLASS_NAMES
except ImportError:
    CLASS_NAMES = ['head']
    print("警告: 无法从算法脚本导入 CLASS_NAMES，使用默认 ['head']")

# =======================
# 更强健的匈牙利匹配+改进卡尔曼IOU多目标跟踪
# =======================

from scipy.optimize import linear_sum_assignment

class ImprovedKalmanFilter2D:
    """更稳定的常速度Kalman滤波器，调整了过程噪声和观测噪声"""

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
        # 调大Q和R，过滤更平滑，增加自恢复能力
        self.Q = np.diag([0.2, 0.2, 2., 2.])
        self.R = np.eye(2, dtype=float) * 1.5
        self.P = np.eye(4, dtype=float) * 10.0
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
    """目标跟踪条目，带更平稳尺寸补偿与关联判定"""

    def __init__(self, track_id, bbox, confidence, class_name):
        self.track_id = track_id
        self.bbox = bbox  # [x1, y1, x2, y2]
        self.confidence = confidence
        self.class_name = class_name
        self.center_history = []
        self.last_update = time.time()
        self.missed = 0
        self.crossed_lines = set()
        self.is_crossed = False
        self.hits = 1

        cx, cy = self.get_center(bbox)
        self.width = bbox[2] - bbox[0]
        self.height = bbox[3] - bbox[1]
        self.kf = ImprovedKalmanFilter2D()
        self.kf.initialize(cx, cy)
        self.center_history.append((cx, cy))

    @staticmethod
    def get_center(bbox):
        x1, y1, x2, y2 = bbox
        return (x1 + x2) / 2.0, (y1 + y2) / 2.0

    def predict(self):
        cx, cy = self.kf.predict()
        self._update_bbox_from_center(cx, cy)
        self.center_history.append((cx, cy))
        if len(self.center_history) > 50:
            self.center_history.pop(0)
        self.missed += 1
        return self.bbox

    def update(self, bbox, confidence):
        cx, cy = self.get_center(bbox)
        self.kf.update(cx, cy)
        self.width = 0.75 * self.width + 0.25 * (bbox[2] - bbox[0])
        self.height = 0.75 * self.height + 0.25 * (bbox[3] - bbox[1])
        cx_smoothed, cy_smoothed = self.kf.state[0, 0], self.kf.state[1, 0]
        self._update_bbox_from_center(cx_smoothed, cy_smoothed)
        self.confidence = confidence
        self.last_update = time.time()
        self.missed = 0
        self.hits += 1

    def _update_bbox_from_center(self, cx, cy):
        half_w = max(self.width / 2.0, 1.0)
        half_h = max(self.height / 2.0, 1.0)
        self.bbox = [
            cx - half_w,
            cy - half_h,
            cx + half_w,
            cy + half_h
        ]

class StrongSortManager:
    """
    稳定性更强的目标跟踪管理器：匈牙利联合IOU+中心距离、老ID激活优先，减少抖动
    """
    def __init__(self, iou_threshold=0.18, max_missed=25, center_distance=100, frame_step=1):
        self.iou_threshold = iou_threshold
        self.max_missed = max_missed * max(1, frame_step)
        self.max_center_distance = center_distance * max(1, frame_step)
        self.frame_step = frame_step
        self.trackers = {}
        self.next_id = 1
        self.line_crossing_counts = {}

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
    def _center(bbox):
        return ((bbox[0] + bbox[2]) / 2.0, (bbox[1] + bbox[3]) / 2.0)

    @staticmethod
    def _center_distance(bbox1, bbox2):
        c1 = StrongSortManager._center(bbox1)
        c2 = StrongSortManager._center(bbox2)
        return np.hypot(c1[0] - c2[0], c1[1] - c2[1])

    def _predict_all(self):
        """对所有trackers预测一帧"""
        for tracker in self.trackers.values():
            tracker.predict()
        to_delete = [
            tid for tid, tracker in self.trackers.items()
            if tracker.missed > self.max_missed
        ]
        for tid in to_delete:
            del self.trackers[tid]
        return list(self.trackers.values())

    def predict_only(self):
        return self._predict_all()

    def update(self, detections):
        self._predict_all()
        trackers_list = list(self.trackers.items())
        n_trk = len(trackers_list)
        n_det = len(detections)
        if n_trk == 0:
            # 全新检测直接建立
            for det in detections:
                tracker = ImprovedObjectTracker(
                    self.next_id, det['bbox'], det['confidence'], det['class']
                )
                self.trackers[self.next_id] = tracker
                self.next_id += 1
            return list(self.trackers.values())

        # 关联代价矩阵(混合IOU与中心距离)
        cost_matrix = np.zeros((n_trk, n_det), dtype=np.float32)
        for i, (tid, tracker) in enumerate(trackers_list):
            for j, det in enumerate(detections):
                iou = self._iou(det['bbox'], tracker.bbox)
                center_dist = self._center_distance(det['bbox'], tracker.bbox)
                # 小的距离 & 高IOU代价更小；iout系数大于距离系数
                cost = 1 - iou + (min(center_dist, self.max_center_distance) / (self.max_center_distance + 1)) * 0.60
                if iou < self.iou_threshold:
                    cost += 2.0  # iou明显不足极大惩罚
                cost_matrix[i, j] = cost

        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        assigned_tracks = set()
        assigned_dets = set()
        # 匹配检测-轨迹
        for i, j in zip(row_ind, col_ind):
            if cost_matrix[i, j] > 2.5:  # 代价太高的不采纳
                continue
            tid, tracker = trackers_list[i]
            det = detections[j]
            tracker.update(det['bbox'], det['confidence'])
            assigned_tracks.add(tid)
            assigned_dets.add(j)

        # 没有被分配检测的轨迹，missed+1（已在_predict_all中做）
        # 新增剩余未匹配检测为新track
        for det_idx, det in enumerate(detections):
            if det_idx not in assigned_dets:
                tracker = ImprovedObjectTracker(
                    self.next_id, det['bbox'], det['confidence'], det['class']
                )
                self.trackers[self.next_id] = tracker
                self.next_id += 1

        # 清理长时间未更新的track
        to_delete = [tid for tid, tracker in self.trackers.items() if tracker.missed > self.max_missed]
        for tid in to_delete:
            del self.trackers[tid]
        return list(self.trackers.values())

    def get_trajectory(self, tracker):
        if len(tracker.center_history) >= 2:
            return tracker.center_history[-2], tracker.center_history[-1]
        return None

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
        elif cp_start < 0 and cp_end > 0:
            return 'out'
        return 'unknown'

    def check_line_crossing(self, line_regions, image_size=None):
        if not line_regions:
            return {}
        crossing_results = {}
        width, height = image_size if image_size else (1920, 1080)
        for region in line_regions:
            if not region.get('enabled', True):
                continue
            if region.get('type') != 'line':
                continue
            region_id = region.get('id', 'line_unknown')
            points = region.get('points', [])
            direction = region.get('properties', {}).get('direction', 'both')
            if len(points) < 2:
                continue
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
            if region_id not in self.line_crossing_counts:
                self.line_crossing_counts[region_id] = 0
            for tracker in self.trackers.values():
                trajectory = self.get_trajectory(tracker)
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
                        cross_key = f"{region_id}_{tracker.track_id}"
                        if cross_key not in tracker.crossed_lines:
                            self.line_crossing_counts[region_id] += 1
                            tracker.crossed_lines.add(cross_key)
                            tracker.is_crossed = True
                            print(f"    [绊线穿越] ID:{tracker.track_id} 跨线 {region_id} ({cross_direction}) -> 累计: {self.line_crossing_counts[region_id]}")
            crossing_results[region_id] = {
                'region_name': region.get('name', region_id),
                'count': self.line_crossing_counts[region_id],
                'direction': direction,
                'points': [p1, p2]
            }
        return crossing_results

def draw_trajectory(image, tracker, color=None, is_crossed=False):
    """在图像上绘制跟踪轨迹，穿越后变色"""
    if color is None:
        np.random.seed(tracker.track_id)
        color = tuple(map(int, np.random.randint(0, 255, 3)))
    if tracker.is_crossed or is_crossed:
        color = (0, 0, 255)  # 红色
    x1, y1, x2, y2 = map(int, tracker.bbox)
    thickness = 3 if tracker.is_crossed else 2
    cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness)
    if len(tracker.center_history) >= 2:
        points = [tuple(map(int, pt)) for pt in tracker.center_history]
        line_thickness = 3 if tracker.is_crossed else 2
        for i in range(len(points) - 1):
            cv2.line(image, points[i], points[i+1], color, line_thickness)
        point_radius = 4 if tracker.is_crossed else 3
        for point in points:
            cv2.circle(image, point, point_radius, color, -1)
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
    """绘制绊线配置"""
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
        cv2.line(image, p1, p2, (0, 255, 255), 3)
        mid_x = (p1[0] + p2[0]) // 2
        mid_y = (p1[1] + p2[1]) // 2
        direction = region.get('properties', {}).get('direction', 'both')
        direction_text = {'in': '入', 'out': '出', 'both': '双向'}.get(direction, direction)
        label = f"{region_name} [{direction_text}]"
        label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(image,
                     (mid_x - label_size[0] // 2 - 5, mid_y - label_size[1] - 5),
                     (mid_x + label_size[0] // 2 + 5, mid_y + 5),
                     (0, 255, 255), -1)
        cv2.putText(image, label,
                   (mid_x - label_size[0] // 2, mid_y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
        if direction != 'both':
            dx = p2[0] - p1[0]
            dy = p2[1] - p1[1]
            length = np.sqrt(dx*dx + dy*dy)
            if length > 0:
                dx /= length
                dy /= length
                arrow_point = p2 if direction == 'out' else p1
                arrow_tip = (int(arrow_point[0] + dx * 15), int(arrow_point[1] + dy * 15))
                cv2.arrowedLine(image, arrow_point, arrow_tip, (0, 255, 255), 3, tipLength=0.3)

def load_algo_config(config_path):
    """加载算法配置文件"""
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
        print(f"✓ 成功加载配置文件: {config_path}")
        return config
    except Exception as e:
        print(f"⚠️  加载配置文件失败: {str(e)}")
        return None

def process_video(input_video, output_video, model_path, device_id=0,
                 confidence_threshold=0.5, frame_step=1, config_path=None):
    """处理视频：推理、跟踪、绘制轨迹、绘制绊线、穿越变色"""
    print("=" * 60)
    print("绊线算法视频测试Demo (稳定跟踪增强版)")
    print("=" * 60)
    print(f"输入视频: {input_video}")
    print(f"输出视频: {output_video}")
    print(f"模型路径: {model_path}")
    print(f"设备ID: {device_id}")
    print(f"置信度阈值: {confidence_threshold}")
    print(f"抽帧频率: 每 {frame_step} 帧处理一次")
    if config_path:
        print(f"配置文件: {config_path}")
    else:
        print(f"配置文件: 未指定（将不绘制绊线）")
    print("=" * 60)
    algo_config = None
    line_regions = []
    if config_path and os.path.exists(config_path):
        algo_config = load_algo_config(config_path)
        if algo_config:
            regions = algo_config.get('regions', [])
            line_regions = [r for r in regions if r.get('type') == 'line' and r.get('enabled', True)]
            print(f"✓ 加载到 {len(line_regions)} 条绊线配置")
    print("\n正在初始化ACL...")
    init_acl_resource(device_id=device_id)
    print("✓ ACL初始化成功")
    print(f"\n正在加载模型: {model_path}")
    load_om_model(model_path)
    print("✓ 模型加载成功")
    cap = cv2.VideoCapture(input_video)
    if not cap.isOpened():
        raise Exception(f"无法打开视频文件: {input_video}")
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"\n视频信息:")
    print(f"  分辨率: {width}x{height}")
    print(f"  帧率: {fps} FPS")
    print(f"  总帧数: {total_frames}")

    tracker_manager = StrongSortManager(
        iou_threshold=0.18,
        max_missed=25,
        center_distance=max(width, height) * 0.08,
        frame_step=frame_step
    )
    print("✓ 稳定增强型跟踪器初始化成功")
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video, fourcc, fps, (width, height))
    frame_count = 0
    total_inference_time = 0
    processed_frames = 0
    line_crossing_results = {}  
    print(f"\n开始处理视频...")
    print("=" * 60)
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame_count += 1
            if frame_step <= 0:
                frame_step = 1
            if (frame_count - 1) % frame_step != 0:
                trackers = tracker_manager.predict_only()
                if line_regions:
                    draw_line_config(frame, line_regions, (width, height))
                for tracker in trackers:
                    draw_trajectory(frame, tracker)
                info_text = [
                    f"Frame: {frame_count}/{total_frames}",
                    f"Skipped (frame_step={frame_step})",
                    f"Tracks (predicted): {len(trackers)}"
                ]
                y_offset = 20
                for text in info_text:
                    cv2.putText(frame, text, (10, y_offset),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 255), 2)
                    y_offset += 25
                if line_crossing_results:
                    info_text = ["Line Crossing:"]
                    for line_id, line_info in line_crossing_results.items():
                        info_text.append(f"  {line_info['region_name']}: {line_info['count']}")
                    for text in info_text:
                        color = (255, 255, 0) if text.startswith("  ") else (0, 200, 255)
                        font_scale = 0.5 if text.startswith("  ") else 0.6
                        cv2.putText(frame, text, (10, y_offset),
                                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, 2)
                        y_offset += 25
                out.write(frame)
                continue

            processed_frames += 1
            inference_start = time.time()
            boxes_out = om_infer(model_path, frame, debug=False)
            inference_time = (time.time() - inference_start) * 1000
            total_inference_time += inference_time
            detections = []
            if boxes_out is not None and len(boxes_out) > 0:
                for b in boxes_out:
                    x1, y1, x2, y2, conf, cls_id = b
                    if float(conf) < confidence_threshold:
                        continue
                    cls_id = int(cls_id)
                    class_name = CLASS_NAMES[cls_id] if 0 <= cls_id < len(CLASS_NAMES) else str(cls_id)
                    detections.append({
                        'bbox': [float(x1), float(y1), float(x2), float(y2)],
                        'confidence': float(conf),
                        'class': class_name
                    })

            trackers = tracker_manager.update(detections)

            line_crossing_results = {}
            if line_regions:
                image_size = (width, height)
                line_crossing_results = tracker_manager.check_line_crossing(line_regions, image_size)

            if line_regions:
                draw_line_config(frame, line_regions, (width, height))
            for tracker in trackers:
                draw_trajectory(frame, tracker)
            info_text = [
                f"Frame: {frame_count}/{total_frames}",
                f"Detections: {len(detections)}",
                f"Tracks: {len(trackers)}",
                f"Inference: {inference_time:.1f}ms"
            ]
            if line_crossing_results:
                info_text.append("")
                info_text.append("Line Crossing:")
                for line_id, line_info in line_crossing_results.items():
                    info_text.append(f"  {line_info['region_name']}: {line_info['count']}")
            y_offset = 20
            for text in info_text:
                if text == "":
                    y_offset += 5
                    continue
                color = (0, 255, 0) if not text.startswith("  ") else (255, 255, 0)
                font_scale = 0.6 if not text.startswith("  ") else 0.5
                cv2.putText(frame, text, (10, y_offset),
                           cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, 2)
                y_offset += 25
            out.write(frame)
            if frame_count % 30 == 0:
                progress = (frame_count / total_frames) * 100
                avg_inference = (total_inference_time / processed_frames) if processed_frames else 0
                print(f"进度: {progress:.1f}% ({frame_count}/{total_frames}) | "
                      f"平均推理时间: {avg_inference:.1f}ms | "
                      f"当前跟踪数: {len(trackers)}")
    finally:
        cap.release()
        out.release()
        release_acl_resource()
        print("\n" + "=" * 60)
        print("处理完成！")
        print(f"总帧数: {frame_count}")
        if processed_frames:
            print(f"处理帧数: {processed_frames} (抽帧频率: 每 {frame_step} 帧)")
            print(f"平均推理时间: {total_inference_time/processed_frames:.1f}ms")
        else:
            print("未处理任何帧（请检查抽帧频率设置）")
        print(f"输出视频: {output_video}")
        print("=" * 60)

def main():
    parser = argparse.ArgumentParser(description='绊线算法视频测试Demo')
    parser.add_argument('--input', '-i', required=True,
                        help='输入视频路径 (mp4)')
    parser.add_argument('--output', '-o', required=True,
                        help='输出视频路径 (mp4)')
    parser.add_argument('--model', '-m', default='./weight/best.om',
                        help='模型路径 (默认: ./weight/best.om)')
    parser.add_argument('--device-id', type=int, default=3,
                        help='Ascend NPU设备ID (默认: 0)')
    parser.add_argument('--confidence', type=float, default=0.05,
                        help='置信度阈值 (默认: 0.5)')
    parser.add_argument('--frame-step', type=int, default=1,
                        help='抽帧频率，每N帧处理一次 (默认: 1，处理全部帧)')
    parser.add_argument('--config', '-c', default=None,
                        help='绊线配置文件路径 (algo_config.json)')
    args = parser.parse_args()
    if not os.path.exists(args.input):
        print(f"错误: 输入视频文件不存在: {args.input}")
        sys.exit(1)
    model_path = args.model
    if not os.path.isabs(model_path):
        model_path = os.path.abspath(model_path)
    if not os.path.exists(model_path):
        print(f"错误: 模型文件不存在: {model_path}")
        sys.exit(1)
    output_dir = os.path.dirname(args.output)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    config_path = args.config
    if config_path and not os.path.exists(config_path):
        print(f"警告: 配置文件不存在: {config_path}，将不绘制绊线")
        config_path = None
    try:
        process_video(
            input_video=args.input,
            output_video=args.output,
            model_path=model_path,
            device_id=args.device_id,
            confidence_threshold=args.confidence,
            frame_step=args.frame_step,
            config_path=config_path
        )
    except KeyboardInterrupt:
        print("\n\n用户中断")
        release_acl_resource()
        sys.exit(0)
    except Exception as e:
        print(f"\n错误: {str(e)}")
        import traceback
        traceback.print_exc()
        release_acl_resource()
        sys.exit(1)

if __name__ == '__main__':
    main()
