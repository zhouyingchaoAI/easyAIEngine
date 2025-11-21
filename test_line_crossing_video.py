#!/usr/bin/env python3
"""
绊线算法视频测试Demo
读取mp4视频，进行目标跟踪，绘制轨迹并保存
"""
import os
import sys
import cv2
import numpy as np
import argparse
from pathlib import Path
import time

# 导入ACL推理函数
from predict import init_acl_resource, load_om_model, om_infer, release_acl_resource

# 尝试导入类别名称（若文件不存在则退回默认）
try:
    from algorithm_service_line_crossing import CLASS_NAMES
except ImportError:
    CLASS_NAMES = ['head']
    print("警告: 无法从算法脚本导入 CLASS_NAMES，使用默认 ['head']")


class KalmanFilter2D:
    """简单的常速度Kalman滤波器，仅估计中心点位置"""

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
        self.Q = np.eye(4, dtype=float) * 0.01
        self.R = np.eye(2, dtype=float) * 5.0
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


class KalmanObjectTracker:
    """带卡尔曼滤波的目标跟踪器"""

    def __init__(self, track_id, bbox, confidence, class_name):
        self.track_id = track_id
        self.bbox = bbox  # [x1, y1, x2, y2]
        self.confidence = confidence
        self.class_name = class_name
        self.center_history = []
        self.last_update = time.time()
        self.missed = 0

        cx, cy = self.get_center(bbox)
        self.width = bbox[2] - bbox[0]
        self.height = bbox[3] - bbox[1]
        self.kf = KalmanFilter2D()
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
        if len(self.center_history) > 30:
            self.center_history.pop(0)
        self.missed += 1
        return self.bbox

    def update(self, bbox, confidence):
        cx, cy = self.get_center(bbox)
        self.kf.update(cx, cy)
        self.width = 0.9 * self.width + 0.1 * (bbox[2] - bbox[0])
        self.height = 0.9 * self.height + 0.1 * (bbox[3] - bbox[1])
        cx_smoothed, cy_smoothed = self.kf.state[0, 0], self.kf.state[1, 0]
        self._update_bbox_from_center(cx_smoothed, cy_smoothed)
        self.confidence = confidence
        self.last_update = time.time()
        self.missed = 0

    def _update_bbox_from_center(self, cx, cy):
        half_w = max(self.width / 2.0, 1.0)
        half_h = max(self.height / 2.0, 1.0)
        self.bbox = [
            cx - half_w,
            cy - half_h,
            cx + half_w,
            cy + half_h
        ]


class KalmanTrackerManager:
    """使用IOU匹配 + 卡尔曼滤波的多目标跟踪管理器"""

    def __init__(self, iou_threshold=0.25, max_missed=15,
                 center_distance=80, frame_step=1):
        self.iou_threshold = iou_threshold
        self.frame_step = max(1, frame_step)
        # 根据抽帧频率自动放宽允许的丢帧次数
        self.max_missed = max_missed * self.frame_step
        # 中心点距离阈值，同步按抽帧频率放大
        self.max_center_distance = center_distance * self.frame_step
        self.trackers = {}
        self.next_id = 1

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
        c1 = KalmanTrackerManager._center(bbox1)
        c2 = KalmanTrackerManager._center(bbox2)
        return np.hypot(c1[0] - c2[0], c1[1] - c2[1])

    def _predict_all(self):
        """对所有跟踪器执行一次预测，返回预测后的列表"""
        for tracker in self.trackers.values():
            tracker.predict()

        to_delete = [
            track_id for track_id, tracker in self.trackers.items()
            if tracker.missed > self.max_missed
        ]
        for track_id in to_delete:
            del self.trackers[track_id]

        return list(self.trackers.values())

    def predict_only(self):
        """仅进行预测（用于抽帧时保持轨迹连续）"""
        return self._predict_all()

    def update(self, detections):
        # 预测一步
        self._predict_all()

        matched_trackers = set()
        matched_detections = set()

        available_trackers = set(self.trackers.keys())

        # 第一阶段：基于IOU的匹配
        for det_idx, det in enumerate(detections):
            best_iou = 0.0
            best_id = None
            for track_id in available_trackers:
                tracker = self.trackers[track_id]
                iou = self._iou(det['bbox'], tracker.bbox)
                if iou > self.iou_threshold and iou > best_iou:
                    best_iou = iou
                    best_id = track_id

            if best_id is not None:
                self.trackers[best_id].update(det['bbox'], det['confidence'])
                matched_trackers.add(best_id)
                matched_detections.add(det_idx)
                available_trackers.remove(best_id)

        # 第二阶段：使用中心点距离匹配剩余检测（应对抽帧导致的IOU不足）
        for det_idx, det in enumerate(detections):
            if det_idx in matched_detections:
                continue

            best_dist = float('inf')
            best_id = None
            for track_id in available_trackers:
                tracker = self.trackers[track_id]
                dist = self._center_distance(det['bbox'], tracker.bbox)
                if dist < self.max_center_distance and dist < best_dist:
                    best_dist = dist
                    best_id = track_id

            if best_id is not None:
                self.trackers[best_id].update(det['bbox'], det['confidence'])
                matched_trackers.add(best_id)
                matched_detections.add(det_idx)
                available_trackers.remove(best_id)

        # 为未匹配的检测创建新跟踪器
        for det_idx, det in enumerate(detections):
            if det_idx in matched_detections:
                continue
            tracker = KalmanObjectTracker(
                self.next_id,
                det['bbox'],
                det['confidence'],
                det['class']
            )
            self.trackers[self.next_id] = tracker
            self.next_id += 1

        # 移除连续未匹配的跟踪器
        to_delete = [
            track_id for track_id, tracker in self.trackers.items()
            if tracker.missed > self.max_missed
        ]
        for track_id in to_delete:
            del self.trackers[track_id]

        return list(self.trackers.values())


def draw_trajectory(image, tracker, color=None):
    """在图像上绘制跟踪轨迹"""
    if color is None:
        # 根据track_id生成颜色
        np.random.seed(tracker.track_id)
        color = tuple(map(int, np.random.randint(0, 255, 3)))
    
    # 绘制当前边界框
    x1, y1, x2, y2 = map(int, tracker.bbox)
    cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
    
    # 绘制轨迹（中心点连线）
    if len(tracker.center_history) >= 2:
        points = []
        for center in tracker.center_history:
            cx, cy = map(int, center)
            points.append((cx, cy))
        
        # 绘制轨迹线
        for i in range(len(points) - 1):
            cv2.line(image, points[i], points[i + 1], color, 2)
        
        # 绘制轨迹点
        for point in points:
            cv2.circle(image, point, 3, color, -1)
    
    # 绘制track_id和置信度
    label = f"ID:{tracker.track_id} {tracker.class_name} {tracker.confidence:.2f}"
    label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    label_y = max(y1 - 5, label_size[1])
    cv2.rectangle(image, (x1, label_y - label_size[1] - 5), 
                  (x1 + label_size[0], label_y + 5), color, -1)
    cv2.putText(image, label, (x1, label_y), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    return image


def process_video(input_video, output_video, model_path, device_id=0,
                 confidence_threshold=0.5, frame_step=1):
    """处理视频：推理、跟踪、绘制轨迹"""
    
    print("=" * 60)
    print("绊线算法视频测试Demo")
    print("=" * 60)
    print(f"输入视频: {input_video}")
    print(f"输出视频: {output_video}")
    print(f"模型路径: {model_path}")
    print(f"设备ID: {device_id}")
    print(f"置信度阈值: {confidence_threshold}")
    print(f"抽帧频率: 每 {frame_step} 帧处理一次")
    print("=" * 60)
    
    # 初始化ACL
    print("\n正在初始化ACL...")
    init_acl_resource(device_id=device_id)
    print("✓ ACL初始化成功")
    
    # 加载模型
    print(f"\n正在加载模型: {model_path}")
    load_om_model(model_path)
    print("✓ 模型加载成功")
    
    # 打开视频
    cap = cv2.VideoCapture(input_video)
    if not cap.isOpened():
        raise Exception(f"无法打开视频文件: {input_video}")
    
    # 获取视频属性
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"\n视频信息:")
    print(f"  分辨率: {width}x{height}")
    print(f"  帧率: {fps} FPS")
    print(f"  总帧数: {total_frames}")

    # 初始化跟踪器（卡尔曼滤波），根据抽帧频率自动调节参数
    tracker_manager = KalmanTrackerManager(
        iou_threshold=0.25,
        max_missed=15,
        center_distance=max(width, height) * 0.1,
        frame_step=frame_step
    )
    print("✓ 卡尔曼滤波跟踪器初始化成功")
    
    # 创建视频写入器
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video, fourcc, fps, (width, height))
    
    frame_count = 0
    total_inference_time = 0
    processed_frames = 0
    
    print(f"\n开始处理视频...")
    print("=" * 60)
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            
            # 抽帧：非处理帧直接写回原图
            if frame_step <= 0:
                frame_step = 1
            skipped = False
            if (frame_count - 1) % frame_step != 0:
                skipped = True
                trackers = tracker_manager.predict_only()
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
                for tracker in trackers:
                    draw_trajectory(frame, tracker)
                out.write(frame)
                continue

            processed_frames += 1

            # 推理
            inference_start = time.time()
            boxes_out = om_infer(model_path, frame, debug=False)
            inference_time = (time.time() - inference_start) * 1000
            total_inference_time += inference_time
            
            # 解析检测结果
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
            
            # 更新跟踪器
            trackers = tracker_manager.update(detections)
            
            # 绘制轨迹
            for tracker in trackers:
                draw_trajectory(frame, tracker)
            
            # 在左上角显示统计信息
            info_text = [
                f"Frame: {frame_count}/{total_frames}",
                f"Detections: {len(detections)}",
                f"Tracks: {len(trackers)}",
                f"Inference: {inference_time:.1f}ms"
            ]
            y_offset = 20
            for text in info_text:
                cv2.putText(frame, text, (10, y_offset), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                y_offset += 25
            
            # 写入视频
            out.write(frame)
            
            # 显示进度
            if frame_count % 30 == 0:
                progress = (frame_count / total_frames) * 100
                avg_inference = (total_inference_time / processed_frames) if processed_frames else 0
                print(f"进度: {progress:.1f}% ({frame_count}/{total_frames}) | "
                      f"平均推理时间: {avg_inference:.1f}ms | "
                      f"当前跟踪数: {len(trackers)}")
    
    finally:
        # 清理资源
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
    
    args = parser.parse_args()
    
    # 检查输入文件
    if not os.path.exists(args.input):
        print(f"错误: 输入视频文件不存在: {args.input}")
        sys.exit(1)
    
    # 检查模型文件
    model_path = args.model
    if not os.path.isabs(model_path):
        model_path = os.path.abspath(model_path)
    
    if not os.path.exists(model_path):
        print(f"错误: 模型文件不存在: {model_path}")
        sys.exit(1)
    
    # 创建输出目录
    output_dir = os.path.dirname(args.output)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    
    # 处理视频
    try:
        process_video(
            input_video=args.input,
            output_video=args.output,
            model_path=model_path,
            device_id=args.device_id,
            confidence_threshold=args.confidence,
            frame_step=args.frame_step
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

