#!/usr/bin/env python3
"""
YOLOv11x 人头检测算法服务（实时检测版本）
符合EasyDarwin智能分析插件规范
支持实时人数统计、客流分析、人头检测
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
from urllib.parse import urlparse
from pathlib import Path
import requests
import urllib.request
import cv2
import numpy as np
from predict import init_acl_resource, load_om_model, om_infer, release_acl_resource
import uuid

# 尝试导入 ThreadingHTTPServer，如果不存在则创建
try:
    from http.server import ThreadingHTTPServer
except ImportError:
    class ThreadingHTTPServer(socketserver.ThreadingMixIn, HTTPServer):
        daemon_threads = True

# 全局配置
CONFIG = {
    'service_id': 'head_detector',
    'name': '客流人数统计算法',
    'version': '2.1.0',
    'model_path': './weight/best.om',
    'task_types': ['人数统计', '客流分析', '人头检测'],
    'port': 7902,
    'host': '172.16.5.207',
    'easydarwin_url': '10.1.6.230:5066',
    'heartbeat_interval': 30,
}

# 全局变量
MODEL = None
# OM/ACL 相关
OM_LOADED = False
CLASS_NAMES = ['head']
RUNNING = True
HEARTBEAT_THREAD = None
REGISTER_THREAD = None
REGISTERED = False  # 注册状态标志

# 统计信息
STATS = {
    'total_requests': 0,
    'total_inference_time': 0.0,
}




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


def filter_objects_by_region(objects, regions, image_size):
    """
    根据区域过滤检测对象（支持矩形和多边形）
    objects: 检测到的对象列表
    regions: 区域配置列表
    image_size: (width, height)
    返回: 过滤后的对象列表
    """
    if not regions:
        return objects
    
    width, height = image_size
    filtered_objects = []
    
    for obj in objects:
        bbox = obj['bbox']
        # 计算物体中心点
        center_x = (bbox[0] + bbox[2]) / 2
        center_y = (bbox[1] + bbox[3]) / 2
        
        # 检查是否在任何一个区域内
        in_any_region = False
        for region in regions:
            if not region.get('enabled', True):
                continue
            
            region_type = region.get('type')
            points = region.get('points', [])
            
            if region_type == 'rectangle' and len(points) >= 2:
                # 矩形区域：points[0] 是左上角，points[1] 是右下角
                p1, p2 = points[0], points[1]
                
                # 转换坐标（如果是归一化坐标）
                if 0 <= p1[0] <= 1 and 0 <= p1[1] <= 1:
                    x1 = int(p1[0] * width)
                    y1 = int(p1[1] * height)
                    x2 = int(p2[0] * width)
                    y2 = int(p2[1] * height)
                else:
                    x1, y1 = int(p1[0]), int(p1[1])
                    x2, y2 = int(p2[0]), int(p2[1])
                
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
                    if 0 <= point[0] <= 1 and 0 <= point[1] <= 1:
                        # 归一化坐标，转换为像素坐标
                        polygon.append((int(point[0] * width), int(point[1] * height)))
                    else:
                        # 已经是像素坐标
                        polygon.append((int(point[0]), int(point[1])))
                
                # 判断中心点是否在多边形内
                if point_in_polygon((center_x, center_y), polygon):
                    in_any_region = True
                    break
        
        if in_any_region:
            filtered_objects.append(obj)
    
    return filtered_objects


class YOLOInferenceHandler(BaseHTTPRequestHandler):
    """HTTP推理请求处理器"""
    
    def log_message(self, format, *args):
        print(f"[{self.log_date_time_string()}] {format % args}")
    
    def do_POST(self):
        if self.path == '/infer':
            self.handle_inference()
        elif self.path == '/health':
            self.handle_health()
        else:
            self.send_error(404, "Not Found")
    
    def do_GET(self):
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
        
        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>{CONFIG['name']}</title>
            <meta charset="utf-8">
        </head>
        <body>
            <h1>{CONFIG['name']}</h1>
            <p><strong>服务ID:</strong> {CONFIG['service_id']}</p>
            <p><strong>版本:</strong> {CONFIG['version']}</p>
            <p><strong>支持任务类型:</strong> {', '.join(CONFIG['task_types'])}</p>
            <p><strong>推理端点:</strong> POST /infer</p>
            <p><strong>健康检查:</strong> GET /health</p>
            <hr>
            <h2>特点</h2>
            <p>✅ 实时检测 - 每次推理都返回当前检测结果</p>
            <p>✅ 无需配置文件</p>
            <p>✅ 支持批处理加速</p>
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
            'model_loaded': OM_LOADED,
            'inference_mode': 'single_thread'
        }
        self.wfile.write(json.dumps(response).encode('utf-8'))
    
    def handle_stats(self):
        """性能统计"""
        self.send_response(200)
        self.send_header('Content-type', 'application/json')
        self.end_headers()
        
        stats = {
            'inference_mode': 'single_thread',
            'statistics': dict(STATS),
            'avg_inference_time_per_request': (
                STATS['total_inference_time'] / STATS['total_requests']
                if STATS['total_requests'] > 0 else 0
            )
        }
        
        self.wfile.write(json.dumps(stats, indent=2).encode('utf-8'))
    
    def handle_inference(self):
        """处理推理请求（实时检测专用）"""
        global MODEL
        
        start_time = time.time()
        
        try:
            content_length = int(self.headers['Content-Length'])
            post_data = self.rfile.read(content_length)
            request_data = json.loads(post_data.decode('utf-8'))
            
            image_url = request_data.get('image_url')
            task_id = request_data.get('task_id', 'unknown')
            task_type = request_data.get('task_type', 'unknown')
            
            if not image_url:
                raise ValueError("缺少image_url参数")
            
            # 加载算法配置文件（用于区域过滤）
            algo_config = load_algo_config(image_url)
            
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
            
            print(f"\n{'='*60}")
            print(f"收到推理请求 [{time.strftime('%H:%M:%S')}]")
            print(f"  任务ID: {task_id}")
            print(f"  任务类型: {task_type}")
            print(f"  推理模式: 单线程直接推理")
            print(f"{'-'*60}")
            
            # 更新统计
            STATS['total_requests'] += 1
            
            # 直接推理（主线程执行，避免ACL跨线程问题）
            inference_start = time.time()
            boxes_out = om_infer(CONFIG['model_path'], image, debug=False)
            inference_time = (time.time() - inference_start) * 1000
            STATS['total_inference_time'] += inference_time
            
            # 置信度阈值
            confidence_threshold = 0.5
            if algo_config:
                algo_params = algo_config.get('algorithm_params', {})
                confidence_threshold = algo_params.get('confidence_threshold', 0.5)
            
            objects = []
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
            
            # 【区域过滤】如果配置了检测区域，只保留区域内的物体
            original_count = len(objects)
            if algo_config:
                regions = algo_config.get('regions', [])
                if regions:
                    image_size = (image.shape[1], image.shape[0])
                    objects = filter_objects_by_region(objects, regions, image_size)
                    filtered_count = original_count - len(objects)
                    if filtered_count > 0:
                        print(f"  ℹ️  区域过滤: 原始 {original_count} 个 → 区域内 {len(objects)} 个 (过滤掉 {filtered_count} 个)")
            
            person_count = len(objects)
            
            result_data = {
                'objects': objects,
                'total_count': len(objects),
            }
            
            if task_type in ['人数统计', '客流分析']:
                result_data['person_count'] = person_count
            
            avg_confidence = 0.0
            if len(objects) > 0:
                avg_confidence = sum(obj['confidence'] for obj in objects) / len(objects)
            
            response = {
                'success': True,
                'result': result_data,
                'confidence': avg_confidence,
                'inference_time_ms': int(inference_time)
            }
            
            total_time = (time.time() - start_time) * 1000
            print(f"  推理完成: {inference_time:.0f}ms, 总耗时 {total_time:.0f}ms")
            print(f"{'='*60}")
            
            # 发送响应
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps(response).encode('utf-8'))
            
        except Exception as e:
            print(f"  推理失败: {str(e)}")
            import traceback
            traceback.print_exc()
            
            error_response = {
                'success': False,
                'error': str(e),
                'confidence': 0.0,
                'inference_time_ms': 0
            }
            
            self.send_response(500)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps(error_response).encode('utf-8'))


def load_model():
    """加载 OM 模型并初始化 ACL 环境"""
    global MODEL, OM_LOADED
    
    print(f"正在初始化 ACL 并加载 OM 模型: {CONFIG['model_path']}")
    start_time = time.time()
    
    # 初始化 ACL
    init_acl_resource(device_id=CONFIG.get('device_id', 0))
    # 加载 OM
    load_om_model(CONFIG['model_path'])
    OM_LOADED = True
    
    load_time = time.time() - start_time
    print(f"✓ OM 模型加载成功 (耗时: {load_time:.2f}秒)")


def register_service(quiet=False):
    """注册到EasyDarwin"""
    global REGISTERED
    
    url = f"{CONFIG['easydarwin_url']}/api/v1/ai_analysis/register"
    
    # 优先使用手动指定的主机IP，然后是自动检测
    endpoint = f"http://{CONFIG['host']}:{CONFIG['port']}/infer"
    if CONFIG['host'] == '0.0.0.0':
        # 如果手动指定了主机IP，直接使用
        if CONFIG.get('host_ip'):
            endpoint = f"http://{CONFIG['host_ip']}:{CONFIG['port']}/infer"
        else:
            # 自动检测主机IP
            import socket
            try:
                # 尝试获取主机的外部IP地址
                # 方法1: 通过连接外部服务获取本机IP
                s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
                try:
                    # 连接到一个外部地址（不会实际发送数据）
                    s.connect(("8.8.8.8", 80))
                    local_ip = s.getsockname()[0]
                except:
                    # 方法2: 回退到hostname解析
                    hostname = socket.gethostname()
                    local_ip = socket.gethostbyname(hostname)
                finally:
                    s.close()
                
                # 如果获取到的是127.0.0.1或容器内部地址，尝试其他方法
                if local_ip.startswith('127.') or local_ip.startswith('172.17.') or local_ip.startswith('192.168.'):
                    # 尝试从环境变量获取主机IP
                    import os
                    host_ip = os.environ.get('HOST_IP') or os.environ.get('HOST_ADDR')
                    if host_ip:
                        local_ip = host_ip
                
                endpoint = f"http://{local_ip}:{CONFIG['port']}/infer"
            except:
                # 如果都失败了，使用默认的0.0.0.0
                endpoint = f"http://0.0.0.0:{CONFIG['port']}/infer"
    
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
        print(f"  任务类型: {CONFIG['task_types']}")
        print(f"  推理端点: {endpoint}")
    
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
            response = requests.post(url, timeout=5)
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
    global RUNNING, HEARTBEAT_THREAD
    
    parser = argparse.ArgumentParser(description='YOLOv11x人头检测算法服务（实时检测）')
    parser.add_argument('--service-id', default='yolo11x_head_detector',
                        help='服务ID')
    parser.add_argument('--name', default='YOLOv11x人头检测算法',
                        help='服务名称')
    parser.add_argument('--task-types', nargs='+', default=['人数统计', '客流分析', '人头检测'],
                        help='支持的任务类型')
    parser.add_argument('--port', type=int, default=7901,
                        help='监听端口 (默认: 7901)')
    parser.add_argument('--host', default='0.0.0.0',
                        help='监听地址')
    parser.add_argument('--easydarwin', default='10.1.6.230:5066',
                        help='EasyDarwin地址')
    parser.add_argument('--model', default='./weight/best.om',
                        help='OM模型路径 (.om)')
    parser.add_argument('--device-id', type=int, default=0,
                        help='Ascend 设备ID (默认: 0)')
    parser.add_argument('--host-ip', type=str, default=None,
                        help='主机IP地址 (用于注册到EasyDarwin，默认自动检测)')
    parser.add_argument('--no-register', action='store_true',
                        help='不注册到EasyDarwin')
    
    args = parser.parse_args()
    
    # Ascend 设备信息
    print(f"使用 Ascend NPU 设备: device_id={args.device_id}")
    
    # 更新配置
    CONFIG['service_id'] = args.service_id
    CONFIG['name'] = args.name
    CONFIG['task_types'] = args.task_types
    CONFIG['port'] = args.port
    CONFIG['host'] = args.host
    CONFIG['device_id'] = args.device_id
    CONFIG['host_ip'] = args.host_ip  # 添加主机IP配置
    CONFIG['easydarwin_url'] = args.easydarwin
    # 规范化 EasyDarwin 基地址，确保包含协议前缀
    if not (CONFIG['easydarwin_url'].startswith('http://') or CONFIG['easydarwin_url'].startswith('https://')):
        CONFIG['easydarwin_url'] = f"http://{CONFIG['easydarwin_url']}"
    CONFIG['model_path'] = args.model
    
    # 注册信号处理器
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    print("=" * 60)
    print(f"  {CONFIG['name']} v{CONFIG['version']}")
    print(f"  实时检测服务（无追踪器）")
    print("=" * 60)
    
    # 加载模型
    load_model()
    
    print("✓ 单线程推理模式已启用")
    
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
    
    # 启动HTTP服务器
    server_address = (CONFIG['host'], CONFIG['port'])
    httpd = HTTPServer(server_address, YOLOInferenceHandler)
    
    print(f"\n✓ 实时检测算法服务已启动")
    print(f"  服务ID: {CONFIG['service_id']}")
    print(f"  服务名称: {CONFIG['name']}")
    print(f"  支持类型: {CONFIG['task_types']}")
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
        # 释放 ACL 资源
        try:
            if OM_LOADED:
                release_acl_resource()
        except Exception as e:
            print(f"释放ACL资源失败: {e}")
        
        print("\n服务已关闭")
        sys.exit(0)


if __name__ == '__main__':
    main()
