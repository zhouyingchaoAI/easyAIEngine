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
import atexit
import tempfile
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
    'name': '人数统计算法',
    'version': '2.1.0',
    'model_path': './weight/best.om',
    'task_types': ['人数统计'],
    'port': 7902,
    'host': '172.16.5.207',
    'easydarwin_url': '127.0.0.1:5066',
    'heartbeat_interval': 30,
    'log_dir': './logs',  # 默认使用相对路径，可通过命令行参数覆盖
    'log_file': 'realtime_detector.log',
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
LOG_FILE_HANDLE = None

ORIGINAL_STDOUT = sys.stdout
ORIGINAL_STDERR = sys.stderr

# 统计信息
STATS = {
    'total_requests': 0,
    'total_inference_time': 0.0,
    'last_inference_time': 0.0,  # 最近一次推理时间（ms）
    'last_total_time': 0.0,      # 最近一次总耗时（ms）
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
                    # 使用相对于工作目录的configs目录
                    # 在打包后的环境中，使用当前工作目录
                    config_dir = Path("configs").resolve()
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
# 日志重定向
class TeeStream:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for stream in self.streams:
            try:
                stream.write(data)
            except Exception:
                continue
        self.flush()
        return len(data)

    def flush(self):
        for stream in self.streams:
            try:
                stream.flush()
            except Exception:
                continue


def setup_logging(log_dir, log_file):
    """将stdout/stderr同时输出到控制台和日志文件"""
    global LOG_FILE_HANDLE, ORIGINAL_STDOUT, ORIGINAL_STDERR

    if LOG_FILE_HANDLE is not None:
        return

    candidates = []
    if log_dir:
        candidates.append(Path(log_dir))

    project_log_dir = Path(__file__).resolve().parent / 'logs'
    if not candidates or project_log_dir not in candidates:
        candidates.append(project_log_dir)

    for directory in candidates:
        try:
            directory.mkdir(parents=True, exist_ok=True)
            log_path = directory / log_file
            LOG_FILE_HANDLE = open(log_path, 'a', encoding='utf-8')
            sys.stdout = TeeStream(ORIGINAL_STDOUT, LOG_FILE_HANDLE)
            sys.stderr = TeeStream(ORIGINAL_STDERR, LOG_FILE_HANDLE)
            sys.stdout.write(f"日志输出重定向到: {log_path}\n")
            sys.stdout.flush()
            return
        except Exception as err:
            ORIGINAL_STDERR.write(f"⚠️ 无法初始化日志文件 {directory / log_file}: {err}\n")
            ORIGINAL_STDERR.flush()

    LOG_FILE_HANDLE = None


def close_log_file():
    global LOG_FILE_HANDLE
    if LOG_FILE_HANDLE:
        try:
            sys.stdout = ORIGINAL_STDOUT
            sys.stderr = ORIGINAL_STDERR
            LOG_FILE_HANDLE.close()
        except Exception:
            pass
        LOG_FILE_HANDLE = None


atexit.register(close_log_file)


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
    regions_or_config: 区域配置列表或完整算法配置（含regions、coordinate_type等）
    image_size: (width, height)
    返回:       过滤后的对象列表
    """
    if not regions_or_config:
        return objects
    
    # 兼容旧调用：既支持直接传regions列表，也支持传完整algo_config
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
        """
        将区域定义的坐标转换为图像像素坐标，支持以下模式：
          - normalized: 归一化到[0,1]，按图像尺寸缩放
          - canvas:     基于画布像素，需要结合canvas_size与实际图像尺寸缩放
          - pixel/其它: 直接认为是像素坐标
        未显式声明时，如果坐标落在[0,1]之间则视为归一化，否则默认像素。
        """
        if point is None or len(point) < 2:
            return None
        
        x, y = point[0], point[1]
        coord_type = (coordinate_type_override or '').lower()
        if not coord_type:
            coord_type = default_coordinate_type
        
        if coord_type in ('normalized', 'relative'):
            return x * width, y * height
        
        if coord_type in ('canvas', 'design', 'ui'):
            # 先将画布坐标转换为图像像素
            if canvas_width and canvas_height:
                scale_x = width / canvas_width
                scale_y = height / canvas_height
                return x * scale_x, y * scale_y
            # 缺省画布尺寸时退化为像素
            return x, y
        
        if coord_type in ('pixel', 'pixels', 'absolute'):
            return x, y
        
        # 自动判断：坐标在0~1之间视为归一化，否则视为像素
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
            
            # 区域阈值：若有配置且检测置信度不足，则该区域认为不命中
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
                
                # 确保 x1 <= x2, y1 <= y2（处理边界情况）
                x1, x2 = min(x1, x2), max(x1, x2)
                y1, y2 = min(y1, y2), max(y1, y2)
                
                # 判断中心点是否在矩形内（包括边界）
                # 注意：必须同时满足 x 和 y 都在范围内，才能认为在矩形内
                if (x1 <= center_x <= x2) and (y1 <= center_y <= y2):
                    in_any_region = True
                    break
                    
            elif region_type == 'polygon' and len(points) >= 3:
                # 多边形区域
                polygon = []
                for point in points:
                    converted = convert_point(point, region_coord_type)
                    if converted is not None:
                        polygon.append(tuple(converted))
                
                # 多边形至少需要三个不同点
                if len(polygon) < 3:
                    continue
                
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
        elif self.path == '/reset_stats':
            self.handle_reset_stats()
        elif self.path == '/config':
            self.handle_config_post()
        else:
            self.send_error(404, "Not Found")
    
    def do_GET(self):
        if self.path == '/health':
            self.handle_health()
        elif self.path == '/':
            self.handle_index()
        elif self.path == '/stats':
            self.handle_stats()
        elif self.path == '/config':
            self.handle_config_get()
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
                    border-bottom: 3px solid #4CAF50;
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
                    border-left: 4px solid #4CAF50;
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
                    background: #e8f5e9;
                    border-radius: 5px;
                }}
                .stats-section h2 {{
                    margin-top: 0;
                    color: #2e7d32;
                }}
                .stat-value {{
                    font-size: 32px;
                    font-weight: bold;
                    color: #1b5e20;
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
                .message.error {{
                    background: #f44336;
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
                .config-section {{
                    margin: 30px 0;
                    padding: 20px;
                    background: #e3f2fd;
                    border-radius: 5px;
                    border-left: 4px solid #2196F3;
                }}
                .config-section h2 {{
                    margin-top: 0;
                    color: #1565c0;
                }}
                .form-group {{
                    margin: 15px 0;
                }}
                .form-group label {{
                    display: block;
                    margin-bottom: 5px;
                    color: #333;
                    font-weight: bold;
                }}
                .form-group input {{
                    width: 100%;
                    padding: 10px;
                    border: 1px solid #ddd;
                    border-radius: 5px;
                    font-size: 14px;
                    box-sizing: border-box;
                }}
                .form-group input:focus {{
                    outline: none;
                    border-color: #2196F3;
                    box-shadow: 0 0 5px rgba(33, 150, 243, 0.3);
                }}
                .btn-primary {{
                    background: #2196F3;
                    color: white;
                    border: none;
                    padding: 12px 30px;
                    font-size: 16px;
                    border-radius: 5px;
                    cursor: pointer;
                    transition: background 0.3s;
                }}
                .btn-primary:hover {{
                    background: #1976D2;
                }}
                .status-badge {{
                    display: inline-block;
                    padding: 5px 10px;
                    border-radius: 15px;
                    font-size: 12px;
                    font-weight: bold;
                    margin-left: 10px;
                }}
                .status-badge.registered {{
                    background: #4CAF50;
                    color: white;
                }}
                .status-badge.unregistered {{
                    background: #f44336;
                    color: white;
                }}
            </style>
        </head>
        <body>
            <div class="container">
                <h1>🎯 {CONFIG['name']}</h1>
                
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

                <div class="config-section">
                    <h2>⚙️ 服务配置</h2>
                    <form id="configForm" onsubmit="updateConfig(event)">
                        <div class="form-group">
                            <label for="easydarwin_url">EasyDarwin地址:</label>
                            <input type="text" id="easydarwin_url" name="easydarwin_url" 
                                   placeholder="127.0.0.1:5066 或 http://127.0.0.1:5066" required>
                        </div>
                        <div class="form-group">
                            <label for="host_ip">主机IP地址 (可选):</label>
                            <input type="text" id="host_ip" name="host_ip" 
                                   placeholder="留空则默认使用 127.0.0.1">
                        </div>
                        <div class="form-group">
                            <label>注册状态:</label>
                            <span id="register-status" class="status-badge unregistered">未注册</span>
                        </div>
                        <button type="submit" class="btn-primary">💾 保存配置</button>
                        <div id="config-message" class="message"></div>
                    </form>
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
                    <p><strong>配置管理:</strong> <code>GET /config</code> | <code>POST /config</code></p>
                    <p><strong>清零统计:</strong> <code>POST /reset_stats</code></p>
                </div>
            </div>

            <script>
                let configRefreshInterval = null;
                let isEditingEasydarwin = false;
                let isEditingHostIp = false;
                
                // 加载配置
                function loadConfig() {{
                    // 如果用户正在编辑，不刷新输入框的值
                    if (isEditingEasydarwin || isEditingHostIp) {{
                        return;
                    }}
                    
                    fetch('/config')
                        .then(res => res.json())
                        .then(data => {{
                            // 显示时去掉 http:// 或 https:// 前缀，让用户看到更简洁的格式
                            let easydarwinUrl = data.easydarwin_url || '';
                            if (easydarwinUrl.startsWith('http://')) {{
                                easydarwinUrl = easydarwinUrl.substring(7);
                            }} else if (easydarwinUrl.startsWith('https://')) {{
                                easydarwinUrl = easydarwinUrl.substring(8);
                            }}
                            
                            if (!isEditingEasydarwin) {{
                                document.getElementById('easydarwin_url').value = easydarwinUrl;
                            }}
                            if (!isEditingHostIp) {{
                                document.getElementById('host_ip').value = data.host_ip || '';
                            }}
                            updateRegisterStatus(data.registered);
                        }})
                        .catch(err => {{
                            console.error('加载配置失败:', err);
                        }});
                }}

                // 更新配置
                function updateConfig(event) {{
                    event.preventDefault();
                    
                    const easydarwinUrl = document.getElementById('easydarwin_url').value.trim();
                    const hostIp = document.getElementById('host_ip').value.trim();
                    
                    if (!easydarwinUrl) {{
                        showConfigMessage('EasyDarwin地址不能为空', 'error');
                        return;
                    }}
                    
                    const payload = {{
                        easydarwin_url: easydarwinUrl,
                        host_ip: hostIp || null
                    }};
                    
                    fetch('/config', {{
                        method: 'POST',
                        headers: {{
                            'Content-Type': 'application/json'
                        }},
                        body: JSON.stringify(payload)
                    }})
                    .then(res => res.json())
                    .then(data => {{
                        if (data.success) {{
                            showConfigMessage('配置已保存并重新注册服务', 'success');
                            updateRegisterStatus(data.config.registered);
                            // 更新输入框显示值（去掉 http:// 前缀）
                            let easydarwinUrl = data.config.easydarwin_url || '';
                            if (easydarwinUrl.startsWith('http://')) {{
                                easydarwinUrl = easydarwinUrl.substring(7);
                            }} else if (easydarwinUrl.startsWith('https://')) {{
                                easydarwinUrl = easydarwinUrl.substring(8);
                            }}
                            document.getElementById('easydarwin_url').value = easydarwinUrl;
                            document.getElementById('host_ip').value = data.config.host_ip || '';
                            // 延迟刷新配置以确保状态同步（但不更新输入框，因为已经更新了）
                            setTimeout(function() {{
                                updateRegisterStatus(data.config.registered);
                            }}, 1000);
                        }} else {{
                            showConfigMessage('保存失败: ' + (data.message || '未知错误'), 'error');
                        }}
                    }})
                    .catch(err => {{
                        console.error('更新配置失败:', err);
                        showConfigMessage('更新配置失败: ' + err, 'error');
                    }});
                }}

                // 更新注册状态显示
                function updateRegisterStatus(registered) {{
                    const statusBadge = document.getElementById('register-status');
                    if (registered) {{
                        statusBadge.textContent = '已注册';
                        statusBadge.className = 'status-badge registered';
                    }} else {{
                        statusBadge.textContent = '未注册';
                        statusBadge.className = 'status-badge unregistered';
                    }}
                }}

                // 显示配置消息
                function showConfigMessage(msg, type) {{
                    const msgDiv = document.getElementById('config-message');
                    msgDiv.textContent = msg;
                    msgDiv.className = 'message ' + (type === 'success' ? 'success' : 'error');
                    msgDiv.style.display = 'block';
                    setTimeout(() => {{
                        msgDiv.style.display = 'none';
                    }}, 5000);
                }}

                // 加载统计数据
                function loadStats() {{
                    fetch('/stats')
                        .then(res => res.json())
                        .then(data => {{
                            const totalRequests = data.statistics.total_requests || 0;
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

                // 初始加载和定时刷新
                loadConfig();
                loadStats();
                setInterval(loadStats, 3000);  // 每3秒刷新一次
                setInterval(loadConfig, 5000);  // 每5秒刷新配置状态（仅在未编辑时）
                
                // 监听输入框焦点事件，防止编辑时被刷新覆盖
                // 使用 setTimeout 确保 DOM 元素已经加载
                setTimeout(function() {{
                    const easydarwinInput = document.getElementById('easydarwin_url');
                    const hostIpInput = document.getElementById('host_ip');
                    
                    if (easydarwinInput) {{
                        easydarwinInput.addEventListener('focus', function() {{
                            isEditingEasydarwin = true;
                        }});
                        easydarwinInput.addEventListener('blur', function() {{
                            isEditingEasydarwin = false;
                            // 失去焦点后立即刷新一次
                            setTimeout(loadConfig, 100);
                        }});
                    }}
                    
                    if (hostIpInput) {{
                        hostIpInput.addEventListener('focus', function() {{
                            isEditingHostIp = true;
                        }});
                        hostIpInput.addEventListener('blur', function() {{
                            isEditingHostIp = false;
                            // 失去焦点后立即刷新一次
                            setTimeout(loadConfig, 100);
                        }});
                    }}
                }}, 100);
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
    
    def handle_config_get(self):
        """获取当前配置"""
        self.send_response(200)
        self.send_header('Content-type', 'application/json')
        self.end_headers()
        
        response = {
            'easydarwin_url': CONFIG.get('easydarwin_url', ''),
            'service_id': CONFIG.get('service_id', ''),
            'name': CONFIG.get('name', ''),
            'port': CONFIG.get('port', 0),
            'host': CONFIG.get('host', ''),
            'host_ip': CONFIG.get('host_ip', ''),
            'registered': REGISTERED
        }
        self.wfile.write(json.dumps(response, indent=2).encode('utf-8'))
    
    def handle_config_post(self):
        """更新配置"""
        global REGISTERED, HEARTBEAT_THREAD, REGISTER_THREAD
        
        try:
            content_length = int(self.headers.get('Content-Length', 0))
            if content_length == 0:
                self.send_error(400, "Bad Request: Empty body")
                return
            
            post_data = self.rfile.read(content_length)
            request_data = json.loads(post_data.decode('utf-8'))
            
            # 更新easydarwin_url
            if 'easydarwin_url' in request_data:
                new_url = request_data['easydarwin_url'].strip()
                if new_url:
                    # 规范化URL，确保包含协议前缀
                    if not (new_url.startswith('http://') or new_url.startswith('https://')):
                        new_url = f"http://{new_url}"
                    
                    old_url = CONFIG['easydarwin_url']
                    CONFIG['easydarwin_url'] = new_url
                    print(f"\n[{time.strftime('%H:%M:%S')}] EasyDarwin地址已更新: {old_url} -> {new_url}")
                    
                    # 如果之前已注册，先注销
                    if REGISTERED:
                        try:
                            unregister_service()
                        except:
                            pass
                        REGISTERED = False
                    
                    # 重新注册服务
                    if register_service():
                        # 如果立即成功，启动心跳线程
                        if HEARTBEAT_THREAD is None or not HEARTBEAT_THREAD.is_alive():
                            HEARTBEAT_THREAD = threading.Thread(target=heartbeat_loop, daemon=True)
                            HEARTBEAT_THREAD.start()
                        # 停止注册重试线程（如果存在）
                        if REGISTER_THREAD and REGISTER_THREAD.is_alive():
                            pass  # 线程会自动停止
                    else:
                        # 如果失败，启动注册重试线程
                        if REGISTER_THREAD is None or not REGISTER_THREAD.is_alive():
                            REGISTER_THREAD = threading.Thread(target=register_retry_loop, daemon=True)
                            REGISTER_THREAD.start()
            
            # 更新其他配置
            if 'host_ip' in request_data:
                old_host_ip = CONFIG.get('host_ip')
                CONFIG['host_ip'] = request_data['host_ip'].strip() or None
                print(f"[{time.strftime('%H:%M:%S')}] 主机IP已更新: {old_host_ip} -> {CONFIG['host_ip']}")
                
                # 如果之前已注册，重新注册服务以使用新的端点地址
                if REGISTERED:
                    try:
                        unregister_service()
                    except:
                        pass
                    REGISTERED = False
                    
                    # 重新注册服务
                    if register_service():
                        # 如果立即成功，启动心跳线程
                        if HEARTBEAT_THREAD is None or not HEARTBEAT_THREAD.is_alive():
                            HEARTBEAT_THREAD = threading.Thread(target=heartbeat_loop, daemon=True)
                            HEARTBEAT_THREAD.start()
                        # 停止注册重试线程（如果存在）
                        if REGISTER_THREAD and REGISTER_THREAD.is_alive():
                            pass  # 线程会自动停止
                    else:
                        # 如果失败，启动注册重试线程
                        if REGISTER_THREAD is None or not REGISTER_THREAD.is_alive():
                            REGISTER_THREAD = threading.Thread(target=register_retry_loop, daemon=True)
                            REGISTER_THREAD.start()
            
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            
            response = {
                'success': True,
                'message': '配置已更新',
                'config': {
                    'easydarwin_url': CONFIG['easydarwin_url'],
                    'host_ip': CONFIG.get('host_ip', ''),
                    'registered': REGISTERED
                }
            }
            self.wfile.write(json.dumps(response).encode('utf-8'))
            
        except json.JSONDecodeError:
            self.send_error(400, "Bad Request: Invalid JSON")
        except Exception as e:
            print(f"[{time.strftime('%H:%M:%S')}] 更新配置失败: {e}")
            self.send_response(500)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            response = {
                'success': False,
                'message': f'更新配置失败: {str(e)}'
            }
            self.wfile.write(json.dumps(response).encode('utf-8'))
    
    def handle_inference(self):
        """处理推理请求（实时检测专用）"""
        global MODEL
        
        start_time = time.time()
        image_url = ''
        task_id = 'unknown'
        request_id = uuid.uuid4().hex
        log_prefix = f"[{CONFIG['service_id']}][req={request_id}]"
        
        try:
            content_length = int(self.headers['Content-Length'])
            post_data = self.rfile.read(content_length)
            request_data = json.loads(post_data.decode('utf-8'))
            
            image_url = request_data.get('image_url', '')
            task_id = request_data.get('task_id', 'unknown')
            task_type = request_data.get('task_type', 'unknown')
            
            if not image_url:
                raise ValueError("缺少image_url参数")
            
            # 加载算法配置文件（用于区域过滤）
            algo_config = request_data.get('algo_config')
            if not algo_config:
                algo_config = load_algo_config(image_url)
            
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
            
            timestamp = time.strftime('%H:%M:%S')
            # print(f"\n{'='*60}")
            # print(f"{log_prefix} 收到推理请求 @ {timestamp}", flush=True)
            # print(f"{log_prefix} 任务ID: {task_id}, 任务类型: {task_type}", flush=True)
            # print(f"{log_prefix} 图片URL: {image_url}", flush=True)
            # print(f"{log_prefix} 推理模式: 单线程直接推理", flush=True)
            # print(f"{'-'*60}")
            
            # 直接推理（主线程执行，避免ACL跨线程问题）
            inference_start = time.time()
            boxes_out = om_infer(CONFIG['model_path'], image, debug=False)
            inference_time = (time.time() - inference_start) * 1000
            
            # 更新统计（只有推理成功才统计）
            STATS['total_requests'] += 1
            STATS['total_inference_time'] += inference_time
            STATS['last_inference_time'] = inference_time
            
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
                    objects = filter_objects_by_region(objects, algo_config, image_size)
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
            
            # 计算总处理时间
            total_time = (time.time() - start_time) * 1000
            
            # 更新最近一次总耗时
            STATS['last_total_time'] = total_time
            
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
            
            # print(f"{log_prefix} 返回告警JSON: {json.dumps(response, ensure_ascii=False)}", flush=True)
            # print(f"{log_prefix} 推理完成: {inference_time:.0f}ms, 总耗时 {total_time:.0f}ms")
            # print(f"{'='*60}")
            
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
            
            print(f"{log_prefix} 返回告警JSON: {json.dumps(error_response, ensure_ascii=False)}", flush=True)
            
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
    
    # 优先使用手动指定的主机IP，然后是默认值127.0.0.1
    endpoint = f"http://{CONFIG['host']}:{CONFIG['port']}/infer"
    if CONFIG['host'] == '0.0.0.0':
        # 如果手动指定了主机IP且不为空，直接使用
        host_ip = CONFIG.get('host_ip')
        if host_ip and host_ip.strip():
            endpoint = f"http://{host_ip.strip()}:{CONFIG['port']}/infer"
        else:
            # 默认使用127.0.0.1
            endpoint = f"http://127.0.0.1:{CONFIG['port']}/infer"
    
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
    global RUNNING, HEARTBEAT_THREAD
    
    parser = argparse.ArgumentParser(description='YOLOv11x人头检测算法服务（实时检测）')
    parser.add_argument('--service-id', default='yolo11x_head_detector',
                        help='服务ID')
    parser.add_argument('--name', default='YOLOv11x人头检测算法',
                        help='服务名称')
    parser.add_argument('--task-types', nargs='+', default=['人数统计'],
                        help='支持的任务类型')
    parser.add_argument('--port', type=int, default=7901,
                        help='监听端口 (默认: 7901)')
    parser.add_argument('--host', default='0.0.0.0',
                        help='监听地址')
    parser.add_argument('--easydarwin', default='127.0.0.1:5066',
                        help='EasyDarwin地址')
    parser.add_argument('--model', default='./weight/best.om',
                        help='OM模型路径 (.om)')
    parser.add_argument('--device-id', type=int, default=0,
                        help='Ascend 设备ID (默认: 0)')
    parser.add_argument('--host-ip', type=str, default=None,
                        help='主机IP地址 (用于注册到EasyDarwin，默认自动检测)')
    parser.add_argument('--no-register', action='store_true',
                        help='不注册到EasyDarwin')
    parser.add_argument('--log-dir', default='./logs',
                        help='日志目录 (默认: ./logs，相对于工作目录)')
    parser.add_argument('--log-file', default='realtime_detector.log',
                        help='日志文件名 (默认: realtime_detector.log)')
    
    args = parser.parse_args()

    # 处理相对路径：如果log_dir是相对路径，转换为绝对路径（相对于工作目录）
    log_dir = args.log_dir
    if not os.path.isabs(log_dir):
        log_dir = os.path.abspath(log_dir)
    
    CONFIG['log_dir'] = log_dir
    CONFIG['log_file'] = args.log_file
    
    # 更新模型路径（如果提供了参数）
    if args.model:
        if not os.path.isabs(args.model):
            CONFIG['model_path'] = os.path.abspath(args.model)
        else:
            CONFIG['model_path'] = args.model
    
    setup_logging(CONFIG['log_dir'], CONFIG['log_file'])
    
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
    # 模型路径已在上面处理，这里不再重复设置
    
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
