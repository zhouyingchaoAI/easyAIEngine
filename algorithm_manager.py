#!/usr/bin/env python3
"""
算法服务管理器 - Web界面
提供启动、停止、监控算法服务的Web界面
"""
import os
import sys
import json
import subprocess
import signal
import time
import psutil
from flask import Flask, render_template_string, jsonify, request
from flask_cors import CORS
import threading
import socket
from datetime import datetime
import urllib.request
import urllib.error
from pathlib import Path

# 获取可执行文件所在目录（支持PyInstaller打包后的情况）
def get_base_dir():
    """获取程序基础目录（logs、configs、weight的同级目录）"""
    if getattr(sys, 'frozen', False):
        # PyInstaller打包后的情况
        base_dir = Path(sys.executable).parent
    else:
        # 开发环境，使用脚本所在目录
        base_dir = Path(__file__).parent
    return base_dir.resolve()

BASE_DIR = get_base_dir()
LOGS_DIR = BASE_DIR / 'logs'
CONFIGS_DIR = BASE_DIR / 'configs'
WEIGHT_DIR = BASE_DIR / 'weight'

app = Flask(__name__)
CORS(app)

# 全局变量存储服务进程
# 检测是否在打包后的环境中运行
def get_service_executable(script_name):
    """获取服务可执行文件路径（支持打包后的环境）"""
    # 首先检查可执行文件是否存在（打包后的环境）
    exe_name = script_name.replace('.py', '')
    exe_path = BASE_DIR / exe_name
    if exe_path.exists() and exe_path.is_file():
        # 检查是否有执行权限（可执行文件）
        if os.access(exe_path, os.X_OK):
            return str(exe_path)
    
    # 如果可执行文件不存在，检查Python脚本（开发环境）
    script_path = BASE_DIR / script_name
    if script_path.exists():
        return 'python3'  # 返回解释器，脚本路径在命令中单独指定
    
    return None

SERVICES = {
    'realtime': {
        'name': '实时检测服务',
        'script': 'algorithm_service.py',
        'default_config': {
            'device_id': '0',
            'batch_size': 8,
            'batch_timeout': 0.1,
        },
        'instances': []
    },
    'line_crossing': {
        'name': '绊线统计算法服务',
        'script': 'algorithm_service_line_crossing.py',
        'default_config': {
            'device_id': '0',  # Ascend NPU设备ID
            'batch_size': 8,
            'batch_timeout': 0.1,
        },
        'instances': []
    }
}

# 存储每个实例的历史统计信息，用于计算每秒请求数
INSTANCE_HISTORY = {}  # {pid: {'last_total_requests': 0, 'last_timestamp': time.time()}}

# 存储历史统计数据，用于绘制曲线图
# 格式: {service_key: {'timestamps': [], 'requests_per_sec': [], 'responses_per_sec': []}}
HISTORY_DATA = {
    'realtime': {'timestamps': [], 'requests_per_sec': [], 'responses_per_sec': []},
    'line_crossing': {'timestamps': [], 'requests_per_sec': [], 'responses_per_sec': []}
}
# 保留历史数据的时间范围（秒），设置为24小时，0表示不限制
HISTORY_RETENTION_HOURS = 48  # 保留24小时的历史数据
HISTORY_RETENTION_SECONDS = HISTORY_RETENTION_HOURS * 3600 if HISTORY_RETENTION_HOURS > 0 else 0

# HTML模板
HTML_TEMPLATE = '''
<!DOCTYPE html>
<html>
<head>
    <title>算法服务管理器</title>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            padding: 16px;
            line-height: 1.5;
        }
        .container {
            max-width: 1600px;
            margin: 0 auto;
        }
        .header {
            background: linear-gradient(135deg, #ffffff 0%, #f8f9fa 100%);
            border-radius: 12px;
            padding: 20px 24px;
            margin-bottom: 16px;
            box-shadow: 0 4px 16px rgba(0,0,0,0.1), 0 2px 6px rgba(0,0,0,0.06);
            border: 1px solid rgba(255,255,255,0.8);
        }
        .header h1 {
            color: #1a202c;
            margin-bottom: 6px;
            font-size: 28px;
            font-weight: 800;
            letter-spacing: -0.5px;
        }
        .header p {
            color: #64748b;
            font-size: 14px;
            font-weight: 500;
        }
        .grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(600px, 1fr));
            gap: 12px;
            margin-bottom: 12px;
        }
        .gpu-layout {
            margin-bottom: 16px;
        }
        .gpu-grid {
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(280px, 1fr));
            gap: 12px;
        }
        .services-layout {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 16px;
            margin-bottom: 16px;
        }
        @media (max-width: 1400px) {
            .services-layout {
                grid-template-columns: 1fr;
            }
        }
        .card {
            background: white;
            border-radius: 12px;
            padding: 18px;
            box-shadow: 0 2px 12px rgba(0,0,0,0.08), 0 1px 3px rgba(0,0,0,0.04);
            border: 1px solid rgba(226, 232, 240, 0.8);
            transition: all 0.3s ease;
        }
        .card:hover {
            box-shadow: 0 4px 20px rgba(0,0,0,0.12), 0 2px 6px rgba(0,0,0,0.06);
            transform: translateY(-1px);
        }
        .card-title {
            font-size: 16px;
            font-weight: 700;
            color: #1a202c;
            margin-bottom: 14px;
            padding-bottom: 12px;
            border-bottom: 2px solid #e2e8f0;
            display: flex;
            align-items: center;
            justify-content: space-between;
        }
        .status-badge {
            display: inline-block;
            padding: 6px 14px;
            border-radius: 20px;
            font-size: 12px;
            font-weight: 700;
            margin-left: 12px;
            letter-spacing: 0.3px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        }
        .status-running { 
            background: linear-gradient(135deg, #48bb78 0%, #38a169 100%);
            color: white;
        }
        .status-stopped { 
            background: linear-gradient(135deg, #cbd5e0 0%, #a0aec0 100%);
            color: #4a5568;
        }
        .form-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 12px 16px;
            margin-bottom: 16px;
            padding: 16px;
            background: #f8fafc;
            border-radius: 10px;
            border: 1px solid #e2e8f0;
        }
        .form-group {
            margin-bottom: 0;
        }
        .form-group.full-width {
            grid-column: 1 / -1;
        }
        .form-group label {
            display: block;
            color: #334155;
            font-weight: 600;
            margin-bottom: 6px;
            font-size: 12px;
            letter-spacing: 0.1px;
        }
        .form-group input, .form-group select {
            width: 100%;
            padding: 10px 12px;
            border: 2px solid #e2e8f0;
            border-radius: 8px;
            font-size: 13px;
            transition: all 0.2s ease;
            background: white;
            color: #1a202c;
        }
        .form-group input:focus, .form-group select:focus {
            outline: none;
            border-color: #667eea;
            box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.1);
            transform: translateY(-1px);
        }
        .form-group input:hover, .form-group select:hover {
            border-color: #cbd5e0;
        }
        select:hover {
            border-color: #cbd5e0 !important;
        }
        select:focus {
            border-color: #667eea !important;
            box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.1) !important;
            outline: none !important;
        }
        .form-group small {
            display: block;
            margin-top: 6px;
            color: #64748b;
            font-size: 11px;
            line-height: 1.4;
        }
        .btn-group {
            display: flex;
            gap: 10px;
            margin-top: 16px;
            padding-top: 16px;
            border-top: 2px solid #e2e8f0;
        }
        .btn {
            padding: 12px 24px;
            border: none;
            border-radius: 8px;
            font-size: 13px;
            font-weight: 700;
            cursor: pointer;
            transition: all 0.3s ease;
            letter-spacing: 0.2px;
            box-shadow: 0 3px 10px rgba(0,0,0,0.12);
            position: relative;
            overflow: hidden;
            flex: 1;
        }
        .btn::before {
            content: '';
            position: absolute;
            top: 50%;
            left: 50%;
            width: 0;
            height: 0;
            border-radius: 50%;
            background: rgba(255,255,255,0.3);
            transform: translate(-50%, -50%);
            transition: width 0.6s, height 0.6s;
        }
        .btn:hover::before {
            width: 300px;
            height: 300px;
        }
        .btn-primary {
            background: linear-gradient(135deg, #667eea 0%, #5568d3 100%);
            color: white;
        }
        .btn-primary:hover { 
            background: linear-gradient(135deg, #5568d3 0%, #4c51bf 100%);
            transform: translateY(-2px);
            box-shadow: 0 6px 20px rgba(102, 126, 234, 0.4);
        }
        .btn-danger {
            background: linear-gradient(135deg, #f56565 0%, #e53e3e 100%);
            color: white;
        }
        .btn-danger:hover { 
            background: linear-gradient(135deg, #e53e3e 0%, #c53030 100%);
            transform: translateY(-2px);
            box-shadow: 0 6px 20px rgba(245, 101, 101, 0.4);
        }
        .btn-success {
            background: linear-gradient(135deg, #48bb78 0%, #38a169 100%);
            color: white;
        }
        .btn-success:hover { 
            background: linear-gradient(135deg, #38a169 0%, #2f855a 100%);
            transform: translateY(-2px);
            box-shadow: 0 6px 20px rgba(72, 187, 120, 0.4);
        }
        .btn:active {
            transform: translateY(0);
        }
        .btn:disabled {
            opacity: 0.5;
            cursor: not-allowed;
            transform: none !important;
        }
        .gpu-card {
            background: linear-gradient(135deg, #ffffff 0%, #f8fafc 100%);
            border: 2px solid #e2e8f0;
            border-radius: 12px;
            padding: 14px;
            transition: all 0.3s ease;
            box-shadow: 0 2px 8px rgba(0,0,0,0.06);
            position: relative;
            overflow: hidden;
        }
        .gpu-card::before {
            content: '';
            position: absolute;
            top: 0;
            left: 0;
            right: 0;
            height: 4px;
            background: linear-gradient(90deg, #48bb78 0%, #38a169 100%);
        }
        .gpu-card.warning::before {
            background: linear-gradient(90deg, #ed8936 0%, #dd6b20 100%);
        }
        .gpu-card.danger::before {
            background: linear-gradient(90deg, #f56565 0%, #e53e3e 100%);
        }
        .gpu-card:hover {
            border-color: #667eea;
            box-shadow: 0 4px 16px rgba(102, 126, 234, 0.2);
            transform: translateY(-2px);
        }
        .gpu-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 10px;
        }
        .gpu-name {
            font-weight: 800;
            color: #1a202c;
            font-size: 14px;
            display: flex;
            align-items: center;
            gap: 6px;
        }
        .gpu-usage-badge {
            background: linear-gradient(135deg, #48bb78 0%, #38a169 100%);
            color: white;
            padding: 6px 12px;
            border-radius: 20px;
            font-size: 12px;
            font-weight: 800;
            box-shadow: 0 2px 8px rgba(72, 187, 120, 0.3);
            min-width: 50px;
            text-align: center;
        }
        .gpu-usage-badge.warning {
            background: linear-gradient(135deg, #ed8936 0%, #dd6b20 100%);
            box-shadow: 0 2px 8px rgba(237, 137, 54, 0.3);
        }
        .gpu-usage-badge.danger {
            background: linear-gradient(135deg, #f56565 0%, #e53e3e 100%);
            box-shadow: 0 2px 8px rgba(245, 101, 101, 0.3);
        }
        .gpu-progress {
            margin-bottom: 12px;
        }
        .progress-bar {
            width: 100%;
            height: 20px;
            background: #e2e8f0;
            border-radius: 10px;
            overflow: hidden;
            box-shadow: inset 0 2px 4px rgba(0,0,0,0.1);
            position: relative;
        }
        .progress-fill {
            height: 100%;
            background: linear-gradient(90deg, #48bb78 0%, #38a169 100%);
            display: flex;
            align-items: center;
            justify-content: center;
            color: white;
            font-size: 11px;
            font-weight: 700;
            transition: width 0.5s ease;
            position: relative;
            overflow: hidden;
        }
        .progress-fill::after {
            content: '';
            position: absolute;
            top: 0;
            left: -100%;
            width: 100%;
            height: 100%;
            background: linear-gradient(90deg, transparent, rgba(255,255,255,0.3), transparent);
            animation: shimmer 2s infinite;
        }
        @keyframes shimmer {
            0% { left: -100%; }
            100% { left: 100%; }
        }
        .progress-fill.warning { background: linear-gradient(90deg, #ed8936 0%, #dd6b20 100%); }
        .progress-fill.danger { background: linear-gradient(90deg, #f56565 0%, #e53e3e 100%); }
        .gpu-info {
            display: grid;
            grid-template-columns: repeat(2, 1fr);
            gap: 8px;
            font-size: 11px;
        }
        .gpu-info-item {
            background: #f1f5f9;
            border-radius: 8px;
            padding: 8px 10px;
            text-align: center;
            border: 1px solid #e2e8f0;
        }
        .gpu-info-label {
            font-weight: 600;
            font-size: 10px;
            color: #64748b;
            margin-bottom: 4px;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }
        .gpu-info-value {
            font-weight: 800;
            color: #1a202c;
            font-size: 13px;
        }
        .instances-grid {
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(280px, 1fr));
            gap: 10px;
            margin-top: 0;
        }
        .instance-card {
            background: linear-gradient(135deg, #ffffff 0%, #f8fafc 100%);
            border: 2px solid #e2e8f0;
            border-radius: 12px;
            padding: 16px;
            transition: all 0.3s ease;
            box-shadow: 0 2px 8px rgba(0,0,0,0.04);
        }
        .instance-card:hover {
            border-color: #667eea;
            box-shadow: 0 8px 24px rgba(102, 126, 234, 0.2), 0 4px 12px rgba(0,0,0,0.08);
            transform: translateY(-4px);
        }
        .instance-header {
            display: flex;
            justify-content: space-between;
            align-items: flex-start;
            margin-bottom: 8px;
        }
        .instance-title {
            font-weight: 600;
            color: #2d3748;
            font-size: 13px;
            margin-bottom: 2px;
        }
        .instance-meta {
            display: flex;
            gap: 8px;
            flex-wrap: wrap;
            font-size: 10px;
            color: #718096;
            margin-bottom: 6px;
        }
        .instance-meta span {
            background: #edf2f7;
            padding: 2px 6px;
            border-radius: 4px;
        }
        .instance-stats {
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding-top: 6px;
            border-top: 1px solid #e2e8f0;
        }
        .instance-count {
            font-size: 14px;
            font-weight: 700;
            color: #48bb78;
        }
        .instance-endpoint {
            font-size: 9px;
            color: #4a5568;
            font-family: 'Courier New', monospace;
            background: #edf2f7;
            padding: 2px 4px;
            border-radius: 3px;
            word-break: break-all;
            margin-top: 4px;
        }
        .log-container {
            background: linear-gradient(135deg, #1a202c 0%, #2d3748 100%);
            color: #e2e8f0;
            padding: 20px;
            border-radius: 12px;
            font-family: 'Courier New', monospace;
            font-size: 12px;
            max-height: 400px;
            overflow-y: auto;
            line-height: 1.8;
            border: 1px solid #4a5568;
            box-shadow: inset 0 2px 8px rgba(0,0,0,0.3);
        }
        .log-container::-webkit-scrollbar {
            width: 8px;
        }
        .log-container::-webkit-scrollbar-track {
            background: #2d3748;
            border-radius: 4px;
        }
        .log-container::-webkit-scrollbar-thumb {
            background: #4a5568;
            border-radius: 4px;
        }
        .log-container::-webkit-scrollbar-thumb:hover {
            background: #718096;
        }
        .log-container div {
            padding: 2px 0;
            border-left: 3px solid transparent;
            padding-left: 8px;
        }
        .log-error {
            color: #fc8181;
            border-left-color: #f56565 !important;
            background: rgba(245, 101, 101, 0.1);
        }
        .log-warning {
            color: #f6ad55;
            border-left-color: #ed8936 !important;
            background: rgba(237, 137, 54, 0.1);
        }
        .log-info {
            color: #68d391;
            border-left-color: #48bb78 !important;
        }
        .service-info {
            background: linear-gradient(135deg, #f8fafc 0%, #f1f5f9 100%);
            padding: 12px 16px;
            border-radius: 10px;
            margin-bottom: 14px;
            font-size: 12px;
            border: 1px solid #e2e8f0;
            box-shadow: 0 1px 4px rgba(0,0,0,0.04);
        }
        .service-info-item {
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 10px 0;
            border-bottom: 1px solid #e2e8f0;
        }
        .service-info-item:last-child { border-bottom: none; }
        .service-info-item span {
            color: #64748b;
            font-weight: 500;
        }
        .service-info-item strong {
            color: #1a202c;
            font-weight: 700;
        }
        .refresh-btn {
            background: linear-gradient(135deg, #4299e1 0%, #3182ce 100%);
            color: white;
            padding: 8px 16px;
            border: none;
            border-radius: 8px;
            cursor: pointer;
            font-size: 12px;
            font-weight: 700;
            transition: all 0.3s ease;
            box-shadow: 0 2px 8px rgba(66, 153, 225, 0.3);
        }
        .refresh-btn:hover { 
            background: linear-gradient(135deg, #3182ce 0%, #2c5282 100%);
            transform: translateY(-2px);
            box-shadow: 0 4px 12px rgba(66, 153, 225, 0.4);
        }
        .refresh-btn:active {
            transform: translateY(0);
        }
        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }
        .loading {
            animation: spin 1s linear infinite;
            display: inline-block;
        }
        .chart-container {
            background: white;
            border-radius: 12px;
            padding: 16px;
            box-shadow: 0 2px 12px rgba(0,0,0,0.06), 0 1px 3px rgba(0,0,0,0.04);
            margin-bottom: 16px;
            border: 1px solid #e2e8f0;
        }
        .chart-wrapper {
            position: relative;
            height: 240px;
            margin-top: 12px;
        }
        .stats-card {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            border-radius: 12px;
            padding: 16px;
            margin-bottom: 16px;
            color: white;
            box-shadow: 0 4px 16px rgba(102, 126, 234, 0.25), 0 2px 8px rgba(0,0,0,0.12);
            border: 1px solid rgba(255,255,255,0.2);
            position: relative;
            overflow: hidden;
        }
        .stats-card::before {
            content: '';
            position: absolute;
            top: -50%;
            right: -50%;
            width: 200%;
            height: 200%;
            background: radial-gradient(circle, rgba(255,255,255,0.1) 0%, transparent 70%);
            animation: pulse 4s ease-in-out infinite;
        }
        @keyframes pulse {
            0%, 100% { transform: scale(1); opacity: 0.5; }
            50% { transform: scale(1.1); opacity: 0.8; }
        }
        .stats-card h3 {
            margin: 0 0 14px 0;
            font-size: 14px;
            font-weight: 700;
            opacity: 0.95;
            position: relative;
            z-index: 1;
        }
        .stats-grid {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 12px;
            position: relative;
            z-index: 1;
        }
        .stat-item {
            text-align: center;
            padding: 10px;
            background: rgba(255,255,255,0.1);
            border-radius: 10px;
            backdrop-filter: blur(10px);
            border: 1px solid rgba(255,255,255,0.2);
        }
        .stat-label {
            font-size: 11px;
            opacity: 0.9;
            margin-bottom: 8px;
            font-weight: 600;
            letter-spacing: 0.3px;
        }
        .stat-value {
            font-size: 26px;
            font-weight: 800;
            text-shadow: 0 2px 6px rgba(0,0,0,0.2);
        }
        .service-card {
            background: white;
            border-radius: 14px;
            padding: 20px;
            box-shadow: 0 4px 16px rgba(0,0,0,0.08), 0 1px 4px rgba(0,0,0,0.04);
            margin-bottom: 0;
            border: 1px solid rgba(226, 232, 240, 0.8);
            transition: all 0.3s ease;
            height: 100%;
            display: flex;
            flex-direction: column;
        }
        .service-card:hover {
            box-shadow: 0 6px 24px rgba(0,0,0,0.12), 0 2px 8px rgba(0,0,0,0.06);
            transform: translateY(-2px);
        }
        .service-card-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 16px;
            padding-bottom: 14px;
            border-bottom: 2px solid #e2e8f0;
        }
        .service-card-title {
            font-size: 18px;
            font-weight: 800;
            color: #1a202c;
            display: flex;
            align-items: center;
            gap: 10px;
            letter-spacing: -0.2px;
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🎯 算法服务管理器</h1>
            <p>管理和监控 YOLOv11x 人头检测算法服务</p>
        </div>

        <!-- GPU监控 -->
        <div class="gpu-layout">
            <div class="card" style="padding: 16px;">
                <div class="card-title" style="margin-bottom: 12px; padding-bottom: 10px;">
                    <span>💻 GPU 监控</span>
                    <button class="refresh-btn" onclick="loadGPUInfo()" style="padding: 6px 12px; font-size: 11px;">🔄 刷新</button>
                </div>
                <div id="gpu-info" class="gpu-grid">加载中...</div>
            </div>
        </div>

        <!-- 服务管理（实时检测和绊线统计同级） -->
        <div class="services-layout">
            <!-- 实时检测服务 -->
            <div class="service-card">
                <div class="service-card-header">
                    <div class="service-card-title">
                        🔴 实时检测服务
                        <span id="realtime-status" class="status-badge status-stopped">已停止</span>
                    </div>
                </div>
                
                <div class="service-info">
                    <div class="service-info-item">
                        <span>任务类型</span>
                        <strong>人数统计</strong>
                        
                    </div>
                    <div class="service-info-item">
                        <span>设备</span>
                        <strong>Ascend NPU（可多实例分配至不同 device_id）</strong>
                    </div>
                </div>

                        <!-- 总计统计和图表 -->
                <div class="stats-card">
                    <h3>📊 实时统计</h3>
                    <div class="stats-grid">
                        <div class="stat-item">
                            <div class="stat-label">📥 总每秒请求数</div>
                            <div class="stat-value" id="realtime-total-requests-per-sec">0.00</div>
                        </div>
                        <div class="stat-item">
                            <div class="stat-label">📤 总每秒返回数</div>
                            <div class="stat-value" id="realtime-total-responses-per-sec">0.00</div>
                        </div>
                    </div>
                </div>
                
                <!-- 曲线图 -->
                <div class="chart-container">
                    <div class="card-title">
                        <span>📈 请求/返回速率趋势图</span>
                        <span id="realtime-chart-info" style="font-size:11px;color:#718096;font-weight:normal;"></span>
                    </div>
                    <div class="chart-wrapper">
                        <canvas id="realtime-chart"></canvas>
                    </div>
                </div>

                <div class="form-grid">
                    <div class="form-group full-width">
                        <label>服务ID前缀（批量实例自动递增）</label>
                        <input type="text" id="realtime-service-prefix-input" value="yolo11x_head_detector" placeholder="例如: yolo11x_head_detector">
                    </div>
                    <div class="form-group">
                        <label>实例数量</label>
                        <input type="number" id="realtime-count-input" value="1" min="1" placeholder="要启动的实例个数">
                    </div>
                    <div class="form-group">
                        <label>设备列表（device_id）</label>
                        <input type="text" id="realtime-devices-input" value="0" placeholder="例如: 0,1,0">
                    </div>
                    <div class="form-group">
                        <label>批处理大小</label>
                        <input type="number" id="realtime-batch-input" value="8">
                    </div>
                    <div class="form-group">
                        <label>端口（0=自动分配 7901-7999）</label>
                        <input type="number" id="realtime-port-input" value="0" placeholder="0=自动分配">
                    </div>
                    <div class="form-group">
                        <label>推理端点IP</label>
                        <input type="text" id="realtime-infer-ip-input" value="172.16.5.207" placeholder="例如: 172.16.5.207">
                    </div>
                    <div class="form-group full-width">
                        <label>EasyDarwin地址</label>
                        <input type="text" id="realtime-easydarwin-input" value="172.16.5.207:5066" placeholder="例如: 172.16.5.207:5066 或 http://172.16.5.207:5066">
                    </div>
                </div>
                
                <div class="btn-group">
                    <button class="btn btn-success" onclick="startService('realtime')">▶️ 批量新增实例</button>
                    <button class="btn btn-danger" onclick="stopService('realtime')">⏹️ 停止全部实例</button>
                </div>

                <div style="margin-top:12px;padding-top:12px;border-top:2px solid #e2e8f0;flex-grow:1;display:flex;flex-direction:column;">
                    <div class="card-title" style="border:none;padding:0;margin:0 0 10px 0;font-size:13px;">📋 实例列表</div>
                    <div id="realtime-instances" class="instances-grid" style="flex-grow:1;">暂无实例</div>
                </div>
            </div>

            <!-- 绊线统计算法服务 -->
            <div class="service-card">
            <div class="service-card-header">
                <div class="service-card-title">
                    🟢 绊线统计算法服务
                    <span id="line_crossing-status" class="status-badge status-stopped">已停止</span>
                </div>
            </div>
            
            <div class="service-info">
                <div class="service-info-item">
                    <span>任务类型</span>
                    <strong>绊线人数统计</strong>
                </div>
                <div class="service-info-item">
                    <span>设备</span>
                    <strong>Ascend NPU（可多实例分配至不同 device_id）</strong>
                </div>
            </div>

            <!-- 总计统计和图表 -->
            <div class="stats-card" style="background: linear-gradient(135deg, #10b981 0%, #059669 100%);">
                <h3>📊 实时统计</h3>
                <div class="stats-grid">
                    <div class="stat-item">
                        <div class="stat-label">📥 总每秒请求数</div>
                        <div class="stat-value" id="line_crossing-total-requests-per-sec">0.00</div>
                    </div>
                    <div class="stat-item">
                        <div class="stat-label">📤 总每秒返回数</div>
                        <div class="stat-value" id="line_crossing-total-responses-per-sec">0.00</div>
                    </div>
                </div>
            </div>
            
            <!-- 曲线图 -->
            <div class="chart-container">
                <div class="card-title">
                    <span>📈 请求/返回速率趋势图</span>
                    <span id="line_crossing-chart-info" style="font-size:11px;color:#718096;font-weight:normal;"></span>
                </div>
                <div class="chart-wrapper">
                    <canvas id="line_crossing-chart"></canvas>
                </div>
            </div>

            <div class="form-grid">
                <div class="form-group full-width">
                    <label>服务ID前缀（批量实例自动递增）</label>
                    <input type="text" id="line_crossing-service-prefix-input" value="yolo11x_line_crossing" placeholder="例如: yolo11x_line_crossing">
                </div>
                <div class="form-group">
                    <label>实例数量</label>
                    <input type="number" id="line_crossing-count-input" value="1" min="1" placeholder="要启动的实例个数">
                </div>
                <div class="form-group">
                    <label>设备列表（device_id）</label>
                    <input type="text" id="line_crossing-devices-input" value="0" placeholder="例如: 0,1,0">
                </div>
                <div class="form-group">
                    <label>批处理大小</label>
                    <input type="number" id="line_crossing-batch-input" value="8">
                </div>
                <div class="form-group">
                    <label>批处理超时（秒）</label>
                    <input type="number" id="line_crossing-batch-timeout-input" value="0.1" step="0.1" min="0.1">
                </div>
                <div class="form-group">
                    <label>端口（0=自动分配 7901-7999）</label>
                    <input type="number" id="line_crossing-port-input" value="0" placeholder="0=自动分配">
                </div>
                <div class="form-group">
                        <label>推理端点IP</label>
                        <input type="text" id="line_crossing-infer-ip-input" value="172.16.5.207" placeholder="例如: 172.16.5.207">
                </div>
                <div class="form-group">
                    <label>模型路径（可选，默认使用./weight/best.om）</label>
                    <input type="text" id="line_crossing-model-input" value="" placeholder="留空使用默认OM模型 ./weight/best.om">
                </div>
                <div class="form-group full-width">
                        <label>EasyDarwin地址</label>
                        <input type="text" id="line_crossing-easydarwin-input" value="172.16.5.207:5066" placeholder="例如: 172.16.5.207:5066 或 http://172.16.5.207:5066">
                </div>
                <div class="form-group full-width" style="border-top: 1px solid #e2e8f0; padding-top: 12px; margin-top: 12px;">
                    <label style="font-weight: 600; color: #2d3748;">📹 视频保存配置</label>
                </div>
                <div class="form-group" style="display: flex; align-items: center; gap: 8px;">
                    <input type="checkbox" id="line_crossing-enable-video-save-input" style="width: auto; margin: 0;">
                    <label for="line_crossing-enable-video-save-input" style="margin: 0; font-weight: normal;">启用视频保存（默认关闭，不开启则不执行绘制操作）</label>
                </div>
                <div class="form-group">
                    <label>视频保存目录</label>
                    <input type="text" id="line_crossing-video-save-dir-input" value="./videos" placeholder="例如: ./videos">
                </div>
                <div class="form-group">
                    <label>视频帧率（FPS）</label>
                    <input type="number" id="line_crossing-video-fps-input" value="25" min="1" max="60">
                </div>
                <div class="form-group">
                    <label>视频分段时长（秒）</label>
                    <input type="number" id="line_crossing-video-segment-duration-input" value="60" min="10" max="3600" step="10">
                    <small style="color: #718096; font-size: 11px; display: block; margin-top: 4px;">每个视频片段的最大时长，默认60秒（1分钟）</small>
                </div>
                <div class="form-group">
                    <label>视频分段最大大小（MB）</label>
                    <input type="number" id="line_crossing-video-segment-max-size-input" value="500" min="10" max="5000" step="10">
                    <small style="color: #718096; font-size: 11px; display: block; margin-top: 4px;">每个视频片段的最大文件大小，默认500MB</small>
                </div>
            </div>
            
            <div class="btn-group">
                <button class="btn btn-success" onclick="startService('line_crossing')">▶️ 批量新增实例</button>
                <button class="btn btn-danger" onclick="stopService('line_crossing')">⏹️ 停止全部实例</button>
            </div>

            <div style="margin-top:12px;padding-top:12px;border-top:2px solid #e2e8f0;flex-grow:1;display:flex;flex-direction:column;">
                <div class="card-title" style="border:none;padding:0;margin:0 0 10px 0;font-size:13px;">📋 实例列表</div>
                <div id="line_crossing-instances" class="instances-grid" style="flex-grow:1;">暂无实例</div>
            </div>
        </div>

        <!-- 视频下载 -->
        <div class="service-card">
            <div class="service-card-header">
                <div class="service-card-title">
                    📹 绊线视频下载
                </div>
            </div>
            
            <div id="video-list-container" style="min-height: 200px; padding: 8px;">
                <p style="color: #64748b; text-align: center; padding: 40px; font-size: 14px; font-weight: 500;">正在加载视频列表...</p>
            </div>
        </div>

        <!-- 系统日志 -->
        <div class="card" style="margin-top: 16px;">
            <div class="card-title">
                <span>📋 系统日志</span>
                <div style="display:flex;gap:10px;align-items:center;flex-wrap:wrap;">
                    <select id="log-service" onchange="loadLogs()" style="padding: 10px 14px; border-radius: 8px; border: 2px solid #e2e8f0; font-size: 13px; font-weight: 600; background: white; color: #334155; cursor: pointer; transition: all 0.2s;">
                        <option value="all">全部日志</option>
                        <option value="manager">管理器日志</option>
                        <option value="realtime">实时检测日志</option>
                        <option value="line_crossing">绊线统计日志</option>
                    </select>
                    <select id="log-lines" onchange="loadLogs()" style="padding: 10px 14px; border-radius: 8px; border: 2px solid #e2e8f0; font-size: 13px; font-weight: 600; background: white; color: #334155; cursor: pointer; transition: all 0.2s;">
                        <option value="50">50行</option>
                        <option value="100" selected>100行</option>
                        <option value="200">200行</option>
                        <option value="500">500行</option>
                    </select>
                    <button class="refresh-btn" onclick="loadLogs()">🔄 刷新</button>
                    <button class="refresh-btn" onclick="clearLogs()" style="background: linear-gradient(135deg, #f56565 0%, #e53e3e 100%); box-shadow: 0 2px 8px rgba(245, 101, 101, 0.3);">🗑️ 清空</button>
                </div>
            </div>
            <div class="log-container" id="logs">
                暂无日志...
            </div>
        </div>
    </div>

    <script>
        // 自动刷新间隔（毫秒）
        const REFRESH_INTERVAL = 3000;
let autoRefresh = true;
let gpuDataLoaded = false;
let videoDataLoaded = false;
        
        // 图表对象
        const charts = {
            realtime: null,
            line_crossing: null
        };

        // 初始化图表
        function initChart(serviceKey) {
            const canvasId = `${serviceKey}-chart`;
            const ctx = document.getElementById(canvasId);
            if (!ctx) return;
            
            charts[serviceKey] = new Chart(ctx, {
                type: 'line',
                data: {
                    labels: [],
                    datasets: [
                        {
                            label: '每秒请求数 (req/s)',
                            data: [],
                            borderColor: 'rgb(66, 153, 225)',
                            backgroundColor: 'rgba(66, 153, 225, 0.1)',
                            tension: 0.4,
                            fill: true
                        },
                        {
                            label: '每秒返回数 (res/s)',
                            data: [],
                            borderColor: 'rgb(72, 187, 120)',
                            backgroundColor: 'rgba(72, 187, 120, 0.1)',
                            tension: 0.4,
                            fill: true
                        }
                    ]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    plugins: {
                        legend: {
                            display: true,
                            position: 'top',
                            labels: {
                                usePointStyle: true,
                                padding: 15,
                                font: {
                                    size: 12
                                }
                            }
                        },
                        tooltip: {
                            mode: 'index',
                            intersect: false
                        }
                    },
                    scales: {
                        x: {
                            display: true,
                            title: {
                                display: true,
                                text: '时间'
                            },
                            ticks: {
                                maxRotation: 45,
                                minRotation: 0,
                                autoSkip: true,
                                maxTicksLimit: 50
                            }
                        },
                        y: {
                            display: true,
                            title: {
                                display: true,
                                text: '速率 (req/s)'
                            },
                            beginAtZero: true
                        }
                    },
                    interaction: {
                        mode: 'nearest',
                        axis: 'x',
                        intersect: false
                    }
                }
            });
        }

        // 更新图表数据
        function updateChart(serviceKey, historyData) {
            if (!charts[serviceKey] || !historyData) return;
            
            const chart = charts[serviceKey];
            const timestamps = historyData.timestamps || [];
            const requests = historyData.requests_per_sec || [];
            const responses = historyData.responses_per_sec || [];
            
            // 更新图表信息显示
            const infoEl = document.getElementById(`${serviceKey}-chart-info`);
            if (infoEl && timestamps.length > 0) {
                const firstTime = new Date(timestamps[0] * 1000);
                const lastTime = new Date(timestamps[timestamps.length - 1] * 1000);
                const duration = Math.round((timestamps[timestamps.length - 1] - timestamps[0]) / 60); // 分钟
                infoEl.textContent = `(${timestamps.length}个数据点，跨度约${duration}分钟)`;
            }
            
            // 格式化时间标签（只显示时分秒）
            // 当数据点很多时，只显示部分标签以提高性能
            const maxLabels = 50; // 最多显示50个时间标签
            const labelStep = Math.max(1, Math.floor(timestamps.length / maxLabels));
            
            const labels = timestamps.map((ts, index) => {
                // 只对部分索引显示标签，其他显示空字符串
                if (index % labelStep === 0 || index === timestamps.length - 1) {
                    const date = new Date(ts * 1000);
                    return date.toLocaleTimeString('zh-CN', { hour: '2-digit', minute: '2-digit', second: '2-digit' });
                }
                return '';
            });
            
            chart.data.labels = labels;
            chart.data.datasets[0].data = requests;
            chart.data.datasets[1].data = responses;
            
            // 优化图表配置以处理大量数据
            chart.options.scales.x.ticks.maxTicksLimit = maxLabels;
            chart.options.scales.x.ticks.autoSkip = true;
            chart.options.scales.x.ticks.maxRotation = timestamps.length > 20 ? 45 : 0;
            
            chart.update('none'); // 'none' 模式避免动画，提高性能
        }

        // 通用请求工具，带超时控制
        async function fetchWithTimeout(url, options = {}, timeoutMs = 5000) {
            const controller = new AbortController();
            const id = setTimeout(() => controller.abort(), timeoutMs);
            try {
                const response = await fetch(url, { ...options, signal: controller.signal });
                clearTimeout(id);
                return response;
            } catch (error) {
                clearTimeout(id);
                throw error;
            }
        }

        // 加载视频列表
        async function loadVideos() {
        const container = document.getElementById('video-list-container');
        if (!videoDataLoaded) {
            container.innerHTML = '<p style="color: #64748b; padding: 20px; text-align: center;">正在加载视频列表...</p>';
        }
            try {
                const response = await fetchWithTimeout('/api/videos', {}, 5000);
                if (!response.ok) {
                    throw new Error(`HTTP ${response.status}`);
                }
                const data = await response.json();
                
                if (data.error) {
                    container.innerHTML = `<p style="color: #f44336; padding: 20px; text-align: center;">${data.error}</p>`;
                    return;
                }
                
                if (!data.videos || data.videos.length === 0) {
                    container.innerHTML = '<p style="color: #666; text-align: center; padding: 20px;">暂无视频文件</p>';
                    return;
                }
                
                videoDataLoaded = true;
                
                let html = `
                    <div style="display:flex; justify-content:space-between; align-items:center; flex-wrap:wrap; gap:10px; margin-bottom:10px;">
                        <div style="color:#4a5568; font-size:12px;">共 ${data.videos.length} 个视频</div>
                        <div style="display:flex; gap:8px;">
                            <button class="refresh-btn" onclick="loadVideos()" style="padding:6px 12px; font-size:11px;">🔄 刷新</button>
                            <button class="refresh-btn" onclick="deleteAllVideos()" style="padding:6px 12px; font-size:11px; background:linear-gradient(135deg,#f56565 0%,#e53e3e 100%); box-shadow:0 2px 8px rgba(245,101,101,0.3);">🗑️ 一键删除</button>
                        </div>
                    </div>
                    <div style="overflow-x:auto;">
                `;
                
                html += '<table style="width: 100%; border-collapse: collapse; min-width: 600px; font-size: 12px;">';
                html += '<thead><tr style="background: #f7fafc; border-bottom: 2px solid #e2e8f0;">';
                html += '<th style="padding: 10px; text-align: left; font-size: 12px; color: #4a5568;">文件名</th>';
                html += '<th style="padding: 10px; text-align: left; font-size: 12px; color: #4a5568;">大小</th>';
                html += '<th style="padding: 10px; text-align: left; font-size: 12px; color: #4a5568;">修改时间</th>';
                html += '<th style="padding: 10px; text-align: left; font-size: 12px; color: #4a5568;">操作</th>';
                html += '</tr></thead><tbody>';
                
                data.videos.forEach(video => {
                    const isWriting = video.is_writing || false;
                    const writingBadge = isWriting ? '<span style="background: #ed8936; color: white; padding: 2px 6px; border-radius: 3px; font-size: 10px; margin-left: 5px;">写入中</span>' : '';
                    const safeFilename = JSON.stringify(video.filename || '');
                    const downloadBtn = isWriting 
                        ? '<span style="background: #cbd5e0; color: #718096; padding: 6px 12px; border-radius: 4px; display: inline-block; font-size: 12px; font-weight: 600; cursor: not-allowed;">⬇️ 下载 (写入中)</span>'
                        : `<a href="/api/videos/${encodeURIComponent(video.filename)}" 
                               style="background: #48bb78; color: white; padding: 6px 12px; border-radius: 4px; text-decoration: none; display: inline-block; font-size: 12px; font-weight: 600;" 
                               download>⬇️ 下载</a>`;
                    const deleteBtn = isWriting
                        ? '<span style="background: #cbd5e0; color: #718096; padding: 6px 12px; border-radius: 4px; display: inline-block; font-size: 12px; font-weight: 600; cursor: not-allowed; margin-left: 5px;">🗑️ 删除 (写入中)</span>'
                        : `<button onclick='deleteVideo(${safeFilename})' 
                               style="background: #f56565; color: white; padding: 6px 12px; border: none; border-radius: 4px; font-size: 12px; font-weight: 600; cursor: pointer; margin-left: 5px;">🗑️ 删除</button>`;
                    
                    html += `<tr style="border-bottom: 1px solid #e2e8f0;">
                        <td style="padding: 10px; font-size: 13px; color: #2d3748;">${video.filename}${writingBadge}</td>
                        <td style="padding: 10px; font-size: 13px; color: #4a5568;">${video.size_mb} MB</td>
                        <td style="padding: 10px; font-size: 13px; color: #4a5568;">${video.modified_time}</td>
                        <td style="padding: 10px;">
                            <div style="display:flex; gap:6px; flex-wrap:wrap; align-items:center;">
                                ${downloadBtn}
                                ${deleteBtn}
                            </div>
                        </td>
                    </tr>`;
                });
                
                html += '</tbody></table></div>';
                container.innerHTML = html;
            } catch (error) {
                console.error('加载视频列表失败:', error);
                videoDataLoaded = false;
                container.innerHTML = 
                    '<p style="color: #f44336; padding: 20px; text-align: center;">加载视频列表失败，请稍后重试</p>';
            }
        }

        async function loadGPUInfo() {
        const container = document.getElementById('gpu-info');
        if (!gpuDataLoaded) {
            container.innerHTML = '<p style="color: #64748b; padding: 12px; text-align: center;">正在加载 GPU 信息...</p>';
        }
            try {
                const response = await fetchWithTimeout('/api/gpu-info', {}, 4000);
                if (!response.ok) {
                    throw new Error(`HTTP ${response.status}`);
                }
                const data = await response.json();
                
                let html = '';
                if (data.gpus && data.gpus.length > 0) {
                    data.gpus.forEach(gpu => {
                        const usage = gpu.memory_used_percent || 0;
                        let progressClass = '';
                        let cardClass = '';
                        let badgeClass = '';
                        if (usage > 80) {
                            progressClass = 'danger';
                            cardClass = 'danger';
                            badgeClass = 'danger';
                        } else if (usage > 60) {
                            progressClass = 'warning';
                            cardClass = 'warning';
                            badgeClass = 'warning';
                        }
                        
                        html += `
                            <div class="gpu-card ${cardClass}">
                                <div class="gpu-header">
                                    <div class="gpu-name">💻 ${gpu.name || 'NPU'} #${gpu.id}</div>
                                    <div class="gpu-usage-badge ${badgeClass}">${usage.toFixed(0)}%</div>
                                </div>
                                <div class="gpu-progress">
                                    <div class="progress-bar">
                                        <div class="progress-fill ${progressClass}" style="width: ${usage}%">${usage.toFixed(0)}%</div>
                                    </div>
                                </div>
                                <div class="gpu-info">
                                    <div class="gpu-info-item">
                                        <div class="gpu-info-label">显存</div>
                                        <div class="gpu-info-value">${gpu.memory_used || 'N/A'}</div>
                                    </div>
                                    <div class="gpu-info-item">
                                        <div class="gpu-info-label">AICore</div>
                                        <div class="gpu-info-value">${gpu.utilization || 'N/A'}%</div>
                                    </div>
                                    <div class="gpu-info-item">
                                        <div class="gpu-info-label">温度</div>
                                        <div class="gpu-info-value">${gpu.temperature || 'N/A'}</div>
                                    </div>
                                    <div class="gpu-info-item">
                                        <div class="gpu-info-label">功率</div>
                                        <div class="gpu-info-value">${gpu.power || 'N/A'}</div>
                                    </div>
                                </div>
                            </div>
                        `;
                    });
                } else {
                    html = '<div style="grid-column: 1/-1; text-align: center; padding: 20px; color: #718096;">无法获取设备信息（npu-smi / nvidia-smi 不可用）</div>';
                }
                
                container.innerHTML = html;
                gpuDataLoaded = true;
            } catch (error) {
                console.error('加载GPU信息失败:', error);
                container.innerHTML = '<p style="color: #f44336; padding: 12px; text-align: center;">GPU 信息加载失败，请稍后重试</p>';
                gpuDataLoaded = false;
            }
        }
        
        // 页面加载时初始化
        window.onload = function() {
            // 初始化图表
            initChart('realtime');
            initChart('line_crossing');
            
            loadGPUInfo();
            loadServiceStatus();
            loadLogs();
            loadHistoryData();
            loadVideos();
            
            // 自动刷新
            setInterval(() => {
                if (autoRefresh) {
                    loadGPUInfo();
                    loadServiceStatus();
                    loadHistoryData();
                }
            }, REFRESH_INTERVAL);
            
            // 每10秒刷新一次视频列表
            setInterval(loadVideos, 10000);
        };
        
        // 加载历史数据
        async function loadHistoryData() {
            try {
                const response = await fetch('/api/history-data');
                const data = await response.json();
                
                Object.keys(data).forEach(serviceKey => {
                    if (charts[serviceKey]) {
                        updateChart(serviceKey, data[serviceKey]);
                    }
                });
            } catch (error) {
                console.error('加载历史数据失败:', error);
            }
        }

        // 渲染实例列表
        function renderInstances(serviceKey, instances) {
            const container = document.getElementById(`${serviceKey}-instances`);
            if (!container) return;
            if (!instances || instances.length === 0) {
                container.innerHTML = '<p style="color:#718096;text-align:center;padding:20px;">暂无实例</p>';
                return;
            }
            const rows = instances.map(ins => {
                const count = (ins.stats && ins.stats.total_requests != null) ? ins.stats.total_requests : '-';
                const lastInferTime = (ins.stats && ins.stats.last_inference_time != null) ? ins.stats.last_inference_time.toFixed(2) : '-';
                const lastTotalTime = (ins.stats && ins.stats.last_total_time != null) ? ins.stats.last_total_time.toFixed(2) : '-';
                const requestsPerSec = (ins.stats && ins.stats.requests_per_second != null) ? ins.stats.requests_per_second.toFixed(2) : '-';
                const responsesPerSec = (ins.stats && ins.stats.responses_per_second != null) ? ins.stats.responses_per_second.toFixed(2) : '-';
                const inferIp = ins.config.infer_ip || '172.16.5.207';
                const inferUrl = `http://${inferIp}:${ins.config.port}/infer`;
                const serviceId = ins.config.service_id || `实例_${ins.pid}`;
                return `
                <div class="instance-card">
                    <div class="instance-header">
                        <div style="flex:1;">
                            <div class="instance-title">${serviceId}</div>
                            <div class="instance-meta">
                                <span>PID: ${ins.pid || '-'}</span>
                                <span>端口: ${ins.config.port}</span>
                                <span>GPU: ${ins.config.device_id || '-'}</span>
                            </div>
                        </div>
                        <button class="btn btn-danger" style="padding:4px 10px;font-size:10px;" onclick="stopInstance('${serviceKey}', ${ins.pid})">⏹️</button>
                    </div>
                    <div class="instance-stats">
                        <div>
                            <div style="font-size:10px;color:#718096;margin-bottom:2px;">累计推理</div>
                            <div class="instance-count">${count}</div>
                        </div>
                        <div style="text-align:right;">
                            <div style="font-size:9px;color:#718096;margin-bottom:2px;">⚡ 推理时间</div>
                            <div style="font-size:13px;font-weight:600;color:#667eea;">${lastInferTime} ms</div>
                        </div>
                    </div>
                    <div style="display:flex;justify-content:space-between;padding-top:6px;border-top:1px solid #e2e8f0;margin-top:6px;">
                        <div style="font-size:9px;color:#718096;">🕒 总耗时</div>
                        <div style="font-size:12px;font-weight:600;color:#4a5568;">${lastTotalTime} ms</div>
                    </div>
                    <div style="display:grid;grid-template-columns:1fr 1fr;gap:8px;padding-top:6px;border-top:1px solid #e2e8f0;margin-top:6px;">
                        <div>
                            <div style="font-size:9px;color:#718096;margin-bottom:2px;">📥 每秒请求数</div>
                            <div style="font-size:13px;font-weight:600;color:#48bb78;">${requestsPerSec} req/s</div>
                        </div>
                        <div style="text-align:right;">
                            <div style="font-size:9px;color:#718096;margin-bottom:2px;">📤 每秒返回数</div>
                            <div style="font-size:13px;font-weight:600;color:#4299e1;">${responsesPerSec} res/s</div>
                        </div>
                    </div>
                    <div class="instance-endpoint">${inferUrl}</div>
                </div>`;
            }).join('');
            container.innerHTML = rows;
        }

        async function stopInstance(serviceKey, pid) {
            if (!confirm(`确定要停止实例 PID ${pid} 吗？`)) return;
            try {
                const response = await fetch('/api/stop-service', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ service: serviceKey, pid })
                });
                const data = await response.json();
                if (data.success) {
                    loadServiceStatus();
                    loadLogs();
                } else {
                    alert(`❌ ${data.message}`);
                }
            } catch (e) {
                alert('❌ 停止失败: ' + e);
            }
        }

        // 加载服务状态
        async function loadServiceStatus() {
            try {
                const response = await fetch('/api/services');
                const data = await response.json();
                
                Object.keys(data).forEach(serviceKey => {
                    const service = data[serviceKey];
                    const instances = service.instances || [];
                    const isRunning = instances.length > 0;
                    
                    // 更新状态标签
                    const statusEl = document.getElementById(`${serviceKey}-status`);
                    statusEl.textContent = isRunning ? `运行中 (${instances.length})` : '已停止';
                    statusEl.className = `status-badge ${isRunning ? 'status-running' : 'status-stopped'}`;
                    renderInstances(serviceKey, instances);
                    
                    // 更新总计统计
                    const totalRequestsPerSec = service.total_requests_per_second || 0;
                    const totalResponsesPerSec = service.total_responses_per_second || 0;
                    const totalRequestsEl = document.getElementById(`${serviceKey}-total-requests-per-sec`);
                    const totalResponsesEl = document.getElementById(`${serviceKey}-total-responses-per-sec`);
                    if (totalRequestsEl) {
                        totalRequestsEl.textContent = totalRequestsPerSec.toFixed(2);
                    }
                    if (totalResponsesEl) {
                        totalResponsesEl.textContent = totalResponsesPerSec.toFixed(2);
                    }
                });
            } catch (error) {
                console.error('加载服务状态失败:', error);
            }
        }

        // 启动服务
        async function startService(serviceKey) {
            const count = parseInt(document.getElementById(`${serviceKey}-count-input`).value || '1');
            const devices = document.getElementById(`${serviceKey}-devices-input`).value;
            const port = document.getElementById(`${serviceKey}-port-input`).value;
            const batchSize = document.getElementById(`${serviceKey}-batch-input`).value;
            const inferIp = document.getElementById(`${serviceKey}-infer-ip-input`) ? document.getElementById(`${serviceKey}-infer-ip-input`).value || '172.16.5.207' : '172.16.5.207';
            const easydarwinUrl = document.getElementById(`${serviceKey}-easydarwin-input`).value || '172.16.5.207:5066';
            const servicePrefix = document.getElementById(`${serviceKey}-service-prefix-input`).value || (serviceKey === 'line_crossing' ? 'yolo11x_line_crossing' : 'yolo11x_head_detector');
            
            // 构建请求体
            const requestBody = {
                service: serviceKey,
                count: count,
                device_ids: devices,
                port: parseInt(port),
                batch_size: parseInt(batchSize),
                easydarwin_url: easydarwinUrl,
                service_id_prefix: servicePrefix
            };
            
            // 绊线算法需要额外参数
            if (serviceKey === 'line_crossing') {
                const batchTimeout = document.getElementById(`${serviceKey}-batch-timeout-input`).value || '0.1';
                const model = document.getElementById(`${serviceKey}-model-input`).value;
                const enableVideoSave = document.getElementById(`${serviceKey}-enable-video-save-input`).checked;
                const videoSaveDir = document.getElementById(`${serviceKey}-video-save-dir-input`).value || './videos';
                const videoFps = document.getElementById(`${serviceKey}-video-fps-input`).value || '25';
                const videoSegmentDuration = document.getElementById(`${serviceKey}-video-segment-duration-input`).value || '60';
                const videoSegmentMaxSize = document.getElementById(`${serviceKey}-video-segment-max-size-input`).value || '500';
                requestBody.batch_timeout = parseFloat(batchTimeout);
                requestBody.infer_ip = inferIp;  // 绊线算法也需要推理端点IP
                requestBody.enable_video_save = enableVideoSave;
                requestBody.video_save_dir = videoSaveDir;
                requestBody.video_fps = parseInt(videoFps);
                requestBody.video_segment_duration = parseInt(videoSegmentDuration);
                requestBody.video_segment_max_size_mb = parseInt(videoSegmentMaxSize);
                if (model) {
                    requestBody.model = model;
                }
            } else {
                requestBody.infer_ip = inferIp;
            }
            
            try {
                const response = await fetch('/api/start-service', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify(requestBody)
                });
                
                const data = await response.json();
                if (data.success) {
                    alert(`✅ ${data.message}`);
                    loadServiceStatus();
                    loadLogs();
                } else {
                    alert(`❌ ${data.message}`);
                }
            } catch (error) {
                alert('❌ 启动失败: ' + error);
            }
        }

        // 停止服务
        async function stopService(serviceKey) {
            if (!confirm('确定要停止此服务吗？')) {
                return;
            }
            
            try {
                const response = await fetch('/api/stop-service', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ service: serviceKey })
                });
                
                const data = await response.json();
                if (data.success) {
                    alert(`✅ ${data.message}`);
                    loadServiceStatus();
                    loadLogs();
                } else {
                    alert(`❌ ${data.message}`);
                }
            } catch (error) {
                alert('❌ 停止失败: ' + error);
            }
        }

        // 加载日志
        async function loadLogs() {
            try {
                const service = document.getElementById('log-service').value;
                const lines = document.getElementById('log-lines').value;
                
                const response = await fetch(`/api/logs?service=${service}&lines=${lines}`);
                const data = await response.json();
                
                const logsEl = document.getElementById('logs');
                if (data.logs && data.logs.length > 0) {
                    logsEl.innerHTML = data.logs.map(log => {
                        // 高亮不同类型的日志
                        let className = '';
                        if (log.includes('ERROR') || log.includes('失败')) {
                            className = 'log-error';
                        } else if (log.includes('WARNING') || log.includes('警告')) {
                            className = 'log-warning';
                        } else if (log.includes('INFO') || log.includes('成功')) {
                            className = 'log-info';
                        }
                        return `<div class="${className}">${escapeHtml(log)}</div>`;
                    }).join('');
                    logsEl.scrollTop = logsEl.scrollHeight;
                } else {
                    logsEl.innerHTML = '暂无日志...';
                }
            } catch (error) {
                console.error('加载日志失败:', error);
            }
        }
        
        // 清空日志
        async function clearLogs() {
            if (!confirm('确定要清空当前日志吗？')) {
                return;
            }
            
            try {
                const service = document.getElementById('log-service').value;
                const response = await fetch('/api/clear-logs', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ service: service })
                });
                
                const data = await response.json();
                if (data.success) {
                    alert(`✅ ${data.message}`);
                    loadLogs();
                } else {
                    alert(`❌ ${data.message}`);
                }
            } catch (error) {
                alert('❌ 清空失败: ' + error);
            }
        }

        function escapeHtml(text) {
            const div = document.createElement('div');
            div.textContent = text;
            return div.innerHTML;
        }
        
        // 删除视频
        async function deleteVideo(filename) {
            if (!confirm(`确定要删除视频 "${filename}" 吗？`)) {
                return;
            }
            
            try {
                const response = await fetchWithTimeout(`/api/videos/${encodeURIComponent(filename)}`, {
                    method: 'DELETE'
                }, 5000);
                const data = await response.json();
                
                if (data.success) {
                    alert(`✅ ${data.message}`);
                    loadVideos();
                } else {
                    alert(`❌ ${data.message}`);
                }
            } catch (error) {
                alert('❌ 删除失败: ' + error);
            }
        }

        async function deleteAllVideos() {
            if (!confirm('确定要删除所有非写入中的视频吗？')) {
                return;
            }
            try {
                const response = await fetchWithTimeout('/api/videos/delete_all', {
                    method: 'POST'
                }, 5000);
                const data = await response.json();
                if (data.success) {
                    alert(`✅ ${data.message}`);
                    loadVideos();
                } else {
                    alert(`❌ ${data.message}`);
                }
            } catch (error) {
                alert('❌ 删除全部失败: ' + error);
            }
        }
    </script>
</body>
</html>
'''


def get_gpu_info():
    """获取设备信息（仅使用 Ascend NPU: npu-smi info）"""
    # Ascend NPU: npu-smi info（两行一组：第一行含 Name/Health/Power/Temp；第二行含 NPU/Device/AICore/Memory-Usage）
    try:
        # 尝试 JSON 输出（优先），不同版本参数可能不同：-t json 或 info -t json
        json_cmds = [
            ['/usr/local/sbin/npu-smi', '-t', 'json'],
            ['/usr/local/sbin/npu-smi', 'info', '-t', 'json'],
            ['npu-smi', '-t', 'json'],
            ['npu-smi', 'info', '-t', 'json']
        ]
        json_text = None
        for cmd in json_cmds:
            try:
                r = subprocess.run(cmd, capture_output=True, text=True, timeout=4)
                if r.returncode == 0 and r.stdout and r.stdout.strip().startswith(('{', '[')):
                    json_text = r.stdout
                    break
            except Exception:
                continue

        if json_text:
            import json as _json
            try:
                data = _json.loads(json_text)
                devices = []

                # 尝试常见结构：顶层 list
                if isinstance(data, list):
                    devices = data
                # 顶层 dict：找包含设备数组的键
                elif isinstance(data, dict):
                    for k, v in data.items():
                        if isinstance(v, list) and len(v) and isinstance(v[0], dict):
                            devices = v
                            break
                        if isinstance(v, dict):
                            for kk, vv in v.items():
                                if isinstance(vv, list) and len(vv) and isinstance(vv[0], dict):
                                    devices = vv
                                    break
                            if devices:
                                break

                gpus = []
                for d in devices:
                    if not isinstance(d, dict):
                        continue
                    npu_id = d.get('id') or d.get('device_id') or d.get('npu_id')
                    name = d.get('name') or d.get('chip_name') or 'Ascend NPU'
                    health = d.get('health') or d.get('health_status') or 'N/A'
                    temperature = d.get('temp') or d.get('temperature')
                    power = d.get('power')
                    aicore = d.get('aicore') or d.get('aicore_usage') or d.get('ai_core')
                    # 内存
                    mem_used = None
                    mem_total = None
                    if isinstance(d.get('memory'), dict):
                        mem_used = d['memory'].get('used') or d['memory'].get('used_mb')
                        mem_total = d['memory'].get('total') or d['memory'].get('total_mb')
                    else:
                        mem_used = d.get('memory_used') or d.get('memory_used_mb')
                        mem_total = d.get('memory_total') or d.get('memory_total_mb')

                    # 组装
                    try:
                        npu_id = int(npu_id) if npu_id is not None else 0
                    except Exception:
                        npu_id = 0
                    try:
                        mem_used = float(mem_used) if mem_used is not None else None
                        mem_total = float(mem_total) if mem_total is not None else None
                    except Exception:
                        mem_used, mem_total = None, None
                    mem_percent = (mem_used / mem_total * 100) if (mem_used is not None and mem_total and mem_total > 0) else 0

                    gpus.append({
                        'id': npu_id,
                        'name': name,
                        'health': health,
                        'memory_used': f'{mem_used:.0f} MB' if mem_used is not None else 'N/A',
                        'memory_total': f'{mem_total:.0f} MB' if mem_total is not None else 'N/A',
                        'memory_used_percent': mem_percent,
                        'utilization': aicore if aicore is not None else 'N/A',
                        'temperature': (str(temperature) + 'C') if isinstance(temperature, (int, float)) else (temperature or 'N/A'),
                        'power': (str(power) + 'W') if isinstance(power, (int, float)) else (power or 'N/A')
                    })

                if gpus:
                    return gpus
            except Exception:
                pass

        # 若 JSON 失败，回退到文本解析
        # 优先绝对路径，避免 PATH 差异
        try:
            result = subprocess.run(
                ['/usr/local/sbin/npu-smi', 'info'], capture_output=True, text=True, timeout=5
            )
        except FileNotFoundError:
            result = subprocess.run(
                ['npu-smi', 'info'], capture_output=True, text=True, timeout=5
            )
        if result.returncode == 0 and result.stdout:
            gpus = []
            import re
            raw_lines = [l for l in result.stdout.split('\n') if l.strip()]
            # 仅取表格数据行
            data_lines = [l.strip() for l in raw_lines if l.strip().startswith('|') and l.strip().endswith('|')]
            # 扫描包含 used/total 的行，向上配对上一行
            for idx, line in enumerate(data_lines):
                if not re.search(r"\d+\s*/\s*\d+", line):
                    continue
                # 第二行（含 used/total）
                second_cells = [c.strip() for c in line.strip('|').split('|')]
                # 第一行（上一行）
                if idx == 0:
                    continue
                first_line = data_lines[idx - 1]
                if ('Name' in first_line) or ('Process id' in first_line):
                    # 如果紧挨着标题，则尝试再往上找一行
                    if idx >= 2:
                        first_line = data_lines[idx - 2]
                    else:
                        continue
                first_cells = [c.strip() for c in first_line.strip('|').split('|')]
                if len(first_cells) < 3 or len(second_cells) < 4:
                    continue
                try:
                    # 第一行提取 name/health/power/temp
                    left_tokens = first_cells[0].split()
                    name = left_tokens[-1] if len(left_tokens) >= 2 else 'Ascend NPU'
                    health = first_cells[1].split()[0] if first_cells[1] else 'N/A'
                    tail = first_cells[2]
                    # 温度：取尾部第一个数字作为温度（单位C）
                    mt = re.findall(r'(\d+)', tail)
                    temperature = (mt[-1] + 'C') if mt else 'N/A'
                    # 功率：如果不是 NA，取开头数字
                    power = 'N/A'
                    mp = re.match(r'^(\d+)', tail.strip())
                    if mp:
                        power = mp.group(1) + 'W'
                    # 第二行提取 npu_id/aicore/used/total
                    id_tokens = second_cells[0].split()
                    npu_id = int(id_tokens[0]) if id_tokens and id_tokens[0].isdigit() else 0
                    aicore = second_cells[2].split()[0] if second_cells[2] else 'N/A'
                    # 从右往左找第一个含 used/total 的列
                    mem_field = ''
                    for col in reversed(second_cells):
                        if '/' in col:
                            mem_field = col
                            break
                    m = re.search(r'(\d+)\s*/\s*(\d+)', mem_field)
                    if not m:
                        continue
                    mem_used = float(m.group(1))
                    mem_total = float(m.group(2))
                    mem_percent = (mem_used / mem_total * 100) if mem_total > 0 else 0
                    gpus.append({
                        'id': npu_id,
                        'name': name,
                        'health': health,
                        'memory_used': f'{mem_used:.0f} MB',
                        'memory_total': f'{mem_total:.0f} MB',
                        'memory_used_percent': mem_percent,
                        'utilization': aicore,
                        'temperature': temperature,
                        'power': power
                    })
                except Exception:
                    continue
            if gpus:
                return gpus
            # 兜底：仅基于含 used/total 的行快速生成条目
            fallback = []
            for idx, line in enumerate(data_lines):
                if not re.search(r"\d+\s*/\s*\d+", line):
                    continue
                cells = [c.strip() for c in line.strip('|').split('|')]
                if not cells:
                    continue
                # id
                id_tokens = cells[0].split()
                try:
                    npu_id = int(id_tokens[0]) if id_tokens and id_tokens[0].isdigit() else 0
                except Exception:
                    npu_id = 0
                # aicore
                aicore = cells[2].split()[0] if len(cells) >= 3 and cells[2] else 'N/A'
                # mem used/total（自右向左找含 / 的列）
                mem_field = ''
                for col in reversed(cells):
                    if '/' in col:
                        mem_field = col; break
                m = re.search(r"(\d+)\s*/\s*(\d+)", mem_field)
                if not m:
                    continue
                used = float(m.group(1)); total = float(m.group(2))
                fallback.append({
                    'id': npu_id,
                    'name': 'Ascend NPU',
                    'health': 'N/A',
                    'memory_used': f'{used:.0f} MB',
                    'memory_total': f'{total:.0f} MB',
                    'memory_used_percent': (used/total*100) if total>0 else 0,
                    'utilization': aicore,
                    'temperature': 'N/A',
                    'power': 'N/A'
                })
            return fallback
    except Exception:
        return []


def get_process_status(pid):
    """检查进程是否运行"""
    if not pid:
        return False
    
    try:
        process = psutil.Process(pid)
        return process.is_running()
    except (psutil.NoSuchProcess, psutil.AccessDenied):
        return False


def find_service_pid(script_name):
    """查找服务进程ID"""
    try:
        for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
            try:
                cmdline = proc.info.get('cmdline', [])
                if cmdline and script_name in ' '.join(cmdline):
                    return proc.info['pid']
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
    except Exception as e:
        print(f"查找进程失败: {str(e)}")
    
    return None


@app.route('/')
def index():
    """首页"""
    return render_template_string(HTML_TEMPLATE)


@app.route('/api/gpu-info')
def api_gpu_info():
    """获取GPU信息API"""
    gpus = get_gpu_info()
    return jsonify({'gpus': gpus})


@app.route('/api/gpu-debug')
def api_gpu_debug():
    """GPU解析调试：返回 npu-smi 原文、提取的表格行与解析结果"""
    try:
        # 取原文
        try:
            result = subprocess.run(
                ['/usr/local/sbin/npu-smi', 'info'], capture_output=True, text=True, timeout=5
            )
        except FileNotFoundError:
            result = subprocess.run(
                ['npu-smi', 'info'], capture_output=True, text=True, timeout=5
            )
        raw = result.stdout if result and result.stdout else ''

        # 解析
        import re
        parsed = []
        data_lines = [l for l in raw.split('\n') if l.strip().startswith('|') and l.strip().endswith('|')]
        i = 0
        while i < len(data_lines) - 1:
            first = data_lines[i]
            second = data_lines[i + 1]
            # 跳过标题
            if ('Name' in first) or ('Process id' in first):
                i += 1
                continue
            first_cells = [c.strip() for c in first.strip('|').split('|')]
            second_cells = [c.strip() for c in second.strip('|').split('|')]
            cond = (
                len(first_cells) >= 3 and len(second_cells) >= 4 and
                first_cells[0][:1].isdigit() and second_cells[0][:1].isdigit() and
                re.search(r'(\d+)\s*/\s*(\d+)', second_cells[-1])
            )
            if not cond:
                i += 1
                continue
            try:
                left_tokens = first_cells[0].split()
                name = left_tokens[-1] if len(left_tokens) >= 2 else 'Ascend NPU'
                health = first_cells[1].split()[0] if first_cells[1] else 'N/A'
                tail_tokens = first_cells[2].split()
                power = 'N/A'
                temperature = 'N/A'
                if tail_tokens:
                    if tail_tokens[0] != 'NA':
                        power = tail_tokens[0] + 'W'
                    if len(tail_tokens) >= 2 and tail_tokens[1].isdigit():
                        temperature = tail_tokens[1] + 'C'

                id_tokens = second_cells[0].split()
                npu_id = int(id_tokens[0]) if id_tokens and id_tokens[0].isdigit() else 0
                aicore = second_cells[2].split()[0] if second_cells[2] else 'N/A'
                m = re.search(r'(\d+)\s*/\s*(\d+)', second_cells[-1])
                mem_used = float(m.group(1)) if m else None
                mem_total = float(m.group(2)) if m else None
                parsed.append({
                    'npu_id': npu_id,
                    'name': name,
                    'health': health,
                    'aicore': aicore,
                    'power': power,
                    'temperature': temperature,
                    'mem_used': mem_used,
                    'mem_total': mem_total,
                    'first': first,
                    'second': second
                })
            except Exception:
                pass
            finally:
                i += 2

        return jsonify({'raw': raw, 'lines': data_lines, 'parsed': parsed})
    except Exception as e:
        return jsonify({'error': str(e)})


@app.route('/api/services')
def api_services():
    """获取服务状态API（多实例）"""
    global INSTANCE_HISTORY, HISTORY_DATA
    result = {}
    current_time = time.time()
    
    for key, service in SERVICES.items():
        instances_out = []
        alive_instances = []
        # 总计变量
        total_requests_per_second = 0.0
        total_responses_per_second = 0.0
        
        # 清理死亡实例并收集状态
        for ins in service.get('instances', []):
            pid = ins.get('pid')
            if pid and get_process_status(pid):
                # 查询实例统计
                stats = None
                requests_per_second = 0.0
                responses_per_second = 0.0
                try:
                    port = ins.get('config', {}).get('port')
                    if port:
                        with urllib.request.urlopen(f'http://127.0.0.1:{port}/stats', timeout=0.5) as resp:
                            data = json.loads(resp.read().decode('utf-8'))
                            if isinstance(data, dict):
                                if 'statistics' in data and isinstance(data['statistics'], dict):
                                    s = data['statistics']
                                    total_requests = s.get('total_requests', 0)
                                    stats = {
                                        'total_requests': total_requests,
                                        'last_inference_time': s.get('last_inference_time', 0),
                                        'last_total_time': s.get('last_total_time', 0)
                                    }
                                elif 'total_requests' in data:
                                    total_requests = data.get('total_requests', 0)
                                    stats = {
                                        'total_requests': total_requests,
                                        'last_inference_time': data.get('last_inference_time', 0),
                                        'last_total_time': data.get('last_total_time', 0)
                                    }
                                
                                # 计算每秒请求数和返回结果数
                                if pid in INSTANCE_HISTORY:
                                    history = INSTANCE_HISTORY[pid]
                                    time_diff = current_time - history['last_timestamp']
                                    if time_diff > 0:
                                        requests_diff = total_requests - history['last_total_requests']
                                        requests_per_second = requests_diff / time_diff
                                        # 返回结果数通常等于请求数（每个请求都会返回结果）
                                        responses_per_second = requests_per_second
                                
                                # 更新历史记录
                                INSTANCE_HISTORY[pid] = {
                                    'last_total_requests': total_requests,
                                    'last_timestamp': current_time
                                }
                except Exception:
                    stats = None
                    # 如果查询失败，清理历史记录
                    if pid in INSTANCE_HISTORY:
                        del INSTANCE_HISTORY[pid]
                
                # 将每秒请求数和返回结果数添加到stats中
                if stats is not None:
                    stats['requests_per_second'] = round(requests_per_second, 2)
                    stats['responses_per_second'] = round(responses_per_second, 2)
                    # 累加到总计
                    total_requests_per_second += requests_per_second
                    total_responses_per_second += responses_per_second
                
                ins['stats'] = stats
                alive_instances.append(ins)
                instances_out.append({
                    'pid': pid,
                    'config': ins.get('config', {}),
                    'stats': stats
                })
            else:
                # 进程已死亡，清理历史记录
                if pid and pid in INSTANCE_HISTORY:
                    del INSTANCE_HISTORY[pid]
        
        # 覆盖为存活实例
        service['instances'] = alive_instances
        
        # 记录历史数据
        if key in HISTORY_DATA:
            history = HISTORY_DATA[key]
            history['timestamps'].append(current_time)
            history['requests_per_sec'].append(total_requests_per_second)
            history['responses_per_sec'].append(total_responses_per_second)
            
            # 基于时间范围清理旧数据（如果设置了保留时间）
            if HISTORY_RETENTION_SECONDS > 0:
                cutoff_time = current_time - HISTORY_RETENTION_SECONDS
                # 找到第一个大于cutoff_time的时间戳索引
                keep_from_index = 0
                for i, ts in enumerate(history['timestamps']):
                    if ts > cutoff_time:
                        keep_from_index = i
                        break
                
                # 如果找到了需要清理的数据，则保留从keep_from_index开始的数据
                if keep_from_index > 0:
                    history['timestamps'] = history['timestamps'][keep_from_index:]
                    history['requests_per_sec'] = history['requests_per_sec'][keep_from_index:]
                    history['responses_per_sec'] = history['responses_per_sec'][keep_from_index:]
        
        result[key] = {
            'name': service['name'],
            'instances': instances_out,
            'total_requests_per_second': round(total_requests_per_second, 2),
            'total_responses_per_second': round(total_responses_per_second, 2)
        }
    
    return jsonify(result)


@app.route('/api/start-service', methods=['POST'])
def api_start_service():
    """启动服务API"""
    data = request.json
    service_key = data.get('service')
    count = int(data.get('count', 1))
    devices_raw = data.get('device_ids', '0')
    port = data.get('port', 0)
    batch_size = data.get('batch_size', 8)
    batch_timeout = data.get('batch_timeout', 0.1)  # 批处理超时
    infer_ip = data.get('infer_ip', '172.16.5.207')  # 推理端点IP，默认为172.16.5.207
    easydarwin_url = data.get('easydarwin_url', '172.16.5.207:5066')  # EasyDarwin地址，默认为172.16.5.207:5066
    service_id_prefix = data.get('service_id_prefix', 'yolo11x_head_detector')
    model_path = data.get('model', None)  # 模型路径（绊线算法需要）
    # 视频保存配置（绊线算法）
    enable_video_save = data.get('enable_video_save', False)  # 是否启用视频保存
    video_save_dir = data.get('video_save_dir', './videos')  # 视频保存目录
    video_fps = data.get('video_fps', 25)  # 视频帧率
    video_segment_duration = data.get('video_segment_duration', 60)  # 视频分段时长（秒）
    video_segment_max_size_mb = data.get('video_segment_max_size_mb', 500)  # 视频分段最大大小（MB）
    
    if service_key not in SERVICES:
        return jsonify({'success': False, 'message': '未知服务'})
    
    service = SERVICES[service_key]
    
    try:
        # 端口自动分配（默认范围 7901-7999）
        def is_port_free(p):
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                s.settimeout(0.2)
                try:
                    s.bind(('0.0.0.0', p))
                    return True
                except OSError:
                    return False

        if not isinstance(port, int):
            try:
                port = int(port)
            except Exception:
                port = 0

        # 解析设备列表
        device_ids = [d.strip() for d in str(devices_raw).split(',') if d.strip() != '']
        if not device_ids:
            device_ids = ['0']

        started = []
        reserved_ports = set()

        # 打开日志文件
        log_dir = str(LOGS_DIR)
        os.makedirs(log_dir, exist_ok=True)
        
        log_file_map = {
            'realtime': 'realtime_detector.log',
            'line_crossing': 'line_crossing.log'
        }
        log_file = os.path.join(log_dir, log_file_map.get(service_key, 'service.log'))
        log_handle = open(log_file, 'a', encoding='utf-8')
        
        for i in range(max(1, count)):
            # 为每个实例分配端口
            inst_port = port
            if inst_port == 0:
                assigned = None
                for candidate in range(7901, 8000):
                    if candidate not in reserved_ports and is_port_free(candidate):
                        assigned = candidate
                        break
                if assigned is None:
                    break
                inst_port = assigned
            else:
                # 非0作为起始端口，向上寻找空闲且未保留端口
                assigned = None
                candidate = inst_port if i == 0 else (inst_port + i)
                # 若被占用或冲突，继续递增查找
                while candidate < 8000:
                    if candidate not in reserved_ports and is_port_free(candidate):
                        assigned = candidate
                        break
                    candidate += 1
                if assigned is None:
                    break
                inst_port = assigned

            # 预留端口，避免本批次重复
            reserved_ports.add(inst_port)

            # 选择设备（循环分配）
            device_id = device_ids[i % len(device_ids)]

            # 根据服务类型构建不同的启动命令
            if service_key == 'line_crossing':
                # 绊线算法使用NPU（OM模型）
                if model_path is None or model_path == '':
                    model_path = str(WEIGHT_DIR / 'best.om')  # 使用OM模型
                
                # 检测是否在打包后的环境中
                service_exe = get_service_executable(service['script'])
                if service_exe and service_exe != 'python3':
                    # 打包后的环境，直接使用可执行文件
                    cmd = [
                        service_exe,
                        '--service-id', f"{service_id_prefix}_{inst_port}",
                        '--port', str(inst_port),
                        '--device-id', str(device_id),  # Ascend NPU设备ID
                        '--easydarwin', easydarwin_url,
                        '--host-ip', infer_ip,  # 传递推理端点IP给服务，用于注册到EasyDarwin
                        '--model', model_path,  # 模型路径（.om文件）
                        '--batch-size', str(batch_size),
                        '--batch-timeout', str(batch_timeout)
                    ]
                    # 如果启用视频保存，添加相关参数
                    if enable_video_save:
                        cmd.extend([
                            '--enable-video-save',
                            '--video-save-dir', str(video_save_dir),
                            '--video-fps', str(video_fps),
                            '--video-segment-duration', str(video_segment_duration),
                            '--video-segment-max-size-mb', str(video_segment_max_size_mb)
                        ])
                else:
                    # 开发环境，使用Python运行脚本
                    script_path = str(BASE_DIR / service['script'])
                    cmd = [
                        'python3',
                        script_path,
                        '--service-id', f"{service_id_prefix}_{inst_port}",
                        '--port', str(inst_port),
                        '--device-id', str(device_id),  # Ascend NPU设备ID
                        '--easydarwin', easydarwin_url,
                        '--host-ip', infer_ip,  # 传递推理端点IP给服务，用于注册到EasyDarwin
                        '--model', model_path,  # 模型路径（.om文件）
                        '--batch-size', str(batch_size),
                        '--batch-timeout', str(batch_timeout)
                    ]
                    # 如果启用视频保存，添加相关参数
                    if enable_video_save:
                        cmd.extend([
                            '--enable-video-save',
                            '--video-save-dir', str(video_save_dir),
                            '--video-fps', str(video_fps),
                            '--video-segment-duration', str(video_segment_duration),
                            '--video-segment-max-size-mb', str(video_segment_max_size_mb)
                        ])
            else:
                # 实时检测服务使用NPU
                # 使用相对路径，确保在打包后也能正确工作
                if model_path is None:
                    model_path = str(WEIGHT_DIR / 'best.om')
                
                # 检测是否在打包后的环境中
                service_exe = get_service_executable(service['script'])
                if service_exe and service_exe != 'python3':
                    # 打包后的环境，直接使用可执行文件
                    cmd = [
                        service_exe,
                        '--service-id', f"{service_id_prefix}_{inst_port}",
                        '--port', str(inst_port),
                        '--device-id', str(device_id),
                        '--easydarwin', easydarwin_url,
                        '--host-ip', infer_ip,  # 传递推理端点IP给服务，用于注册到EasyDarwin
                        '--model', model_path,  # 模型路径
                        '--log-dir', str(LOGS_DIR)  # 日志目录
                    ]
                else:
                    # 开发环境，使用Python运行脚本
                    script_path = str(BASE_DIR / service['script'])
                    cmd = [
                        'python3',
                        script_path,
                        '--service-id', f"{service_id_prefix}_{inst_port}",
                        '--port', str(inst_port),
                        '--device-id', str(device_id),
                        '--easydarwin', easydarwin_url,
                        '--host-ip', infer_ip,  # 传递推理端点IP给服务，用于注册到EasyDarwin
                        '--model', model_path,  # 模型路径
                        '--log-dir', str(LOGS_DIR)  # 日志目录
                    ]

            # 记录日志标记
            log_handle.write(f"\n{'='*60}\n")
            log_handle.write(f"服务启动: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            if service_key == 'line_crossing':
                log_handle.write(f"NPU设备: {device_id}, 端口: {inst_port}, 批处理: {batch_size}, 超时: {batch_timeout}\n")
                log_handle.write(f"模型路径: {model_path}\n")
                log_handle.write(f"推理端点IP: {infer_ip} (用于注册到EasyDarwin)\n")
                if enable_video_save:
                    log_handle.write(f"视频保存: 已启用 (目录: {video_save_dir}, 帧率: {video_fps}, 分段时长: {video_segment_duration}秒, 最大大小: {video_segment_max_size_mb}MB)\n")
                else:
                    log_handle.write(f"视频保存: 未启用\n")
            else:
                log_handle.write(f"DEVICE: {device_id}, 端口: {inst_port}, 批处理: {batch_size}\n")
                log_handle.write(f"推理端点IP: {infer_ip} (用于注册到EasyDarwin)\n")
            log_handle.write(f"服务ID: {service_id_prefix}_{inst_port}\n")
            log_handle.write(f"EasyDarwin地址: {easydarwin_url}\n")
            log_handle.write(f"{'='*60}\n")
            log_handle.flush()

            process = subprocess.Popen(
                cmd,
                cwd=str(BASE_DIR),
                stdout=log_handle,
                stderr=subprocess.STDOUT,
                start_new_session=True
            )

            time.sleep(0.5)
            if process.poll() is None:
                service_id = f"{service_id_prefix}_{inst_port}"
                instance_config = {
                    'device_id': device_id,  # 对于绊线算法，这里实际是gpu_id
                    'port': inst_port,
                    'batch_size': batch_size,
                    'service_id': service_id,
                    'infer_ip': infer_ip  # 所有服务都保存推理端点IP
                }
                if service_key == 'line_crossing':
                    instance_config['batch_timeout'] = batch_timeout
                    if model_path:
                        instance_config['model_path'] = model_path
                    instance_config['enable_video_save'] = enable_video_save
                    if enable_video_save:
                        instance_config['video_save_dir'] = video_save_dir
                        instance_config['video_fps'] = video_fps
                        instance_config['video_segment_duration'] = video_segment_duration
                        instance_config['video_segment_max_size_mb'] = video_segment_max_size_mb
                
                instance = {
                    'process': process,
                    'pid': process.pid,
                    'config': instance_config,
                    'stats': None
                }
                service.setdefault('instances', []).append(instance)
                started.append({'pid': process.pid, 'port': inst_port, 'device_id': device_id, 'service_id': service_id})

        if not started:
            return jsonify({'success': False, 'message': '实例启动失败'}), 500

        return jsonify({'success': True, 'message': f"已启动 {len(started)} 个实例", 'instances': started})
            
    except Exception as e:
        return jsonify({
            'success': False,
            'message': f'启动失败: {str(e)}'
        })


@app.route('/api/stop-service', methods=['POST'])
def api_stop_service():
    """停止服务API（支持停止单实例或全部）"""
    global INSTANCE_HISTORY
    data = request.json
    service_key = data.get('service')
    pid_to_stop = data.get('pid')
    
    if service_key not in SERVICES:
        return jsonify({'success': False, 'message': '未知服务'})
    
    service = SERVICES[service_key]
    instances = service.get('instances', [])
    
    if not instances:
        return jsonify({'success': False, 'message': f'{service["name"]}无运行实例'})
    
    try:
        targets = []
        if pid_to_stop:
            targets = [ins for ins in instances if ins.get('pid') == pid_to_stop]
            if not targets:
                return jsonify({'success': False, 'message': f'未找到 PID {pid_to_stop} 实例'})
        else:
            targets = list(instances)
        
        stopped = 0
        for ins in targets:
            pid = ins.get('pid')
            try:
                process = psutil.Process(pid)
                # 杀掉整个进程组
                children = process.children(recursive=True)
                for child in children:
                    try:
                        child.terminate()
                    except:
                        pass
                process.terminate()
                try:
                    process.wait(timeout=3)
                except psutil.TimeoutExpired:
                    for child in children:
                        try:
                            child.kill()
                        except:
                            pass
                    process.kill()
            except psutil.NoSuchProcess:
                pass
            # 清理历史记录
            if pid and pid in INSTANCE_HISTORY:
                del INSTANCE_HISTORY[pid]
            stopped += 1
        
        # 从实例列表移除
        if pid_to_stop:
            service['instances'] = [ins for ins in instances if ins.get('pid') != pid_to_stop]
        else:
            service['instances'] = []
        
        return jsonify({'success': True, 'message': f'已停止 {stopped} 个实例'})
    except Exception as e:
        return jsonify({'success': False, 'message': f'停止失败: {str(e)}'})


@app.route('/api/logs')
def api_logs():
    """获取最近的日志"""
    service = request.args.get('service', 'all')
    lines = int(request.args.get('lines', 100))
    
    try:
        logs = []
        log_dir = str(LOGS_DIR)
        
        if service == 'all':
            # 合并所有日志
            log_files = [
                ('manager', os.path.join(log_dir, 'manager.log')),
                ('realtime', os.path.join(log_dir, 'realtime_detector.log')),
                ('line_crossing', os.path.join(log_dir, 'line_crossing.log'))
            ]
            
            all_logs = []
            for svc_name, log_file in log_files:
                if os.path.exists(log_file):
                    with open(log_file, 'r', encoding='utf-8', errors='ignore') as f:
                        file_lines = f.readlines()
                        for line in file_lines[-lines:]:
                            all_logs.append(f"[{svc_name}] {line.strip()}")
            
            logs = all_logs[-lines:]
        else:
            # 单个服务日志
            log_file_map = {
                'manager': 'manager.log',
                'realtime': 'realtime_detector.log',
                'line_crossing': 'line_crossing.log'
            }
            
            log_file = os.path.join(log_dir, log_file_map.get(service, 'manager.log'))
            if os.path.exists(log_file):
                with open(log_file, 'r', encoding='utf-8', errors='ignore') as f:
                    file_lines = f.readlines()
                    logs = [line.strip() for line in file_lines[-lines:]]
        
        return jsonify({'logs': logs})
    except Exception as e:
        return jsonify({'logs': [f'读取日志失败: {str(e)}']})


@app.route('/api/clear-logs', methods=['POST'])
def api_clear_logs():
    """清空日志"""
    data = request.json
    service = data.get('service', 'all')
    
    try:
        log_dir = str(LOGS_DIR)
        
        log_file_map = {
            'all': ['manager.log', 'realtime_detector.log', 'line_crossing.log'],
            'manager': ['manager.log'],
            'realtime': ['realtime_detector.log'],
            'line_crossing': ['line_crossing.log']
        }
        
        files_to_clear = log_file_map.get(service, [])
        cleared_count = 0
        
        for log_file in files_to_clear:
            log_path = os.path.join(log_dir, log_file)
            if os.path.exists(log_path):
                # 清空文件内容
                with open(log_path, 'w', encoding='utf-8') as f:
                    f.write('')
                cleared_count += 1
        
        return jsonify({
            'success': True,
            'message': f'已清空 {cleared_count} 个日志文件'
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'message': f'清空失败: {str(e)}'
        })


@app.route('/api/log-stats')
def api_log_stats():
    """获取日志统计信息"""
    try:
        log_dir = str(LOGS_DIR)
        stats = {}
        
        log_files = {
            'manager': 'manager.log',
            'realtime': 'realtime_detector.log',
            'line_crossing': 'line_crossing.log'
        }
        
        for key, filename in log_files.items():
            log_path = os.path.join(log_dir, filename)
            if os.path.exists(log_path):
                stat = os.stat(log_path)
                with open(log_path, 'r', encoding='utf-8', errors='ignore') as f:
                    lines = sum(1 for _ in f)
                
                stats[key] = {
                    'size': stat.st_size,
                    'size_mb': round(stat.st_size / (1024 * 1024), 2),
                    'lines': lines,
                    'last_modified': datetime.fromtimestamp(stat.st_mtime).strftime('%Y-%m-%d %H:%M:%S')
                }
            else:
                stats[key] = {'exists': False}
        
        return jsonify(stats)
    except Exception as e:
        return jsonify({'error': str(e)})


@app.route('/api/history-data')
def api_history_data():
    """获取历史统计数据API"""
    global HISTORY_DATA
    try:
        # 返回所有服务的历史数据
        result = {}
        for key, history in HISTORY_DATA.items():
            result[key] = {
                'timestamps': history['timestamps'].copy(),
                'requests_per_sec': history['requests_per_sec'].copy(),
                'responses_per_sec': history['responses_per_sec'].copy()
            }
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)})


def resolve_video_directory_and_writing_videos():
    """解析视频目录并获取正在写入的视频列表"""
    video_dir = None
    writing_videos = set()
    
    line_crossing_service = SERVICES.get('line_crossing', {})
    instances = line_crossing_service.get('instances', [])
    
    for instance in instances:
        config = instance.get('config', {})
        if config.get('enable_video_save'):
            video_save_dir = config.get('video_save_dir', './videos')
            if os.path.isabs(video_save_dir):
                video_dir = Path(video_save_dir)
            else:
                video_dir = (BASE_DIR / video_save_dir).resolve()
            try:
                port = config.get('port')
                if port:
                    import urllib.request
                    with urllib.request.urlopen(f'http://127.0.0.1:{port}/api/writing-videos', timeout=0.5) as resp:
                        data = json.loads(resp.read().decode('utf-8'))
                        if isinstance(data, dict) and 'videos' in data:
                            writing_videos = set(data['videos'])
            except Exception as e:
                print(f"获取正在写入的视频列表失败: {e}")
            break
    
    if video_dir is None:
        video_dir = (BASE_DIR / 'videos').resolve()
    return video_dir, writing_videos


@app.route('/api/videos')
def api_videos():
    """获取视频列表API"""
    try:
        video_dir, writing_videos = resolve_video_directory_and_writing_videos()
        
        # 检查目录是否存在
        if not video_dir.exists():
            return jsonify({'videos': []})
        
        # 获取所有视频文件
        video_files = []
        current_time = time.time()
        for video_file in sorted(video_dir.glob('*.mp4'), key=lambda x: x.stat().st_mtime, reverse=True):
            try:
                stat = video_file.stat()
                size_mb = stat.st_size / (1024 * 1024)
                mtime = datetime.fromtimestamp(stat.st_mtime).strftime('%Y-%m-%d %H:%M:%S')
                
                # 检查是否正在写入（在writing_videos中，或者最后修改时间很近）
                is_writing = False
                video_path_str = str(video_file.resolve())
                if video_path_str in writing_videos:
                    is_writing = True
                else:
                    # 如果最后修改时间在30秒内，认为可能正在写入
                    time_since_modify = current_time - stat.st_mtime
                    if time_since_modify < 30:
                        is_writing = True
                
                video_files.append({
                    'filename': video_file.name,
                    'size_mb': round(size_mb, 2),
                    'modified_time': mtime,
                    'is_writing': is_writing
                })
            except Exception as e:
                print(f"处理视频文件 {video_file} 时出错: {e}")
                continue
        
        return jsonify({'videos': video_files})
    except Exception as e:
        return jsonify({'error': str(e)})


@app.route('/api/videos/<path:filename>')
def api_video_download(filename):
    """视频下载API"""
    from flask import send_file, abort
    from urllib.parse import unquote
    
    try:
        # URL解码文件名
        filename = unquote(filename)
        
        # 安全检查：防止路径遍历攻击
        if '..' in filename or '/' in filename or '\\' in filename:
            abort(400)
        
        video_dir, _ = resolve_video_directory_and_writing_videos()
        
        # 确保目录存在
        if not video_dir.exists():
            abort(404, "Video directory not found")
        
        video_path = video_dir / filename
        
        # 检查文件是否存在
        if not video_path.exists() or not video_path.is_file():
            abort(404, "Video not found")
        
        # 检查文件扩展名
        if not filename.lower().endswith('.mp4'):
            abort(400)
        
        # 检查是否正在写入（最后修改时间在30秒内）
        try:
            stat = video_path.stat()
            current_time = time.time()
            time_since_modify = current_time - stat.st_mtime
            if time_since_modify < 30:
                abort(409, "视频正在写入中，无法下载")  # 409 Conflict
        except:
            pass
        
        return send_file(
            str(video_path),
            mimetype='video/mp4',
            as_attachment=True,
            download_name=filename
        )
    except Exception as e:
        if hasattr(e, 'code'):
            raise
        print(f"视频下载错误: {str(e)}")
        abort(500)


@app.route('/api/videos/<path:filename>', methods=['DELETE'])
def api_video_delete(filename):
    """删除视频API"""
    try:
        # URL解码文件名
        from urllib.parse import unquote
        filename = unquote(filename)
        
        # 安全检查：防止路径遍历攻击
        if '..' in filename or '/' in filename or '\\' in filename:
            return jsonify({'success': False, 'message': '无效的文件名'}), 400
        
        video_dir, writing_videos = resolve_video_directory_and_writing_videos()
        
        # 确保目录存在
        if not video_dir.exists():
            return jsonify({'success': False, 'message': f'视频目录不存在: {video_dir}'}), 404
        
        video_path = video_dir / filename
        
        # 调试信息
        print(f"[DEBUG] 删除视频: filename={filename}, video_dir={video_dir}, video_path={video_path}, exists={video_path.exists()}")
        
        # 检查文件是否存在
        if not video_path.exists() or not video_path.is_file():
            # 尝试列出目录中的所有文件，用于调试
            try:
                existing_files = [f.name for f in video_dir.glob('*.mp4')]
                print(f"[DEBUG] 目录中的视频文件: {existing_files}")
            except:
                pass
            return jsonify({'success': False, 'message': f'文件不存在: {filename} (目录: {video_dir})'}), 404
        
        # 检查文件扩展名
        if not filename.lower().endswith('.mp4'):
            return jsonify({'success': False, 'message': '无效的文件类型'}), 400
        
        # 检查是否正在写入
        video_path_str = str(video_path.resolve())
        if video_path_str in writing_videos:
            return jsonify({'success': False, 'message': '视频正在写入中，无法删除'}), 409
        
        # 删除文件
        try:
            video_path.unlink()
            return jsonify({'success': True, 'message': f'已删除视频: {filename}'})
        except Exception as e:
            return jsonify({'success': False, 'message': f'删除失败: {str(e)}'}), 500
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)}), 500


@app.route('/api/videos/delete_all', methods=['POST'])
def api_video_delete_all():
    """删除所有非写入中的视频"""
    try:
        video_dir, writing_videos = resolve_video_directory_and_writing_videos()
        if not video_dir.exists():
            return jsonify({'success': True, 'message': '没有视频文件可删除'})
        
        deleted = 0
        skipped = 0
        for video_file in video_dir.glob('*.mp4'):
            try:
                video_path_str = str(video_file.resolve())
                if video_path_str in writing_videos:
                    skipped += 1
                    continue
                video_file.unlink()
                deleted += 1
            except Exception as e:
                print(f"删除视频 {video_file} 失败: {e}")
                skipped += 1
        
        message = f'已删除 {deleted} 个视频'
        if skipped > 0:
            message += f'，跳过 {skipped} 个正在写入或无法删除的视频'
        return jsonify({'success': True, 'message': message})
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)}), 500


def cleanup_on_exit():
    """退出时清理"""
    print("\n正在关闭管理器...")
    for key, service in SERVICES.items():
        if service['pid'] and get_process_status(service['pid']):
            print(f"保持 {service['name']} 运行 (PID: {service['pid']})")


if __name__ == '__main__':
    import atexit
    atexit.register(cleanup_on_exit)
    
    # 创建日志目录
    os.makedirs(str(LOGS_DIR), exist_ok=True)
    
    # 设置管理器日志
    import logging
    from datetime import datetime
    
    log_file = str(LOGS_DIR / 'manager.log')
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s',
        handlers=[
            logging.FileHandler(log_file, encoding='utf-8'),
            logging.StreamHandler()
        ]
    )
    
    logger = logging.getLogger('manager')
    logger.info("="*60)
    logger.info("算法服务管理器启动")
    logger.info("="*60)
    logger.info(f"管理界面: http://0.0.0.0:7900")
    logger.info(f"日志文件: {log_file}")
    logger.info("="*60)
    
    print("=" * 60)
    print("  算法服务管理器")
    print("=" * 60)
    print("\n🌐 管理界面: http://0.0.0.0:7900")
    print(f"📋 日志文件: {log_file}")
    print("\n等待连接... (按Ctrl+C退出)")
    print("=" * 60)
    
    try:
        app.run(host='0.0.0.0', port=7900, debug=False)
    except KeyboardInterrupt:
        logger.info("管理器已关闭")
        print("\n\n管理器已关闭")
        sys.exit(0)


