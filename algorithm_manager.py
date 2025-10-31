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

app = Flask(__name__)
CORS(app)

# 全局变量存储服务进程
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
    }
}

# HTML模板
HTML_TEMPLATE = '''
<!DOCTYPE html>
<html>
<head>
    <title>算法服务管理器</title>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            padding: 20px;
        }
        .container {
            max-width: 1400px;
            margin: 0 auto;
        }
        .header {
            background: white;
            border-radius: 12px;
            padding: 30px;
            margin-bottom: 20px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        }
        .header h1 {
            color: #2d3748;
            margin-bottom: 10px;
            font-size: 32px;
        }
        .header p {
            color: #718096;
            font-size: 16px;
        }
        .grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(600px, 1fr));
            gap: 20px;
            margin-bottom: 20px;
        }
        .card {
            background: white;
            border-radius: 12px;
            padding: 25px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        }
        .card-title {
            font-size: 20px;
            font-weight: 600;
            color: #2d3748;
            margin-bottom: 20px;
            padding-bottom: 15px;
            border-bottom: 2px solid #e2e8f0;
        }
        .status-badge {
            display: inline-block;
            padding: 6px 16px;
            border-radius: 20px;
            font-size: 14px;
            font-weight: 600;
            margin-left: 10px;
        }
        .status-running { background: #48bb78; color: white; }
        .status-stopped { background: #cbd5e0; color: #4a5568; }
        .form-group {
            margin-bottom: 20px;
        }
        .form-group label {
            display: block;
            color: #4a5568;
            font-weight: 600;
            margin-bottom: 8px;
            font-size: 14px;
        }
        .form-group input, .form-group select {
            width: 100%;
            padding: 12px;
            border: 2px solid #e2e8f0;
            border-radius: 8px;
            font-size: 14px;
            transition: border-color 0.3s;
        }
        .form-group input:focus, .form-group select:focus {
            outline: none;
            border-color: #667eea;
        }
        .btn {
            padding: 12px 24px;
            border: none;
            border-radius: 8px;
            font-size: 14px;
            font-weight: 600;
            cursor: pointer;
            transition: all 0.3s;
            margin-right: 10px;
        }
        .btn-primary {
            background: #667eea;
            color: white;
        }
        .btn-primary:hover { background: #5568d3; transform: translateY(-2px); }
        .btn-danger {
            background: #f56565;
            color: white;
        }
        .btn-danger:hover { background: #e53e3e; transform: translateY(-2px); }
        .btn-success {
            background: #48bb78;
            color: white;
        }
        .btn-success:hover { background: #38a169; }
        .btn:disabled {
            opacity: 0.5;
            cursor: not-allowed;
        }
        .gpu-card {
            background: #f7fafc;
            border: 2px solid #e2e8f0;
            border-radius: 8px;
            padding: 15px;
            margin-bottom: 15px;
        }
        .gpu-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 10px;
        }
        .gpu-name {
            font-weight: 600;
            color: #2d3748;
            font-size: 16px;
        }
        .gpu-id {
            background: #667eea;
            color: white;
            padding: 4px 12px;
            border-radius: 12px;
            font-size: 12px;
        }
        .progress-bar {
            width: 100%;
            height: 24px;
            background: #e2e8f0;
            border-radius: 12px;
            overflow: hidden;
            margin-bottom: 8px;
        }
        .progress-fill {
            height: 100%;
            background: linear-gradient(90deg, #48bb78 0%, #38a169 100%);
            display: flex;
            align-items: center;
            justify-content: center;
            color: white;
            font-size: 12px;
            font-weight: 600;
            transition: width 0.3s;
        }
        .progress-fill.warning { background: linear-gradient(90deg, #ed8936 0%, #dd6b20 100%); }
        .progress-fill.danger { background: linear-gradient(90deg, #f56565 0%, #e53e3e 100%); }
        .gpu-info {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 10px;
            font-size: 13px;
            color: #4a5568;
        }
        .info-item {
            display: flex;
            justify-content: space-between;
        }
        .info-label { font-weight: 500; }
        .info-value { font-weight: 600; color: #2d3748; }
        .log-container {
            background: #1a202c;
            color: #e2e8f0;
            padding: 15px;
            border-radius: 8px;
            font-family: 'Courier New', monospace;
            font-size: 12px;
            max-height: 400px;
            overflow-y: auto;
            line-height: 1.8;
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
            background: #f7fafc;
            padding: 15px;
            border-radius: 8px;
            margin-bottom: 20px;
        }
        .service-info-item {
            display: flex;
            justify-content: space-between;
            padding: 8px 0;
            border-bottom: 1px solid #e2e8f0;
        }
        .service-info-item:last-child { border-bottom: none; }
        .refresh-btn {
            background: #4299e1;
            color: white;
            padding: 8px 16px;
            border: none;
            border-radius: 6px;
            cursor: pointer;
            font-size: 12px;
            margin-top: 10px;
        }
        .refresh-btn:hover { background: #3182ce; }
        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }
        .loading {
            animation: spin 1s linear infinite;
            display: inline-block;
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
        <div class="card">
            <div class="card-title">
                💻 GPU 状态监控
                <button class="refresh-btn" onclick="loadGPUInfo()">🔄 刷新</button>
            </div>
            <div id="gpu-info">加载中...</div>
        </div>

        <!-- 服务管理（仅实时检测） -->
        <div class="grid">
            <div class="card">
                <div class="card-title">
                    🔴 实时检测服务
                    <span id="realtime-status" class="status-badge status-stopped">已停止</span>
                </div>
                
                <div class="service-info">
                    <div class="service-info-item">
                        <span>任务类型</span>
                        <strong>人数统计、客流分析、人头检测</strong>
                    </div>
                    <div class="service-info-item">
                        <span>设备</span>
                        <strong>Ascend NPU（可多实例分配至不同 device_id）</strong>
                    </div>
                </div>

                <div class="form-group">
                    <label>服务ID前缀（批量实例自动递增）</label>
                    <input type="text" id="realtime-service-prefix-input" value="yolo11x_head_detector" placeholder="例如: yolo11x_head_detector">
                </div>
                <div class="form-group">
                    <label>实例数量</label>
                    <input type="number" id="realtime-count-input" value="1" min="1" placeholder="要启动的实例个数">
                </div>
                <div class="form-group">
                    <label>设备列表（device_id）</label>
                    <input type="text" id="realtime-devices-input" value="0" placeholder="例如: 0,1,0 表示按顺序分配">
                </div>
                <div class="form-group">
                    <label>批处理大小</label>
                    <input type="number" id="realtime-batch-input" value="8">
                </div>
                <div class="form-group">
                    <label>端口（可选，0=自动分配 7901-7999）</label>
                    <input type="number" id="realtime-port-input" value="0" placeholder="0=自动分配(7901-7999)">
                </div>
                <div class="form-group">
                    <label>推理端点IP（可选，默认为172.17.0.2）</label>
                    <input type="text" id="realtime-infer-ip-input" value="172.17.0.2" placeholder="例如: 172.17.0.2 或 10.1.6.230">
                </div>
                
                <button class="btn btn-success" onclick="startService('realtime')">▶️ 批量新增实例</button>
                <button class="btn btn-danger" onclick="stopService('realtime')">⏹️ 停止全部实例</button>

                <div style="margin-top:15px;">
                    <div class="card-title" style="border:none;padding:0;margin:10px 0 5px 0;">实例列表</div>
                    <div id="realtime-instances">暂无实例</div>
                </div>
            </div>
        </div>

        <!-- 系统日志 -->
        <div class="card">
            <div class="card-title">
                📋 系统日志
                <select id="log-service" onchange="loadLogs()" style="margin-left: 10px; padding: 6px 12px; border-radius: 6px; border: 2px solid #e2e8f0;">
                    <option value="all">全部日志</option>
                    <option value="manager">管理器日志</option>
                    <option value="realtime">实时检测日志</option>
                    <option value="line_crossing">绊线统计日志</option>
                </select>
                <select id="log-lines" onchange="loadLogs()" style="margin-left: 10px; padding: 6px 12px; border-radius: 6px; border: 2px solid #e2e8f0;">
                    <option value="50">50行</option>
                    <option value="100" selected>100行</option>
                    <option value="200">200行</option>
                    <option value="500">500行</option>
                </select>
                <button class="refresh-btn" onclick="loadLogs()">🔄 刷新</button>
                <button class="refresh-btn" onclick="clearLogs()" style="background: #f56565;">🗑️ 清空</button>
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

        // 页面加载时初始化
        window.onload = function() {
            loadGPUInfo();
            loadServiceStatus();
            loadLogs();
            
            // 自动刷新
            setInterval(() => {
                if (autoRefresh) {
                    loadGPUInfo();
                    loadServiceStatus();
                }
            }, REFRESH_INTERVAL);
        };

        // 加载GPU信息
        async function loadGPUInfo() {
            try {
                const response = await fetch('/api/gpu-info');
                const data = await response.json();
                
                let html = '';
                if (data.gpus && data.gpus.length > 0) {
                    data.gpus.forEach(gpu => {
                        const usage = gpu.memory_used_percent || 0;
                        let progressClass = '';
                        if (usage > 80) progressClass = 'danger';
                        else if (usage > 60) progressClass = 'warning';
                        
                        html += `
                            <div class="gpu-card">
                                <div class="gpu-header">
                                    <div class="gpu-name">🎮 ${gpu.name || 'GPU'}</div>
                                    <div class="gpu-id">GPU ${gpu.id}</div>
                                </div>
                                <div class="progress-bar">
                                    <div class="progress-fill ${progressClass}" style="width: ${usage}%">
                                        ${usage.toFixed(1)}%
                                    </div>
                                </div>
                                <div class="gpu-info">
                                    <div class="info-item">
                                        <span class="info-label">显存使用</span>
                                        <span class="info-value">${gpu.memory_used} / ${gpu.memory_total}</span>
                                    </div>
                                    <div class="info-item">
                                        <span class="info-label">AICore 利用率</span>
                                        <span class="info-value">${gpu.utilization || 'N/A'}%</span>
                                    </div>
                                    <div class="info-item">
                                        <span class="info-label">温度</span>
                                        <span class="info-value">${gpu.temperature || 'N/A'}°C</span>
                                    </div>
                                    <div class="info-item">
                                        <span class="info-label">功率</span>
                                        <span class="info-value">${gpu.power || 'N/A'} W</span>
                                    </div>
                                    <div class="info-item">
                                        <span class="info-label">健康</span>
                                        <span class="info-value">${gpu.health || 'N/A'}</span>
                                    </div>
                                </div>
                            </div>
                        `;
                    });
                } else {
                    html = '<p style="color: #718096;">无法获取设备信息（npu-smi / nvidia-smi 不可用）</p>';
                }
                
                document.getElementById('gpu-info').innerHTML = html;
            } catch (error) {
                console.error('加载GPU信息失败:', error);
            }
        }

        // 渲染实例列表
        function renderInstances(serviceKey, instances) {
            const container = document.getElementById(`${serviceKey}-instances`);
            if (!container) return;
            if (!instances || instances.length === 0) {
                container.innerHTML = '<p style="color:#718096;">暂无实例</p>';
                return;
            }
            const rows = instances.map(ins => {
                const count = (ins.stats && ins.stats.total_requests != null) ? ins.stats.total_requests : '-';
                const inferIp = ins.config.infer_ip || '172.17.0.2';
                const inferUrl = `http://${inferIp}:${ins.config.port}/infer`;
                return `
                <div class="gpu-card" style="padding:10px;margin-bottom:8px;">
                    <div style="display:flex;gap:12px;flex-wrap:wrap;align-items:center;margin-bottom:8px;">
                        <span class="info-label">PID</span><span class="info-value">${ins.pid || '-'}</span>
                        <span class="info-label">端口</span><span class="info-value">${ins.config.port}</span>
                        <span class="info-label">GPU</span><span class="info-value">${ins.config.device_id || '-'}</span>
                        <span class="info-label">服务ID</span><span class="info-value">${ins.config.service_id || '-'}</span>
                        <span class="info-label">累计推理</span><span class="info-value">${count}</span>
                        <button class="btn btn-danger" style="margin-left:auto;" onclick="stopInstance('${serviceKey}', ${ins.pid})">⏹️ 停止</button>
                    </div>
                    <div style="font-size:12px;color:#4a5568;padding-top:8px;border-top:1px solid #e2e8f0;">
                        <strong>推理端点:</strong> <code style="background:#f7fafc;padding:2px 6px;border-radius:4px;">${inferUrl}</code>
                    </div>
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
            const inferIp = document.getElementById(`${serviceKey}-infer-ip-input`).value || '172.17.0.2';
            const servicePrefix = document.getElementById(`${serviceKey}-service-prefix-input`).value || 'yolo11x_head_detector';
            
            try {
                const response = await fetch('/api/start-service', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                        service: serviceKey,
                        count: count,
                        device_ids: devices,
                        port: parseInt(port),
                        batch_size: parseInt(batchSize),
                        infer_ip: inferIp,
                        service_id_prefix: servicePrefix
                    })
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
    result = {}
    
    for key, service in SERVICES.items():
        instances_out = []
        alive_instances = []
        # 清理死亡实例并收集状态
        for ins in service.get('instances', []):
            pid = ins.get('pid')
            if pid and get_process_status(pid):
                # 查询实例统计
                stats = None
                try:
                    port = ins.get('config', {}).get('port')
                    if port:
                        with urllib.request.urlopen(f'http://127.0.0.1:{port}/stats', timeout=0.5) as resp:
                            data = json.loads(resp.read().decode('utf-8'))
                            if isinstance(data, dict):
                                if 'statistics' in data and isinstance(data['statistics'], dict):
                                    stats = {'total_requests': data['statistics'].get('total_requests')}
                                elif 'total_requests' in data:
                                    stats = {'total_requests': data.get('total_requests')}
                except Exception:
                    stats = None
                ins['stats'] = stats
                alive_instances.append(ins)
                instances_out.append({
                    'pid': pid,
                    'config': ins.get('config', {}),
                    'stats': stats
                })
        # 覆盖为存活实例
        service['instances'] = alive_instances
        
        result[key] = {
            'name': service['name'],
            'instances': instances_out
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
    infer_ip = data.get('infer_ip', '172.17.0.2')  # 推理端点IP，默认为172.17.0.2
    service_id_prefix = data.get('service_id_prefix', 'yolo11x_head_detector')
    
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
        log_dir = '/cv_space/predict/logs'
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

            # 构建启动命令（Ascend: 通过 --device-id 传入）
            cmd = [
                'python3',
                service['script'],
                '--service-id', f"{service_id_prefix}_{inst_port}",
                '--port', str(inst_port),
                '--device-id', str(device_id),
                '--easydarwin', 'http://10.1.6.230:5066',
                '--host-ip', infer_ip  # 传递推理端点IP给服务，用于注册到EasyDarwin
            ]

            # 记录日志标记
            log_handle.write(f"\n{'='*60}\n")
            log_handle.write(f"服务启动: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            log_handle.write(f"DEVICE: {device_id}, 端口: {inst_port}, 批处理: {batch_size}\n")
            log_handle.write(f"服务ID: {service_id_prefix}_{inst_port}\n")
            log_handle.write(f"推理端点IP: {infer_ip} (用于注册到EasyDarwin)\n")
            log_handle.write(f"{'='*60}\n")
            log_handle.flush()

            process = subprocess.Popen(
                cmd,
                cwd='/code/predict',
                stdout=log_handle,
                stderr=subprocess.STDOUT,
                start_new_session=True
            )

            time.sleep(0.5)
            if process.poll() is None:
                instance = {
                    'process': process,
                    'pid': process.pid,
                    'config': {
                        'device_id': device_id,
                        'port': inst_port,
                        'batch_size': batch_size,
                        'infer_ip': infer_ip  # 保存推理端点IP
                    },
                    'stats': None
                }
                service.setdefault('instances', []).append(instance)
                started.append({'pid': process.pid, 'port': inst_port, 'device_id': device_id, 'infer_ip': infer_ip, 'service_id': f"{service_id_prefix}_{inst_port}"})

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
        log_dir = '/cv_space/predict/logs'
        
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
        log_dir = '/cv_space/predict/logs'
        
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
        log_dir = '/cv_space/predict/logs'
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
    os.makedirs('/cv_space/predict/logs', exist_ok=True)
    
    # 设置管理器日志
    import logging
    from datetime import datetime
    
    log_file = f'/cv_space/predict/logs/manager.log'
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


