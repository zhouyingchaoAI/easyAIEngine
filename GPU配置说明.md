# GPU 配置说明

## 概述

两个算法服务都支持配置运行在不同的GPU设备上，可以灵活分配GPU资源。

---

## 配置方式

### 1. 使用启动脚本配置（推荐）

#### 实时检测服务（默认GPU 3，端口8501）
```bash
# 使用默认GPU 3
./start_algorithm_service.sh

# 指定使用GPU 0
./start_algorithm_service.sh --gpu-id 0

# 指定使用GPU 1
./start_algorithm_service.sh --gpu-id 1

# 使用多个GPU（0和1）
./start_algorithm_service.sh --gpu-id "0,1"
```

#### 绊线统计服务（默认GPU 3，端口8502）
```bash
# 使用默认GPU 3
./start_line_crossing_service.sh

# 指定使用GPU 2
./start_line_crossing_service.sh --gpu-id 2

# 使用多个GPU（2和3）
./start_line_crossing_service.sh --gpu-id "2,3"
```

---

### 2. 直接使用Python命令配置

#### 实时检测服务
```bash
python3 algorithm_service.py --gpu-id 0 --port 8501
```

#### 绊线统计服务
```bash
python3 algorithm_service_line_crossing.py --gpu-id 1 --port 8502
```

---

## 常见使用场景

### 场景1：两个服务运行在同一个GPU上
```bash
# 实时检测服务 - GPU 0
./start_algorithm_service.sh --gpu-id 0 --port 8501

# 绊线统计服务 - GPU 0（同一个GPU）
./start_line_crossing_service.sh --gpu-id 0 --port 8502
```

**适用于：**
- 只有一块GPU
- GPU显存足够大
- 两个服务负载不高

---

### 场景2：两个服务运行在不同GPU上（推荐）
```bash
# 实时检测服务 - GPU 0
./start_algorithm_service.sh --gpu-id 0 --port 8501

# 绊线统计服务 - GPU 1
./start_line_crossing_service.sh --gpu-id 1 --port 8502
```

**适用于：**
- 有多块GPU
- 需要最大化性能
- 避免GPU资源竞争

---

### 场景3：单个服务使用多个GPU
```bash
# 实时检测服务 - 使用GPU 0和1
./start_algorithm_service.sh --gpu-id "0,1" --port 8501

# 绊线统计服务 - 使用GPU 2和3
./start_line_crossing_service.sh --gpu-id "2,3" --port 8502
```

**适用于：**
- 负载非常高
- 需要极致性能
- 有足够的GPU资源

---

### 场景4：默认配置（都使用GPU 3）
```bash
# 两个服务都使用默认GPU 3
./start_algorithm_service.sh       # GPU 3, 端口 8501
./start_line_crossing_service.sh   # GPU 3, 端口 8502
```

**适用于：**
- 快速测试
- GPU 3 显存和性能足够

---

## GPU选择建议

### 根据显存选择
```bash
# 查看GPU显存使用情况
nvidia-smi

# 选择显存充足的GPU
./start_algorithm_service.sh --gpu-id 1  # 假设GPU 1显存充足
```

### 根据负载选择
```bash
# 实时检测服务负载较高 → 使用性能更好的GPU
./start_algorithm_service.sh --gpu-id 0 --port 8501  # 最好的GPU

# 绊线统计服务负载较低 → 使用次要GPU
./start_line_crossing_service.sh --gpu-id 1 --port 8502  # 次要GPU
```

---

## 完整启动示例

### 示例1：生产环境（两块GPU）
```bash
# 实时检测服务 - GPU 0（主要业务）
./start_algorithm_service.sh \
  --gpu-id 0 \
  --port 8501 \
  --service-id realtime_detector \
  --easydarwin http://10.1.6.230:5066

# 绊线统计服务 - GPU 1（辅助业务）
./start_line_crossing_service.sh \
  --gpu-id 1 \
  --port 8502 \
  --service-id line_crossing_counter \
  --easydarwin http://10.1.6.230:5066
```

---

### 示例2：开发环境（单块GPU）
```bash
# 两个服务都使用GPU 0
./start_algorithm_service.sh --gpu-id 0 --port 8501 --no-register
./start_line_crossing_service.sh --gpu-id 0 --port 8502 --no-register
```

---

### 示例3：高性能环境（多GPU负载均衡）
```bash
# 实时检测服务 - GPU 0,1（高并发）
./start_algorithm_service.sh --gpu-id "0,1" --port 8501

# 绊线统计服务 - GPU 2,3（高并发）
./start_line_crossing_service.sh --gpu-id "2,3" --port 8502
```

---

## 验证GPU配置

### 启动后查看日志
```bash
# 实时检测服务启动日志
启动配置:
  服务ID: yolo11x_head_detector
  服务名称: YOLOv11x人头检测算法
  监听地址: 0.0.0.0:8501
  EasyDarwin: http://10.1.6.230:5066
  模型路径: /cv_space/predict/weight/best.pt
  GPU设备: 0  ← 确认GPU配置
  注册模式: 是

设置GPU设备: CUDA_VISIBLE_DEVICES=0  ← 确认环境变量
```

### 使用nvidia-smi监控
```bash
# 实时监控GPU使用情况
watch -n 1 nvidia-smi

# 查看指定进程的GPU使用
nvidia-smi | grep python
```

---

## 注意事项

### 1. GPU显存要求
- **YOLOv11x 模型** 大约需要 **4-6 GB** 显存
- 确保GPU有足够显存运行模型
- 如果GPU显存不足，会报 CUDA out of memory 错误

### 2. 多个服务同时运行
- 确保每个服务使用不同的**端口**
- 可以使用相同或不同的GPU
- 相同GPU可能会导致显存不足

### 3. GPU ID范围
- GPU ID从0开始
- 使用 `nvidia-smi` 查看可用的GPU列表
- 无效的GPU ID会导致启动失败

### 4. 环境变量
- 使用 `CUDA_VISIBLE_DEVICES` 环境变量控制GPU
- 启动脚本会自动设置此环境变量
- 服务启动后无法更改GPU配置，需重启

---

## 故障排查

### 问题1：CUDA out of memory
**原因：** GPU显存不足

**解决方案：**
```bash
# 方案1：使用其他GPU
./start_algorithm_service.sh --gpu-id 1

# 方案2：减小批处理大小
./start_algorithm_service.sh --gpu-id 0 --batch-size 4

# 方案3：停止其他占用GPU的进程
nvidia-smi
kill -9 <PID>
```

---

### 问题2：GPU未正确使用
**原因：** CUDA配置错误或GPU ID无效

**解决方案：**
```bash
# 检查CUDA是否可用
python3 -c "import torch; print(torch.cuda.is_available())"

# 检查可用GPU数量
python3 -c "import torch; print(torch.cuda.device_count())"

# 使用有效的GPU ID
nvidia-smi  # 查看可用GPU列表
./start_algorithm_service.sh --gpu-id 0  # 使用第一个GPU
```

---

### 问题3：两个服务冲突
**原因：** 端口冲突或GPU显存不足

**解决方案：**
```bash
# 确保使用不同端口
./start_algorithm_service.sh --gpu-id 0 --port 8501
./start_line_crossing_service.sh --gpu-id 1 --port 8502  # 不同端口

# 或使用不同GPU
./start_algorithm_service.sh --gpu-id 0 --port 8501
./start_line_crossing_service.sh --gpu-id 1 --port 8502  # 不同GPU
```

---

## 性能优化建议

### 1. GPU资源充足
- 每个服务使用独立GPU
- 最大化并发处理能力

### 2. GPU资源有限
- 两个服务共享GPU
- 适当降低批处理大小
- 错峰运行高负载任务

### 3. 极致性能
- 单个服务使用多GPU
- 启用批处理加速
- 优化网络带宽

---

## 命令速查表

| 场景 | 命令 |
|------|------|
| 默认配置 | `./start_algorithm_service.sh` |
| 指定GPU 0 | `./start_algorithm_service.sh --gpu-id 0` |
| 指定GPU 1 | `./start_algorithm_service.sh --gpu-id 1` |
| 多GPU (0,1) | `./start_algorithm_service.sh --gpu-id "0,1"` |
| 绊线GPU 2 | `./start_line_crossing_service.sh --gpu-id 2` |
| 查看GPU | `nvidia-smi` |
| 监控GPU | `watch -n 1 nvidia-smi` |

---

## 相关文档

- [服务说明.md](服务说明.md) - 服务功能详细说明
- [告警机制说明.md](告警机制说明.md) - 告警机制说明
- [配置参数说明.md](配置参数说明.md) - 配置文件参数说明


