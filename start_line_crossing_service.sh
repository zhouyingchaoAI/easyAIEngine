#!/bin/bash

# YOLOv11x绊线人数统计算法服务启动脚本

# 默认配置
SERVICE_ID="yolo11x_line_crossing"
NAME="YOLOv11x绊线人数统计算法"
PORT=7903
HOST="0.0.0.0"
EASYDARWIN="http://10.1.6.230:5066"
MODEL="/cv_space/predict/weight/best.pt"
GPU_ID="3"
NO_REGISTER=false

# 显示帮助信息
show_help() {
    cat << EOF
YOLOv11x绊线人数统计算法服务启动脚本
专门用于绊线检测和跨线人数统计

用法:
    $0 [选项]

选项:
    -h, --help              显示帮助信息
    -i, --service-id ID     服务ID (默认: yolo11x_line_crossing)
    -n, --name NAME         服务名称 (默认: YOLOv11x绊线人数统计算法)
    -p, --port PORT         监听端口 (默认: 7903)
    -H, --host HOST         监听地址 (默认: 0.0.0.0)
    -e, --easydarwin URL    EasyDarwin地址 (默认: http://10.1.6.230:5066)
    -m, --model PATH        模型路径
    -g, --gpu-id ID         GPU设备ID (默认: 3, 可设置多个如 "0,1")
    --no-register           不注册到EasyDarwin

功能特性:
    - 人头检测: YOLOv11x 模型
    - 目标跟踪: 基于IOU的目标跟踪
    - 绊线检测: 自动统计跨线人数
    - 增量告警: 只在有新跨线时触发告警

示例:
    # 使用默认配置启动
    $0

    # 指定端口启动
    $0 --port 7903

    # 指定GPU设备
    $0 --gpu-id 0

    # 使用多个GPU
    $0 --gpu-id "0,1"

    # 不注册到EasyDarwin
    $0 --no-register

    # 完整自定义配置
    $0 --gpu-id 2 --port 9001

绊线配置:
    需要在图片目录下提供 algo_config.json 配置文件
    详细文档请参阅: 绊线增量告警说明.md

EOF
}

# 解析命令行参数
while [[ $# -gt 0 ]]; do
    case $1 in
        -h|--help)
            show_help
            exit 0
            ;;
        -i|--service-id)
            SERVICE_ID="$2"
            shift 2
            ;;
        -n|--name)
            NAME="$2"
            shift 2
            ;;
        -p|--port)
            PORT="$2"
            shift 2
            ;;
        -H|--host)
            HOST="$2"
            shift 2
            ;;
        -e|--easydarwin)
            EASYDARWIN="$2"
            shift 2
            ;;
        -m|--model)
            MODEL="$2"
            shift 2
            ;;
        -g|--gpu-id)
            GPU_ID="$2"
            shift 2
            ;;
        --no-register)
            NO_REGISTER=true
            shift
            ;;
        *)
            echo "未知选项: $1"
            echo "使用 --help 查看帮助信息"
            exit 1
            ;;
    esac
done

# 检查Python和依赖
if ! command -v python3 &> /dev/null; then
    echo "错误: 未找到python3"
    exit 1
fi

# 检查模型文件
if [ ! -f "$MODEL" ]; then
    echo "警告: 模型文件不存在: $MODEL"
    echo "请确认模型路径是否正确"
fi

# 构建命令
CMD="python3 algorithm_service_line_crossing.py"
CMD="$CMD --service-id '$SERVICE_ID'"
CMD="$CMD --name '$NAME'"
CMD="$CMD --port $PORT"
CMD="$CMD --host $HOST"
CMD="$CMD --easydarwin $EASYDARWIN"
CMD="$CMD --model '$MODEL'"
CMD="$CMD --gpu-id '$GPU_ID'"

if [ "$NO_REGISTER" = true ]; then
    CMD="$CMD --no-register"
fi

# 创建日志目录
LOG_DIR="/cv_space/predict/logs"
mkdir -p "$LOG_DIR"

# 日志文件路径
LOG_FILE="$LOG_DIR/line_crossing.log"

# 显示配置信息
echo "启动配置:"
echo "  服务ID: $SERVICE_ID"
echo "  服务名称: $NAME"
echo "  监听地址: $HOST:$PORT"
echo "  EasyDarwin: $EASYDARWIN"
echo "  模型路径: $MODEL"
echo "  GPU设备: $GPU_ID"
echo "  注册模式: $([ "$NO_REGISTER" = true ] && echo "否" || echo "是")"
echo "  日志文件: $LOG_FILE"
echo ""

# 写入启动标记到日志
echo "" >> "$LOG_FILE"
echo "==========================================================" >> "$LOG_FILE"
echo "服务启动: $(date '+%Y-%m-%d %H:%M:%S')" >> "$LOG_FILE"
echo "GPU: $GPU_ID, 端口: $PORT" >> "$LOG_FILE"
echo "==========================================================" >> "$LOG_FILE"

# 启动服务（输出同时到控制台和日志文件）
eval $CMD 2>&1 | tee -a "$LOG_FILE"

