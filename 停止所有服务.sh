#!/bin/bash

echo "正在停止所有算法服务..."

# 停止所有相关进程
pkill -f "algorithm_service.py"
pkill -f "algorithm_service_line_crossing.py"
pkill -f "algorithm_manager.py"

sleep 1

# 检查是否还有残留进程
REMAINING=$(ps aux | grep -E "algorithm_service|algorithm_manager" | grep -v grep | wc -l)

if [ $REMAINING -eq 0 ]; then
    echo "✓ 所有服务已停止"
else
    echo "⚠️  仍有 $REMAINING 个进程在运行"
    echo ""
    ps aux | grep -E "algorithm_service|algorithm_manager" | grep -v grep
    echo ""
    read -p "是否强制杀死？(y/N): " confirm
    if [ "$confirm" = "y" ] || [ "$confirm" = "Y" ]; then
        pkill -9 -f "algorithm_service"
        pkill -9 -f "algorithm_manager"
        echo "✓ 已强制停止所有进程"
    fi
fi
