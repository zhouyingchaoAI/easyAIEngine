#!/bin/bash

# 算法服务管理器启动脚本

echo "启动算法服务管理器..."
echo ""
echo "管理界面将在以下地址启动:"
echo "  http://localhost:7901"
echo "  http://0.0.0.0:7901"
echo ""
echo "功能:"
echo "  ✓ 启动/停止算法服务"
echo "  ✓ 实时监控GPU状态"
echo "  ✓ 配置GPU设备"
echo "  ✓ 调整算法参数"
echo "  ✓ 查看系统日志"
echo ""
echo "按 Ctrl+C 退出管理器"
echo ""

# 启动Flask服务
python3 algorithm_manager.py


