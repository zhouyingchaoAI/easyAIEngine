#!/bin/bash

# 绊线告警逻辑验证脚本

echo "=========================================="
echo "  绊线告警逻辑验证"
echo "=========================================="
echo ""

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
LOG_FILE="${PROJECT_ROOT}/logs/line_crossing.log"

# 检查服务是否运行
echo "1. 检查服务状态..."
if curl -s http://localhost:7903/health > /dev/null 2>&1; then
    echo "   ✓ 绊线统计服务正在运行（端口 7903）"
else
    echo "   ✗ 绊线统计服务未运行"
    echo "   请先启动: ./scripts/start_line_crossing_service.sh"
    exit 1
fi
echo ""

# 检查日志文件
echo "2. 检查日志文件..."
if [ -f "$LOG_FILE" ]; then
    LINES=$(wc -l < "$LOG_FILE")
    echo "   ✓ 日志文件存在: $LOG_FILE"
    echo "   行数: $LINES"
else
    echo "   ✗ 日志文件不存在"
    exit 1
fi
echo ""

# 查看最近的穿越事件
echo "3. 最近的穿越事件："
echo "=========================================="
grep -E "检测到新穿越|无新穿越" "$LOG_FILE" | tail -10
echo "=========================================="
echo ""

# 统计告警次数
ALARM_COUNT=$(grep -c "检测到新穿越" "$LOG_FILE")
NO_ALARM_COUNT=$(grep -c "无新穿越" "$LOG_FILE")

echo "4. 统计信息："
echo "   触发告警次数: $ALARM_COUNT"
echo "   未触发次数: $NO_ALARM_COUNT"
echo "   总推理次数: $((ALARM_COUNT + NO_ALARM_COUNT))"
if [ $((ALARM_COUNT + NO_ALARM_COUNT)) -gt 0 ]; then
    ALARM_RATE=$((ALARM_COUNT * 100 / (ALARM_COUNT + NO_ALARM_COUNT)))
    echo "   告警率: ${ALARM_RATE}%"
fi
echo ""

# 显示累积计数变化
echo "5. 累积计数变化历史："
echo "=========================================="
grep "检测到新穿越" "$LOG_FILE" | tail -20 | \
  sed 's/.*检测到新穿越: //' | \
  while read line; do
    echo "   $line"
  done
echo "=========================================="
echo ""

echo "验证完成！"
echo ""
echo "说明："
echo "  ✓ 检测到新穿越 → 累积计数增加 → person_count有值 → 触发告警"
echo "  ℹ️ 无新穿越     → 累积计数不变 → 无person_count → 不触发告警"
echo ""
