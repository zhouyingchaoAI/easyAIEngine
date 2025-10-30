#!/bin/bash

# 日志管理脚本

LOG_DIR="/cv_space/predict/logs"

show_help() {
    cat << EOF
日志管理工具

用法:
    $0 [命令]

命令:
    view [service]      查看日志（实时）
    tail [service] [N]  查看最后N行（默认100行）
    clear [service]     清空日志
    stats               显示日志统计
    search <keyword>    搜索日志内容
    backup              备份所有日志
    help                显示帮助信息

服务名称:
    all         所有服务
    manager     Web管理界面
    realtime    实时检测服务
    line        绊线统计服务

示例:
    $0 view realtime           # 实时查看实时检测服务日志
    $0 tail line 200          # 查看绊线服务最后200行
    $0 clear realtime         # 清空实时检测日志
    $0 search "ERROR"         # 搜索所有ERROR
    $0 stats                  # 显示日志统计
    $0 backup                 # 备份所有日志

EOF
}

get_log_file() {
    local service=$1
    case $service in
        manager)
            echo "$LOG_DIR/manager.log"
            ;;
        realtime)
            echo "$LOG_DIR/realtime_detector.log"
            ;;
        line)
            echo "$LOG_DIR/line_crossing.log"
            ;;
        all)
            echo "$LOG_DIR/*.log"
            ;;
        *)
            echo "$LOG_DIR/*.log"
            ;;
    esac
}

cmd_view() {
    local service=${1:-all}
    local log_file=$(get_log_file $service)
    
    echo "实时查看日志: $log_file"
    echo "按 Ctrl+C 退出"
    echo "=========================================="
    
    tail -f $log_file
}

cmd_tail() {
    local service=${1:-all}
    local lines=${2:-100}
    local log_file=$(get_log_file $service)
    
    echo "查看最后 $lines 行: $log_file"
    echo "=========================================="
    
    tail -n $lines $log_file
}

cmd_clear() {
    local service=${1:-all}
    
    if [ "$service" = "all" ]; then
        read -p "确定要清空所有日志吗？(y/N): " confirm
        if [ "$confirm" != "y" ] && [ "$confirm" != "Y" ]; then
            echo "已取消"
            exit 0
        fi
        > "$LOG_DIR/manager.log"
        > "$LOG_DIR/realtime_detector.log"
        > "$LOG_DIR/line_crossing.log"
        echo "✓ 已清空所有日志"
    else
        local log_file=$(get_log_file $service)
        read -p "确定要清空 $log_file 吗？(y/N): " confirm
        if [ "$confirm" != "y" ] && [ "$confirm" != "Y" ]; then
            echo "已取消"
            exit 0
        fi
        > $log_file
        echo "✓ 已清空: $log_file"
    fi
}

cmd_stats() {
    echo "=========================================="
    echo "  日志统计"
    echo "=========================================="
    echo ""
    
    for log in "$LOG_DIR"/*.log; do
        if [ -f "$log" ]; then
            local filename=$(basename "$log")
            local size=$(du -h "$log" | cut -f1)
            local lines=$(wc -l < "$log")
            local errors=$(grep -c -i "error\|失败" "$log" 2>/dev/null || echo 0)
            local warnings=$(grep -c -i "warning\|警告" "$log" 2>/dev/null || echo 0)
            local modified=$(stat -c %y "$log" | cut -d'.' -f1)
            
            echo "📋 $filename"
            echo "  大小: $size"
            echo "  行数: $lines"
            echo "  错误: $errors"
            echo "  警告: $warnings"
            echo "  更新: $modified"
            echo ""
        fi
    done
}

cmd_search() {
    local keyword=$1
    
    if [ -z "$keyword" ]; then
        echo "请提供搜索关键词"
        exit 1
    fi
    
    echo "搜索: $keyword"
    echo "=========================================="
    
    grep -i --color=always -n "$keyword" "$LOG_DIR"/*.log
}

cmd_backup() {
    local backup_dir="$LOG_DIR/backups"
    mkdir -p "$backup_dir"
    
    local timestamp=$(date +%Y%m%d_%H%M%S)
    local backup_file="$backup_dir/logs_backup_$timestamp.tar.gz"
    
    echo "备份日志到: $backup_file"
    
    tar -czf "$backup_file" -C "$LOG_DIR" \
        --exclude="backups" \
        --exclude="archive" \
        *.log 2>/dev/null
    
    if [ $? -eq 0 ]; then
        echo "✓ 备份成功: $backup_file"
        echo "  大小: $(du -h "$backup_file" | cut -f1)"
    else
        echo "✗ 备份失败"
        exit 1
    fi
}

# 主程序
case ${1:-help} in
    view)
        cmd_view $2
        ;;
    tail)
        cmd_tail $2 $3
        ;;
    clear)
        cmd_clear $2
        ;;
    stats)
        cmd_stats
        ;;
    search)
        cmd_search "$2"
        ;;
    backup)
        cmd_backup
        ;;
    help|--help|-h)
        show_help
        ;;
    *)
        echo "未知命令: $1"
        echo "使用 '$0 help' 查看帮助"
        exit 1
        ;;
esac

