#!/usr/bin/env python3
"""
统一日志配置模块
为所有算法服务提供统一的日志管理
"""
import logging
import os
from logging.handlers import RotatingFileHandler
from datetime import datetime


class ColoredFormatter(logging.Formatter):
    """带颜色的日志格式化器（用于控制台）"""
    
    COLORS = {
        'DEBUG': '\033[36m',    # 青色
        'INFO': '\033[32m',     # 绿色
        'WARNING': '\033[33m',  # 黄色
        'ERROR': '\033[31m',    # 红色
        'CRITICAL': '\033[35m', # 紫色
        'RESET': '\033[0m'      # 重置
    }
    
    def format(self, record):
        log_color = self.COLORS.get(record.levelname, self.COLORS['RESET'])
        record.levelname = f"{log_color}{record.levelname}{self.COLORS['RESET']}"
        return super().format(record)


def setup_logger(service_name, log_dir='/cv_space/predict/logs', console_level=logging.INFO, file_level=logging.DEBUG):
    """
    设置日志记录器
    
    Args:
        service_name: 服务名称（用于日志文件名）
        log_dir: 日志目录
        console_level: 控制台日志级别
        file_level: 文件日志级别
    
    Returns:
        logger: 配置好的日志记录器
    """
    # 创建日志目录
    os.makedirs(log_dir, exist_ok=True)
    
    # 创建logger
    logger = logging.getLogger(service_name)
    logger.setLevel(logging.DEBUG)
    
    # 清除已有的处理器
    logger.handlers.clear()
    
    # 控制台处理器（带颜色）
    console_handler = logging.StreamHandler()
    console_handler.setLevel(console_level)
    console_formatter = ColoredFormatter(
        '%(asctime)s [%(levelname)s] %(message)s',
        datefmt='%H:%M:%S'
    )
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)
    
    # 文件处理器（带轮转，最大10MB，保留5个备份）
    log_file = os.path.join(log_dir, f'{service_name}.log')
    file_handler = RotatingFileHandler(
        log_file,
        maxBytes=10*1024*1024,  # 10MB
        backupCount=5,
        encoding='utf-8'
    )
    file_handler.setLevel(file_level)
    file_formatter = logging.Formatter(
        '%(asctime)s [%(levelname)s] [%(funcName)s:%(lineno)d] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)
    
    # 每日日志文件（用于归档）
    daily_log_file = os.path.join(log_dir, f'{service_name}_{datetime.now().strftime("%Y%m%d")}.log')
    daily_handler = logging.FileHandler(daily_log_file, encoding='utf-8')
    daily_handler.setLevel(file_level)
    daily_handler.setFormatter(file_formatter)
    logger.addHandler(daily_handler)
    
    logger.info(f"日志系统已初始化: {service_name}")
    logger.info(f"日志文件: {log_file}")
    logger.info(f"每日日志: {daily_log_file}")
    
    return logger


def get_log_stats(log_file):
    """
    获取日志文件统计信息
    
    Returns:
        dict: 日志统计信息
    """
    if not os.path.exists(log_file):
        return {
            'exists': False,
            'size': 0,
            'lines': 0,
            'last_modified': None
        }
    
    try:
        stat = os.stat(log_file)
        
        # 计算行数
        with open(log_file, 'r', encoding='utf-8', errors='ignore') as f:
            lines = sum(1 for _ in f)
        
        return {
            'exists': True,
            'size': stat.st_size,
            'size_mb': stat.st_size / (1024 * 1024),
            'lines': lines,
            'last_modified': datetime.fromtimestamp(stat.st_mtime).strftime('%Y-%m-%d %H:%M:%S')
        }
    except Exception as e:
        return {
            'exists': True,
            'error': str(e)
        }


def read_log_tail(log_file, lines=100):
    """
    读取日志文件最后N行
    
    Args:
        log_file: 日志文件路径
        lines: 读取行数
    
    Returns:
        list: 日志行列表
    """
    if not os.path.exists(log_file):
        return []
    
    try:
        with open(log_file, 'r', encoding='utf-8', errors='ignore') as f:
            all_lines = f.readlines()
            return [line.rstrip() for line in all_lines[-lines:]]
    except Exception as e:
        return [f'读取日志失败: {str(e)}']


def clear_old_logs(log_dir='/cv_space/predict/logs', days=7):
    """
    清理旧日志文件
    
    Args:
        log_dir: 日志目录
        days: 保留天数
    """
    if not os.path.exists(log_dir):
        return
    
    import time
    current_time = time.time()
    cutoff_time = current_time - (days * 24 * 60 * 60)
    
    deleted_count = 0
    for filename in os.listdir(log_dir):
        if not filename.endswith('.log'):
            continue
        
        file_path = os.path.join(log_dir, filename)
        if os.path.getmtime(file_path) < cutoff_time:
            try:
                os.remove(file_path)
                deleted_count += 1
                print(f"已删除旧日志: {filename}")
            except Exception as e:
                print(f"删除日志失败 {filename}: {str(e)}")
    
    if deleted_count > 0:
        print(f"共删除 {deleted_count} 个旧日志文件")
    
    return deleted_count

