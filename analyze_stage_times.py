#!/usr/bin/env python3
"""
分析推理各阶段耗时统计脚本
从日志中提取各阶段耗时数据并生成统计报告
"""

import sys
import json
import re
from collections import defaultdict
from datetime import datetime

def parse_log_line(line):
    """解析日志行，提取性能数据"""
    # 提取时间戳 - 尝试多种格式
    time_match = re.search(r'\[(\d{2}/\w{3}/\d{4} \d{2}):(\d{2}):(\d{2})\]', line)
    
    hour_min = None
    if time_match:
        hour_min = f'{time_match.group(1)}:{time_match.group(2)}'
    else:
        # 如果没有时间戳，跳过这一行
        return None
    
    # 提取JSON - 查找"返回告警JSON: "
    json_start = line.find('返回告警JSON: ')
    if json_start == -1:
        return None
    
    json_str = line[json_start + len('返回告警JSON: '):].strip()
    # 移除可能的重复部分（如果同一行有多个JSON）
    if '[' in json_str:
        # 找到第一个完整的JSON对象
        brace_count = 0
        end_pos = -1
        for i, char in enumerate(json_str):
            if char == '{':
                brace_count += 1
            elif char == '}':
                brace_count -= 1
                if brace_count == 0:
                    end_pos = i + 1
                    break
        if end_pos > 0:
            json_str = json_str[:end_pos]
    
    try:
        data = json.loads(json_str)
        return {
            'time': hour_min,
            'data': data
        }
    except json.JSONDecodeError as e:
        # 如果JSON解析失败，尝试用正则表达式提取关键字段
        data = {}
        
        # 提取基础字段
        total_match = re.search(r'"total_time_ms":\s*([0-9.]+)', json_str)
        if total_match:
            data['total_time_ms'] = float(total_match.group(1))
        
        infer_match = re.search(r'"inference_time_ms":\s*([0-9.]+)', json_str)
        if infer_match:
            data['inference_time_ms'] = float(infer_match.group(1))
        
        # 尝试提取stage_times
        stage_times_match = re.search(r'"stage_times":\s*(\{[^}]+\})', json_str)
        if stage_times_match:
            try:
                stage_times_str = stage_times_match.group(1)
                # 尝试解析stage_times对象
                stage_times = {}
                for stage_match in re.finditer(r'"(\w+)":\s*([0-9.]+)', stage_times_str):
                    stage_times[stage_match.group(1)] = float(stage_match.group(2))
                if stage_times:
                    data['stage_times'] = stage_times
            except:
                pass
        
        if data:
            return {
                'time': hour_min,
                'data': data
            }
    
    return None

def analyze_stage_times(log_file, num_lines=10000):
    """分析日志中的各阶段耗时"""
    stage_data = defaultdict(list)
    time_groups = defaultdict(lambda: defaultdict(list))
    basic_stats = defaultdict(list)  # 基础统计数据（inference_time_ms, total_time_ms）
    
    print(f"正在分析日志文件: {log_file}")
    print(f"分析最近 {num_lines} 行...")
    
    with open(log_file, 'r', encoding='utf-8', errors='ignore') as f:
        lines = f.readlines()
        recent_lines = lines[-num_lines:] if len(lines) > num_lines else lines
        
        for line in recent_lines:
            parsed = parse_log_line(line)
            if parsed:
                hour_min = parsed['time']
                data = parsed['data']
                
                # 检查是否有详细的stage_times
                if 'stage_times' in data and isinstance(data['stage_times'], dict):
                    stage_times = data['stage_times']
                    for stage, time_ms in stage_times.items():
                        if isinstance(time_ms, (int, float)) and time_ms > 0:
                            stage_data[stage].append(time_ms)
                            time_groups[hour_min][stage].append(time_ms)
                
                # 同时收集基础统计数据
                if 'inference_time_ms' in data:
                    basic_stats['inference_time_ms'].append(data['inference_time_ms'])
                if 'total_time_ms' in data:
                    basic_stats['total_time_ms'].append(data['total_time_ms'])
    
    # 如果有详细的stage_times数据，使用它
    if stage_data:
        print("\n✓ 找到详细的各阶段耗时数据")
    elif basic_stats:
        print("\n⚠️  未找到详细的stage_times字段，但找到了基础性能数据")
        print("提示: 服务需要重启才能记录详细的各阶段耗时")
        print("\n当前可用的基础性能统计:")
        print("-" * 60)
        for key, values in basic_stats.items():
            if values:
                avg = sum(values) / len(values)
                min_val = min(values)
                max_val = max(values)
                median = sorted(values)[len(values) // 2]
                print(f"{key}:")
                print(f"  样本数: {len(values)}")
                print(f"  平均值: {avg:.2f}ms")
                print(f"  最小值: {min_val:.2f}ms")
                print(f"  最大值: {max_val:.2f}ms")
                print(f"  中位数: {median:.2f}ms")
        print("\n建议:")
        print("1. 重启算法服务以启用详细的各阶段耗时统计")
        print("2. 重启后，新的请求响应将包含stage_times字段")
        print("3. 然后可以重新运行此脚本查看详细统计")
        return
    else:
        print("\n未找到任何性能数据")
        print("提示: 请确保日志文件路径正确，且日志中包含性能数据")
        return
    
    # 生成统计报告
    print("\n" + "=" * 80)
    print("各阶段耗时统计报告")
    print("=" * 80)
    
    # 整体统计
    print("\n【整体统计】")
    print("-" * 80)
    print(f"{'阶段':<25} {'样本数':<10} {'平均值(ms)':<15} {'最小值(ms)':<15} {'最大值(ms)':<15} {'中位数(ms)':<15}")
    print("-" * 80)
    
    stage_order = [
        'download_ms', 'read_image_ms', 'preprocess_ms', 'memcopy_ms',
        'inference_ms', 'output_ms', 'postprocess_ms', 'region_filter_ms',
        'total_inference_ms', 'total_ms'
    ]
    
    for stage in stage_order:
        if stage in stage_data:
            times = stage_data[stage]
            avg = sum(times) / len(times)
            min_val = min(times)
            max_val = max(times)
            median = sorted(times)[len(times) // 2]
            print(f"{stage:<25} {len(times):<10} {avg:<15.2f} {min_val:<15.2f} {max_val:<15.2f} {median:<15.2f}")
    
    # 按时间段统计
    if time_groups:
        print("\n【按时间段统计】(最近10个时间段)")
        print("-" * 80)
        for hour_min in sorted(time_groups.keys())[-10:]:
            print(f"\n时间段: {hour_min}")
            print(f"{'阶段':<25} {'平均值(ms)':<15} {'最大值(ms)':<15}")
            print("-" * 50)
            for stage in stage_order:
                if stage in time_groups[hour_min]:
                    times = time_groups[hour_min][stage]
                    avg = sum(times) / len(times)
                    max_val = max(times)
                    print(f"{stage:<25} {avg:<15.2f} {max_val:<15.2f}")
    
    # 性能占比分析
    print("\n【各阶段耗时占比分析】")
    print("-" * 80)
    if 'total_ms' in stage_data:
        total_avg = sum(stage_data['total_ms']) / len(stage_data['total_ms'])
        print(f"总平均耗时: {total_avg:.2f}ms")
        print(f"\n各阶段占比:")
        for stage in stage_order:
            if stage in stage_data and stage != 'total_ms':
                stage_avg = sum(stage_data[stage]) / len(stage_data[stage])
                percentage = (stage_avg / total_avg * 100) if total_avg > 0 else 0
                print(f"  {stage:<25} {stage_avg:>8.2f}ms ({percentage:>5.2f}%)")
    
    # 性能瓶颈识别
    print("\n【性能瓶颈识别】")
    print("-" * 80)
    if 'total_ms' in stage_data:
        total_avg = sum(stage_data['total_ms']) / len(stage_data['total_ms'])
        bottlenecks = []
        for stage in stage_order:
            if stage in stage_data and stage != 'total_ms':
                stage_avg = sum(stage_data[stage]) / len(stage_data[stage])
                percentage = (stage_avg / total_avg * 100) if total_avg > 0 else 0
                if percentage > 20:  # 占比超过20%认为是瓶颈
                    bottlenecks.append((stage, stage_avg, percentage))
        
        if bottlenecks:
            bottlenecks.sort(key=lambda x: x[2], reverse=True)
            print("主要性能瓶颈:")
            for stage, avg, pct in bottlenecks:
                print(f"  ⚠️  {stage}: 平均 {avg:.2f}ms, 占比 {pct:.2f}%")
        else:
            print("未发现明显的性能瓶颈（各阶段占比相对均衡）")

if __name__ == '__main__':
    log_file = '/code/predict/logs/realtime_detector.log'
    num_lines = 10000
    
    if len(sys.argv) > 1:
        log_file = sys.argv[1]
    if len(sys.argv) > 2:
        num_lines = int(sys.argv[2])
    
    analyze_stage_times(log_file, num_lines)

