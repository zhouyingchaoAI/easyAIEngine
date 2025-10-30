#!/usr/bin/env python3
"""
批处理推理并发测试脚本
测试算法服务的批处理和并发性能
"""
import requests
import time
import json
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import statistics


def test_single_inference(base_url, image_url, task_id="test_batch"):
    """
    执行单次推理测试
    返回: (success, inference_time_ms, total_time_ms, result)
    """
    start_time = time.time()
    
    payload = {
        "image_url": image_url,
        "task_id": task_id,
        "task_type": "人数统计",
        "image_path": f"test/{task_id}/test.jpg"
    }
    
    try:
        response = requests.post(
            f"{base_url}/infer",
            json=payload,
            timeout=30
        )
        total_time = (time.time() - start_time) * 1000
        
        if response.status_code == 200:
            data = response.json()
            if data.get('success'):
                return True, data.get('inference_time_ms', 0), total_time, data
            else:
                return False, 0, total_time, {'error': data.get('error', 'Unknown error')}
        else:
            return False, 0, total_time, {'error': f'HTTP {response.status_code}'}
    except Exception as e:
        total_time = (time.time() - start_time) * 1000
        return False, 0, total_time, {'error': str(e)}


def test_concurrent_inference(base_url, image_url, num_requests=10, num_workers=5):
    """
    测试并发推理
    
    Args:
        base_url: 算法服务URL
        image_url: 测试图片URL
        num_requests: 总请求数
        num_workers: 并发工作线程数
    """
    print(f"\n{'='*70}")
    print(f"并发推理测试")
    print(f"{'='*70}")
    print(f"  服务URL: {base_url}")
    print(f"  图片URL: {image_url[:80]}...")
    print(f"  总请求数: {num_requests}")
    print(f"  并发数: {num_workers}")
    print(f"{'='*70}\n")
    
    start_time = time.time()
    
    # 使用线程池执行并发请求
    results = []
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = []
        for i in range(num_requests):
            future = executor.submit(
                test_single_inference, 
                base_url, 
                image_url, 
                f"test_batch_{i}"
            )
            futures.append(future)
        
        # 收集结果
        for i, future in enumerate(as_completed(futures), 1):
            success, inference_time, total_time, result = future.result()
            results.append({
                'success': success,
                'inference_time_ms': inference_time,
                'total_time_ms': total_time,
                'result': result
            })
            
            if success:
                print(f"  ✓ 请求 {i}/{num_requests} 完成: 推理={inference_time:.0f}ms, 总计={total_time:.0f}ms")
            else:
                print(f"  ✗ 请求 {i}/{num_requests} 失败: {result.get('error', 'Unknown')}")
    
    total_elapsed = (time.time() - start_time) * 1000
    
    # 统计结果
    successful_requests = [r for r in results if r['success']]
    failed_requests = [r for r in results if not r['success']]
    
    print(f"\n{'-'*70}")
    print(f"测试结果统计")
    print(f"{'-'*70}")
    print(f"  总请求数: {num_requests}")
    print(f"  成功数: {len(successful_requests)}")
    print(f"  失败数: {len(failed_requests)}")
    print(f"  成功率: {len(successful_requests)/num_requests*100:.1f}%")
    print(f"  总耗时: {total_elapsed:.0f}ms")
    print(f"  吞吐量: {num_requests / (total_elapsed/1000):.2f} req/s")
    
    if successful_requests:
        inference_times = [r['inference_time_ms'] for r in successful_requests]
        total_times = [r['total_time_ms'] for r in successful_requests]
        
        print(f"\n  推理时间统计:")
        print(f"    平均: {statistics.mean(inference_times):.0f}ms")
        print(f"    中位数: {statistics.median(inference_times):.0f}ms")
        print(f"    最小: {min(inference_times):.0f}ms")
        print(f"    最大: {max(inference_times):.0f}ms")
        if len(inference_times) > 1:
            print(f"    标准差: {statistics.stdev(inference_times):.0f}ms")
        
        print(f"\n  端到端时间统计:")
        print(f"    平均: {statistics.mean(total_times):.0f}ms")
        print(f"    中位数: {statistics.median(total_times):.0f}ms")
        print(f"    最小: {min(total_times):.0f}ms")
        print(f"    最大: {max(total_times):.0f}ms")
        if len(total_times) > 1:
            print(f"    标准差: {statistics.stdev(total_times):.0f}ms")
    
    print(f"{'='*70}")
    
    return results


def get_service_stats(base_url):
    """
    获取服务统计信息
    """
    try:
        response = requests.get(f"{base_url}/stats", timeout=5)
        if response.status_code == 200:
            return response.json()
    except Exception as e:
        print(f"获取统计信息失败: {str(e)}")
    return None


def print_service_stats(base_url):
    """
    打印服务统计信息
    """
    stats = get_service_stats(base_url)
    if not stats:
        return
    
    print(f"\n{'='*70}")
    print(f"服务统计信息")
    print(f"{'='*70}")
    print(f"  批处理状态: {'启用' if stats.get('batching_enabled') else '禁用'}")
    
    if stats.get('batching_enabled'):
        config = stats.get('batch_config', {})
        print(f"\n  批处理配置:")
        print(f"    批大小: {config.get('batch_size')}")
        print(f"    批超时: {config.get('batch_timeout')}s")
        print(f"    队列大小: {config.get('max_queue_size')}")
        print(f"    当前队列: {stats.get('queue_size')}")
        
        statistics_data = stats.get('statistics', {})
        if statistics_data and statistics_data.get('total_batches', 0) > 0:
            print(f"\n  批处理统计:")
            print(f"    总请求数: {statistics_data.get('total_requests', 0)}")
            print(f"    总批次数: {statistics_data.get('total_batches', 0)}")
            print(f"    平均批大小: {statistics_data.get('avg_batch_size', 0):.2f}")
            print(f"    最大批大小: {statistics_data.get('max_batch_size', 0)}")
            print(f"    总推理时间: {statistics_data.get('total_inference_time', 0):.0f}ms")
            print(f"    平均每批推理时间: {statistics_data.get('avg_inference_time_per_batch', 0):.0f}ms")
            print(f"    平均每请求推理时间: {statistics_data.get('avg_inference_time_per_request', 0):.0f}ms")
    
    print(f"{'='*70}")


def compare_batch_vs_nobatch(base_url, image_url, num_requests=20, num_workers=10):
    """
    对比批处理和非批处理模式的性能
    """
    print(f"\n{'='*70}")
    print(f"批处理 vs 非批处理性能对比")
    print(f"{'='*70}")
    
    # 先获取当前状态
    stats = get_service_stats(base_url)
    if not stats:
        print("无法获取服务状态，跳过对比测试")
        return
    
    batching_enabled = stats.get('batching_enabled', False)
    
    if batching_enabled:
        print("\n✓ 当前启用了批处理模式")
        print(f"  批大小: {stats.get('batch_config', {}).get('batch_size')}")
        print(f"  批超时: {stats.get('batch_config', {}).get('batch_timeout')}s")
    else:
        print("\n⚠ 当前禁用了批处理模式（使用原有单张推理）")
    
    print(f"\n开始测试...")
    results = test_concurrent_inference(base_url, image_url, num_requests, num_workers)
    
    # 打印服务统计
    print_service_stats(base_url)
    
    return results


def main():
    parser = argparse.ArgumentParser(description='批处理推理并发测试')
    parser.add_argument('--url', default='http://localhost:8000',
                        help='算法服务URL (默认: http://localhost:8000)')
    parser.add_argument('--image-url', required=True,
                        help='测试图片URL')
    parser.add_argument('--requests', type=int, default=20,
                        help='总请求数 (默认: 20)')
    parser.add_argument('--workers', type=int, default=10,
                        help='并发工作线程数 (默认: 10)')
    parser.add_argument('--stats-only', action='store_true',
                        help='仅查看统计信息，不执行测试')
    
    args = parser.parse_args()
    
    if args.stats_only:
        print_service_stats(args.url)
        return
    
    # 执行并发测试
    compare_batch_vs_nobatch(
        args.url, 
        args.image_url,
        args.requests,
        args.workers
    )


if __name__ == '__main__':
    main()

