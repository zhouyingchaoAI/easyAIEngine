#!/usr/bin/env python3
"""
算法服务测试脚本
"""
import requests
import json
import time
import argparse


def test_health(base_url):
    """测试健康检查接口"""
    print("\n1. 测试健康检查接口")
    print("-" * 50)
    
    try:
        response = requests.get(f"{base_url}/health", timeout=5)
        print(f"状态码: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            print(f"响应内容:")
            print(json.dumps(data, indent=2, ensure_ascii=False))
            print("✓ 健康检查通过")
            return True
        else:
            print(f"✗ 健康检查失败: {response.status_code}")
            return False
    except Exception as e:
        print(f"✗ 健康检查失败: {str(e)}")
        return False


def test_index(base_url):
    """测试首页"""
    print("\n2. 测试首页")
    print("-" * 50)
    
    try:
        response = requests.get(base_url, timeout=5)
        print(f"状态码: {response.status_code}")
        
        if response.status_code == 200:
            print(f"✓ 首页访问成功 (内容长度: {len(response.text)} 字节)")
            return True
        else:
            print(f"✗ 首页访问失败: {response.status_code}")
            return False
    except Exception as e:
        print(f"✗ 首页访问失败: {str(e)}")
        return False


def test_inference(base_url, image_url):
    """测试推理接口"""
    print("\n3. 测试推理接口")
    print("-" * 50)
    
    payload = {
        "image_url": image_url,
        "task_id": "test_task",
        "task_type": "人数统计",
        "image_path": "test/test_task/test.jpg"
    }
    
    print(f"请求URL: {base_url}/infer")
    print(f"请求体:")
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    
    try:
        start_time = time.time()
        response = requests.post(
            f"{base_url}/infer",
            json=payload,
            timeout=30
        )
        elapsed_time = (time.time() - start_time) * 1000
        
        print(f"\n状态码: {response.status_code}")
        print(f"总耗时: {elapsed_time:.0f}ms")
        
        if response.status_code == 200:
            data = response.json()
            print(f"\n响应内容:")
            print(json.dumps(data, indent=2, ensure_ascii=False))
            
            if data.get('success'):
                result = data.get('result', {})
                print(f"\n✓ 推理成功")
                print(f"  检测目标数: {result.get('total_count', 0)}")
                print(f"  人数统计: {result.get('person_count', 0)}")
                print(f"  平均置信度: {data.get('confidence', 0):.3f}")
                print(f"  推理耗时: {data.get('inference_time_ms', 0)}ms")
                return True
            else:
                print(f"✗ 推理失败: {data.get('error', 'Unknown error')}")
                return False
        else:
            print(f"✗ 推理请求失败: {response.status_code}")
            print(f"响应: {response.text}")
            return False
    except Exception as e:
        print(f"✗ 推理请求失败: {str(e)}")
        return False


def test_easydarwin_integration(easydarwin_url, service_id):
    """测试EasyDarwin集成"""
    print("\n4. 测试EasyDarwin集成")
    print("-" * 50)
    
    # 查询注册的服务
    try:
        response = requests.get(
            f"{easydarwin_url}/api/v1/ai_analysis/services",
            timeout=5
        )
        
        print(f"状态码: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            services = data.get('services', [])
            
            print(f"注册的服务数: {data.get('total', 0)}")
            
            # 查找目标服务
            found = False
            for service in services:
                if service.get('service_id') == service_id:
                    found = True
                    print(f"\n✓ 找到服务: {service_id}")
                    print(f"  服务名称: {service.get('name')}")
                    print(f"  任务类型: {service.get('task_types')}")
                    print(f"  推理端点: {service.get('endpoint')}")
                    print(f"  版本: {service.get('version')}")
                    print(f"  注册时间: {service.get('register_at')}")
                    print(f"  最后心跳: {service.get('last_heartbeat')}")
                    break
            
            if not found:
                print(f"✗ 未找到服务: {service_id}")
                print(f"提示: 服务可能未启动或使用了--no-register选项")
                return False
            
            return True
        else:
            print(f"✗ 查询失败: {response.status_code}")
            return False
    except Exception as e:
        print(f"✗ EasyDarwin连接失败: {str(e)}")
        print(f"提示: 请确认EasyDarwin正在运行且AI分析插件已启用")
        return False


def main():
    parser = argparse.ArgumentParser(description='测试算法服务')
    parser.add_argument('--url', default='http://localhost:8000',
                        help='算法服务URL (默认: http://localhost:8000)')
    parser.add_argument('--image-url', default='',
                        help='测试图片URL (可选)')
    parser.add_argument('--easydarwin', default='http://localhost:5066',
                        help='EasyDarwin URL (默认: http://localhost:5066)')
    parser.add_argument('--service-id', default='yolo11x_head_detector',
                        help='服务ID (默认: yolo11x_head_detector)')
    parser.add_argument('--skip-inference', action='store_true',
                        help='跳过推理测试（需要有效图片URL）')
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("算法服务测试")
    print("=" * 60)
    print(f"算法服务: {args.url}")
    print(f"EasyDarwin: {args.easydarwin}")
    print(f"服务ID: {args.service_id}")
    
    results = []
    
    # 测试1: 健康检查
    results.append(("健康检查", test_health(args.url)))
    
    # 测试2: 首页
    results.append(("首页访问", test_index(args.url)))
    
    # 测试3: 推理接口
    if not args.skip_inference:
        if args.image_url:
            results.append(("推理接口", test_inference(args.url, args.image_url)))
        else:
            print("\n3. 跳过推理测试")
            print("-" * 50)
            print("提示: 使用 --image-url 参数提供测试图片URL")
    
    # 测试4: EasyDarwin集成
    results.append(("EasyDarwin集成", test_easydarwin_integration(args.easydarwin, args.service_id)))
    
    # 汇总结果
    print("\n" + "=" * 60)
    print("测试结果汇总")
    print("=" * 60)
    
    passed = 0
    total = 0
    
    for name, result in results:
        total += 1
        if result:
            passed += 1
            status = "✓ 通过"
        else:
            status = "✗ 失败"
        print(f"{name:20s} {status}")
    
    print("-" * 60)
    print(f"通过: {passed}/{total}")
    
    if passed == total:
        print("\n🎉 所有测试通过！")
        return 0
    else:
        print(f"\n⚠ {total - passed} 个测试失败")
        return 1


if __name__ == '__main__':
    exit(main())

