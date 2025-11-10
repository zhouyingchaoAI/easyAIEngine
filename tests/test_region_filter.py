#!/usr/bin/env python3
"""
区域过滤功能测试脚本
测试算法服务的区域过滤是否正常工作
"""
import json
import sys
from pathlib import Path

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from algorithm_service import filter_objects_by_region, point_in_polygon


def test_point_in_polygon():
    """测试点在多边形内的判断"""
    print("=" * 60)
    print("测试 1: 点在多边形内的判断")
    print("=" * 60)
    
    # 定义一个正方形
    square = [(100, 100), (400, 100), (400, 400), (100, 400)]
    
    test_cases = [
        ((250, 250), True, "中心点"),
        ((100, 100), True, "顶点"),
        ((250, 100), True, "边上"),
        ((50, 50), False, "外部左上"),
        ((450, 450), False, "外部右下"),
    ]
    
    for point, expected, desc in test_cases:
        result = point_in_polygon(point, square)
        status = "✓" if result == expected else "✗"
        print(f"  {status} {desc}: {point} -> {result} (期望: {expected})")
    
    print()


def test_rectangle_filter():
    """测试矩形区域过滤"""
    print("=" * 60)
    print("测试 2: 矩形区域过滤")
    print("=" * 60)
    
    # 测试对象（归一化坐标）
    objects = [
        {'class': 'head', 'confidence': 0.9, 'bbox': [0.1, 0.1, 0.2, 0.2]},  # 左上角
        {'class': 'head', 'confidence': 0.8, 'bbox': [0.4, 0.4, 0.5, 0.5]},  # 中心
        {'class': 'head', 'confidence': 0.7, 'bbox': [0.8, 0.8, 0.9, 0.9]},  # 右下角
    ]
    
    # 矩形区域：中心区域 (0.3, 0.3) 到 (0.7, 0.7)
    regions = [{
        'id': 'area_1',
        'type': 'rectangle',
        'enabled': True,
        'points': [[0.3, 0.3], [0.7, 0.7]]
    }]
    
    image_size = (1000, 1000)
    
    print(f"  原始对象数量: {len(objects)}")
    for i, obj in enumerate(objects):
        center_x = (obj['bbox'][0] + obj['bbox'][2]) / 2
        center_y = (obj['bbox'][1] + obj['bbox'][3]) / 2
        print(f"    对象 {i+1}: 中心点 ({center_x:.2f}, {center_y:.2f})")
    
    filtered = filter_objects_by_region(objects, regions, image_size)
    
    print(f"\n  过滤后对象数量: {len(filtered)}")
    for i, obj in enumerate(filtered):
        center_x = (obj['bbox'][0] + obj['bbox'][2]) / 2
        center_y = (obj['bbox'][1] + obj['bbox'][3]) / 2
        print(f"    对象 {i+1}: 中心点 ({center_x:.2f}, {center_y:.2f})")
    
    assert len(filtered) == 1, f"期望保留1个对象，实际保留{len(filtered)}个"
    assert filtered[0]['bbox'] == [0.4, 0.4, 0.5, 0.5], "保留的对象不正确"
    print("\n  ✓ 测试通过")
    print()


def test_polygon_filter():
    """测试多边形区域过滤"""
    print("=" * 60)
    print("测试 3: 多边形区域过滤")
    print("=" * 60)
    
    # 测试对象（像素坐标）
    objects = [
        {'class': 'head', 'confidence': 0.9, 'bbox': [100, 100, 150, 150]},  # 外部
        {'class': 'head', 'confidence': 0.8, 'bbox': [250, 250, 300, 300]},  # 内部
        {'class': 'head', 'confidence': 0.7, 'bbox': [450, 450, 500, 500]},  # 外部
    ]
    
    # 三角形区域
    regions = [{
        'id': 'area_1',
        'type': 'polygon',
        'enabled': True,
        'points': [[200, 100], [400, 400], [100, 400]]
    }]
    
    image_size = (600, 600)
    
    print(f"  原始对象数量: {len(objects)}")
    filtered = filter_objects_by_region(objects, regions, image_size)
    
    print(f"  过滤后对象数量: {len(filtered)}")
    
    assert len(filtered) == 1, f"期望保留1个对象，实际保留{len(filtered)}个"
    print("  ✓ 测试通过")
    print()


def test_multiple_regions():
    """测试多个区域过滤"""
    print("=" * 60)
    print("测试 4: 多个区域过滤（OR逻辑）")
    print("=" * 60)
    
    # 测试对象
    objects = [
        {'class': 'head', 'confidence': 0.9, 'bbox': [0.1, 0.1, 0.2, 0.2]},  # 区域1
        {'class': 'head', 'confidence': 0.8, 'bbox': [0.4, 0.4, 0.5, 0.5]},  # 中间（不在任何区域）
        {'class': 'head', 'confidence': 0.7, 'bbox': [0.7, 0.7, 0.8, 0.8]},  # 区域2
    ]
    
    # 两个矩形区域
    regions = [
        {
            'id': 'area_1',
            'type': 'rectangle',
            'enabled': True,
            'points': [[0.0, 0.0], [0.3, 0.3]]
        },
        {
            'id': 'area_2',
            'type': 'rectangle',
            'enabled': True,
            'points': [[0.6, 0.6], [0.9, 0.9]]
        }
    ]
    
    image_size = (1000, 1000)
    
    print(f"  原始对象数量: {len(objects)}")
    filtered = filter_objects_by_region(objects, regions, image_size)
    
    print(f"  过滤后对象数量: {len(filtered)}")
    
    assert len(filtered) == 2, f"期望保留2个对象，实际保留{len(filtered)}个"
    print("  ✓ 测试通过（OR逻辑生效）")
    print()


def test_disabled_region():
    """测试禁用区域"""
    print("=" * 60)
    print("测试 5: 禁用区域")
    print("=" * 60)
    
    objects = [
        {'class': 'head', 'confidence': 0.9, 'bbox': [0.4, 0.4, 0.5, 0.5]},
    ]
    
    # 禁用的区域
    regions = [{
        'id': 'area_1',
        'type': 'rectangle',
        'enabled': False,  # 禁用
        'points': [[0.3, 0.3], [0.7, 0.7]]
    }]
    
    image_size = (1000, 1000)
    
    print(f"  原始对象数量: {len(objects)}")
    filtered = filter_objects_by_region(objects, regions, image_size)
    
    print(f"  过滤后对象数量: {len(filtered)}")
    
    # 因为区域被禁用，应该返回所有对象
    assert len(filtered) == len(objects), "禁用区域应该不过滤任何对象"
    print("  ✓ 测试通过（禁用区域不生效）")
    print()


def test_no_regions():
    """测试无区域配置"""
    print("=" * 60)
    print("测试 6: 无区域配置")
    print("=" * 60)
    
    objects = [
        {'class': 'head', 'confidence': 0.9, 'bbox': [0.1, 0.1, 0.2, 0.2]},
        {'class': 'head', 'confidence': 0.8, 'bbox': [0.5, 0.5, 0.6, 0.6]},
        {'class': 'head', 'confidence': 0.7, 'bbox': [0.8, 0.8, 0.9, 0.9]},
    ]
    
    regions = []
    image_size = (1000, 1000)
    
    print(f"  原始对象数量: {len(objects)}")
    filtered = filter_objects_by_region(objects, regions, image_size)
    
    print(f"  过滤后对象数量: {len(filtered)}")
    
    assert len(filtered) == len(objects), "无区域配置应该返回所有对象"
    print("  ✓ 测试通过（无过滤）")
    print()


def test_config_file():
    """测试配置文件加载"""
    print("=" * 60)
    print("测试 7: 配置文件格式")
    print("=" * 60)
    
    config_file = Path(__file__).parent / "configs" / "test_region_filter_config.json"
    
    if config_file.exists():
        with open(config_file, 'r', encoding='utf-8') as f:
            config = json.load(f)
        
        print(f"  ✓ 配置文件存在: {config_file}")
        print(f"  任务ID: {config.get('task_id')}")
        print(f"  任务类型: {config.get('task_type')}")
        print(f"  区域数量: {len(config.get('regions', []))}")
        
        regions = config.get('regions', [])
        for region in regions:
            print(f"\n  区域配置:")
            print(f"    ID: {region.get('id')}")
            print(f"    名称: {region.get('name')}")
            print(f"    类型: {region.get('type')}")
            print(f"    启用: {region.get('enabled')}")
            print(f"    坐标点数: {len(region.get('points', []))}")
        
        print("\n  ✓ 配置文件格式正确")
    else:
        print(f"  ⚠️  配置文件不存在: {config_file}")
    
    print()


def main():
    """运行所有测试"""
    print("\n" + "=" * 60)
    print("算法区域过滤功能测试")
    print("=" * 60 + "\n")
    
    try:
        test_point_in_polygon()
        test_rectangle_filter()
        test_polygon_filter()
        test_multiple_regions()
        test_disabled_region()
        test_no_regions()
        test_config_file()
        
        print("=" * 60)
        print("✓ 所有测试通过！")
        print("=" * 60)
        
        print("\n使用说明:")
        print("1. 区域过滤功能已集成到算法服务中")
        print("2. 支持矩形和多边形区域")
        print("3. 支持多个区域（OR逻辑）")
        print("4. 可以通过enabled字段禁用区域")
        print("5. 无区域配置时不进行过滤")
        print("\n详细文档请查看: 区域过滤配置示例.md\n")
        
        return 0
    
    except AssertionError as e:
        print(f"\n✗ 测试失败: {e}\n")
        return 1
    except Exception as e:
        print(f"\n✗ 错误: {e}\n")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())

