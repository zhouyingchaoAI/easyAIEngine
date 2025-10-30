# [root@3813ff504d4a code]# atc --model=best.onnx --framework=5 --output=best --soc_version=Ascend310P3
# ATC start working now, please wait for a moment.
# ........
# ATC run success, welcome to the next use.

# [root@3813ff504d4a code]# ls

import os
import cv2
import numpy as np
import acl
import subprocess

# 手动定义内存分配类型常量（兼容昇腾acl API变化）
try:
    MEM_MALLOC_NORMAL_ONLY = acl.rt.MEM_MALLOC_NORMAL_ONLY
except AttributeError:
    # 最新acl库常量定义可能变更，有时MEM_MALLOC_NORMAL_ONLY不存在，可用0替代
    MEM_MALLOC_NORMAL_ONLY = 0

# 内存拷贝方向常量适配（兼容不同ACL版本）
try:
    MEMCPY_HOST_TO_DEVICE = acl.rt.MEMCPY_HOST_TO_DEVICE
    MEMCPY_DEVICE_TO_HOST = acl.rt.MEMCPY_DEVICE_TO_HOST
except AttributeError:
    try:
        MEMCPY_HOST_TO_DEVICE = acl.memcpy_host_to_device
        MEMCPY_DEVICE_TO_HOST = acl.memcpy_device_to_host
    except AttributeError:
        # Fallback: 常见实际为0和1
        MEMCPY_HOST_TO_DEVICE = 0
        MEMCPY_DEVICE_TO_HOST = 1

# 全局变量存储ACL资源
ACL_RESOURCE = {
    'context': None,
    'stream': None,
    'model_id': None,
    'model_desc': None,
    'device_id': 0
}

# 2. 初始化ACL资源
def init_acl_resource(device_id=0):
    """
    初始化昇腾ACL运行环境
    """
    ret = acl.init()
    if ret != 0:
        raise Exception(f"ACL初始化失败, 错误码: {ret}")
    print(f"ACL初始化成功")
    ret = acl.rt.set_device(device_id)
    if ret != 0:
        raise Exception(f"设置设备失败, 错误码: {ret}")
    context, ret = acl.rt.create_context(device_id)
    if ret != 0:
        raise Exception(f"创建Context失败, 错误码: {ret}")
    stream, ret = acl.rt.create_stream()
    if ret != 0:
        raise Exception(f"创建Stream失败, 错误码: {ret}")
    ACL_RESOURCE['context'] = context
    ACL_RESOURCE['stream'] = stream
    ACL_RESOURCE['device_id'] = device_id
    print(f"ACL资源初始化完成 (Device: {device_id})")

# 3. 加载OM模型
def load_om_model(om_path):
    """
    加载OM模型到NPU
    """
    model_id, ret = acl.mdl.load_from_file(om_path)
    if ret != 0:
        raise Exception(f"加载模型失败, 错误码: {ret}")
    model_desc = acl.mdl.create_desc()
    ret = acl.mdl.get_desc(model_desc, model_id)
    if ret != 0:
        raise Exception(f"获取模型描述失败, 错误码: {ret}")
    ACL_RESOURCE['model_id'] = model_id
    ACL_RESOURCE['model_desc'] = model_desc
    print(f"OM模型加载成功, Model ID: {model_id}")
    # 打印模型输入输出信息
    input_num = acl.mdl.get_num_inputs(model_desc)
    output_num = acl.mdl.get_num_outputs(model_desc)
    print(f"模型输入数量: {input_num}, 输出数量: {output_num}")
    return model_id, model_desc

# 4. 读取图片
def load_image(image_path):
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Image not found: {image_path}")
    return img

# 5. OM模型推理（修复版）
def om_infer(om_path, img_bgr, conf_thres=0.25, iou_thres=0.45, img_size=1024, debug=True):
    """
    使用昇腾310进行OM模型推理
    """
    orig_h, orig_w = img_bgr.shape[:2]
    img_in, ratio, (dw, dh) = letterbox(img_bgr, new_shape=(img_size, img_size))
    
    if debug:
        print(f"原始图片尺寸: {img_bgr.shape}")
        print(f"Letterbox后尺寸: {img_in.shape}")
        print(f"缩放比例: {ratio}")
        print(f"Padding: dw={dw}, dh={dh}")
    
    # 预处理
    img_in = cv2.cvtColor(img_in, cv2.COLOR_BGR2RGB)
    img_in = img_in.transpose(2, 0, 1)
    img_in = np.ascontiguousarray(img_in).astype(np.float32) / 255.0
    img_in = np.expand_dims(img_in, 0)  # (1, 3, 1024, 1024)
    
    model_id = ACL_RESOURCE['model_id']
    model_desc = ACL_RESOURCE['model_desc']
    
    # 准备输入数据
    input_size = acl.mdl.get_input_size_by_index(model_desc, 0)
    if debug:
        print(f"模型期望输入大小: {input_size} bytes")
        print(f"实际数据大小: {img_in.nbytes} bytes")
    
    # 确保数据大小匹配
    if img_in.nbytes != input_size:
        print(f"警告: 数据大小不匹配! 期望{input_size}, 实际{img_in.nbytes}")
    
    # 申请设备内存
    input_buffer, ret = acl.rt.malloc(input_size, MEM_MALLOC_NORMAL_ONLY)
    if ret != 0:
        raise Exception(f"申请输入内存失败, 错误码: {ret}")
    
    # 准备输入数据指针
    input_bytes = img_in.tobytes()
    input_ptr = acl.util.bytes_to_ptr(input_bytes)
    
    # 拷贝数据到设备
    ret = acl.rt.memcpy(input_buffer, input_size, input_ptr, len(input_bytes), MEMCPY_HOST_TO_DEVICE)
    if ret != 0:
        acl.rt.free(input_buffer)
        raise Exception(f"拷贝输入数据失败, 错误码: {ret}")
    
    # 创建输入dataset
    input_dataset = acl.mdl.create_dataset()
    input_data = acl.create_data_buffer(input_buffer, input_size)
    
    # 添加buffer到dataset（修复返回值处理）
    ret = acl.mdl.add_dataset_buffer(input_dataset, input_data)
    if isinstance(ret, tuple):
        # 某些ACL版本返回元组 (dataset, ret_code)
        actual_ret = ret[1] if len(ret) > 1 else ret[0]
        if debug:
            print(f"add_dataset_buffer返回元组: {ret}, 使用ret_code: {actual_ret}")
    else:
        actual_ret = ret
    
    if actual_ret != 0 and debug:
        print(f"警告: 添加输入buffer返回非0值: {actual_ret}")
    
    # 准备输出
    output_size = acl.mdl.get_output_size_by_index(model_desc, 0)
    if debug:
        print(f"模型输出大小: {output_size} bytes")
    
    output_buffer, ret = acl.rt.malloc(output_size, MEM_MALLOC_NORMAL_ONLY)
    if ret != 0:
        acl.rt.free(input_buffer)
        acl.destroy_data_buffer(input_data)
        acl.mdl.destroy_dataset(input_dataset)
        raise Exception(f"申请输出内存失败, 错误码: {ret}")
    
    # 创建输出dataset
    output_dataset = acl.mdl.create_dataset()
    output_data = acl.create_data_buffer(output_buffer, output_size)
    
    ret = acl.mdl.add_dataset_buffer(output_dataset, output_data)
    if isinstance(ret, tuple):
        actual_ret = ret[1] if len(ret) > 1 else ret[0]
        if debug:
            print(f"add_dataset_buffer返回元组: {ret}, 使用ret_code: {actual_ret}")
    else:
        actual_ret = ret
    
    if actual_ret != 0 and debug:
        print(f"警告: 添加输出buffer返回非0值: {actual_ret}")
    
    # 执行推理
    if debug:
        print("开始执行模型推理...")
    
    # 初始化输出numpy数组变量
    output_np = None
    
    try:
        ret = acl.mdl.execute(model_id, input_dataset, output_dataset)
        if ret != 0:
            raise Exception(f"模型推理失败, 错误码: {ret}")
        
        if debug:
            print("模型推理成功")
        
        # 获取输出数据 - 修复方案
        if debug:
            print("开始获取输出数据...")
        
        # 方案1: 先拷贝到Host内存再读取
        try:
            # 在Host端分配内存（使用ctypes）
            import ctypes
            host_buffer = (ctypes.c_byte * output_size)()
            host_ptr = ctypes.cast(host_buffer, ctypes.c_void_p).value
            
            # 从Device拷贝到Host
            ret = acl.rt.memcpy(host_ptr, output_size, output_buffer, output_size, MEMCPY_DEVICE_TO_HOST)
            if ret != 0:
                raise Exception(f"拷贝输出数据到Host失败, 错误码: {ret}")
            
            if debug:
                print("成功拷贝输出数据到Host")
            
            # 从Host buffer读取
            output_np = np.frombuffer(host_buffer, dtype=np.float32).copy()
            
        except Exception as e:
            if debug:
                print(f"方案1失败: {e}, 尝试方案2")
            
            # 方案2: 使用ACL推荐的API
            try:
                # 创建Host内存
                host_ptr, ret = acl.rt.malloc_host(output_size)
                if ret != 0:
                    raise Exception(f"分配Host内存失败, 错误码: {ret}")
                
                # 拷贝数据
                ret = acl.rt.memcpy(host_ptr, output_size, output_buffer, output_size, MEMCPY_DEVICE_TO_HOST)
                if ret != 0:
                    acl.rt.free_host(host_ptr)
                    raise Exception(f"拷贝数据失败, 错误码: {ret}")
                
                if debug:
                    print("方案2: 使用malloc_host成功")
                
                # 转换为numpy
                output_np = acl.util.ptr_to_numpy(host_ptr, (output_size // 4,), np.float32).copy()
                
                # 释放Host内存
                acl.rt.free_host(host_ptr)
                
            except Exception as e2:
                if debug:
                    print(f"方案2失败: {e2}, 尝试方案3")
                
                # 方案3: 直接从Device地址读取（某些ACL版本支持）
                output_ptr = acl.get_data_buffer_addr(output_data)
                output_np = acl.util.ptr_to_numpy(output_ptr, (output_size // 4,), np.float32).copy()
        
        # 重塑输出
        output_dims = acl.mdl.get_output_dims(model_desc, 0)
        
        # 处理不同ACL版本的返回格式
        if isinstance(output_dims, dict):
            dims = output_dims['dims']
        elif isinstance(output_dims, tuple):
            # 返回格式通常是 (dims_tuple, ret_code)
            if len(output_dims) == 2:
                dims_info, ret = output_dims
                if ret != 0:
                    raise Exception(f"获取输出维度失败, 错误码: {ret}")
                # dims_info可能是字典或直接是维度列表
                if isinstance(dims_info, dict):
                    dims = dims_info['dims']
                else:
                    dims = dims_info
            else:
                # 直接是维度元组
                dims = output_dims
        else:
            dims = output_dims
        
        if debug:
            print(f"输出维度信息: {dims}")
        
        # 转换为列表（如果需要）
        if hasattr(dims, '__iter__'):
            dims = list(dims)
        
        preds = output_np.reshape(dims)
        
    finally:
        # 清理资源
        if debug:
            print("开始清理ACL资源...")
        acl.rt.free(input_buffer)
        acl.rt.free(output_buffer)
        acl.destroy_data_buffer(input_data)
        acl.destroy_data_buffer(output_data)
        acl.mdl.destroy_dataset(input_dataset)
        acl.mdl.destroy_dataset(output_dataset)
        if debug:
            print("ACL资源清理完成")
    
    # 确保preds已定义
    if output_np is None:
        raise Exception("未能成功获取输出数据")
    
    if debug:
        print(f"OM输出shape: {preds.shape}")
    
    # 转置处理
    if preds.shape[1] < preds.shape[2]:
        preds = np.transpose(preds, (0, 2, 1))
    preds = preds[0]  # (N, C)
    
    num_outputs = preds.shape[1]
    if num_outputs == 5:
        is_single_class = True
        num_classes = 1
        if debug:
            print("检测模式: 单类别")
    else:
        is_single_class = False
        num_classes = num_outputs - 4
        if debug:
            print(f"检测模式: 多类别 ({num_classes}类)")
    
    # 置信度过滤
    boxes = []
    all_max_confs = []
    for pred in preds:
        if is_single_class:
            cx, cy, w, h, conf = pred
            cls_id = 0
        else:
            cx, cy, w, h = pred[:4]
            class_scores = pred[4:]
            cls_id = np.argmax(class_scores)
            conf = class_scores[cls_id]
        
        all_max_confs.append(conf)
        if conf < conf_thres:
            continue
        
        # 坐标转换
        if isinstance(ratio, tuple):
            ratio_w, ratio_h = ratio
        else:
            ratio_w = ratio_h = ratio
        
        x1 = (cx - w / 2 - dw) / ratio_w
        y1 = (cy - h / 2 - dh) / ratio_h
        x2 = (cx + w / 2 - dw) / ratio_w
        y2 = (cy + h / 2 - dh) / ratio_h
        
        # 边界检查
        x1 = max(0, min(x1, orig_w))
        y1 = max(0, min(y1, orig_h))
        x2 = max(0, min(x2, orig_w))
        y2 = max(0, min(y2, orig_h))
        
        if x2 <= x1 or y2 <= y1:
            continue
        
        boxes.append([x1, y1, x2, y2, float(conf), int(cls_id)])
    
    if debug:
        all_max_confs = np.array(all_max_confs)
        print(f"\n置信度统计:")
        print(f"  - 最大值: {all_max_confs.max():.4f}")
        print(f"  - 均值: {all_max_confs.mean():.4f}")
        print(f"  - >0.25的数量: {(all_max_confs > 0.25).sum()}")
        print(f"置信度过滤后boxes数量: {len(boxes)}")
    
    # NMS处理
    boxes_out = []
    if len(boxes) > 0:
        boxes_arr = np.array(boxes)
        xyxy = boxes_arr[:, :4]
        scores = boxes_arr[:, 4]
        classes = boxes_arr[:, 5].astype(int)
        unique_classes = np.unique(classes)
        
        for cls in unique_classes:
            cls_mask = classes == cls
            cls_boxes = xyxy[cls_mask]
            cls_scores = scores[cls_mask]
            cls_indices_in_filtered = np.where(cls_mask)[0]
            order = np.argsort(-cls_scores)
            keep = []
            
            while len(order) > 0:
                i = order[0]
                keep.append(i)
                if len(order) == 1:
                    break
                
                xx1 = np.maximum(cls_boxes[i, 0], cls_boxes[order[1:], 0])
                yy1 = np.maximum(cls_boxes[i, 1], cls_boxes[order[1:], 1])
                xx2 = np.minimum(cls_boxes[i, 2], cls_boxes[order[1:], 2])
                yy2 = np.minimum(cls_boxes[i, 3], cls_boxes[order[1:], 3])
                
                w = np.maximum(0.0, xx2 - xx1)
                h = np.maximum(0.0, yy2 - yy1)
                inter = w * h
                
                area_i = (cls_boxes[i, 2] - cls_boxes[i, 0]) * (cls_boxes[i, 3] - cls_boxes[i, 1])
                area_other = (cls_boxes[order[1:], 2] - cls_boxes[order[1:], 0]) * \
                            (cls_boxes[order[1:], 3] - cls_boxes[order[1:], 1])
                
                iou = inter / (area_i + area_other - inter + 1e-6)
                inds = np.where(iou <= iou_thres)[0]
                order = order[inds + 1]
            
            for idx in keep:
                original_idx = cls_indices_in_filtered[int(idx)]
                boxes_out.append(boxes[original_idx])
    
    if debug:
        print(f"NMS后boxes数量: {len(boxes_out)}")
    
    return boxes_out

# Letterbox预处理
def letterbox(img, new_shape=(640, 640), color=(114, 114, 114), auto=False, 
              scaleFill=False, scaleup=True, stride=32):
    shape = img.shape[:2]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:
        r = min(r, 1.0)
    ratio = r, r
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]
    if auto:
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)
    elif scaleFill:
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])
        ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]
    dw /= 2
    dh /= 2
    if shape[::-1] != new_unpad:
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    return img, ratio, (dw, dh)

# 6. 绘制并保存结果
def save_result(img_bgr, boxes_out, out_img_path, class_names=None):
    img_draw = img_bgr.copy()
    np.random.seed(42)
    colors = np.random.randint(0, 255, size=(80, 3), dtype=np.uint8)
    for box in boxes_out:
        x1, y1, x2, y2, conf, cls = box
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        cls = int(cls)
        color = tuple(map(int, colors[cls % len(colors)]))
        cv2.rectangle(img_draw, (x1, y1), (x2, y2), color, 2)
        label = f"Class {cls}: {conf:.2f}" if class_names is None else f"{class_names[cls]}: {conf:.2f}"
        (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(img_draw, (x1, y1 - h - 4), (x1 + w, y1), color, -1)
        cv2.putText(img_draw, label, (x1, y1 - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    cv2.imwrite(out_img_path, img_draw)
    print(f"检测到 {len(boxes_out)} 个目标")

# 7. 释放ACL资源
def release_acl_resource():
    """
    释放所有ACL资源
    """
    if ACL_RESOURCE['model_id'] is not None:
        acl.mdl.unload(ACL_RESOURCE['model_id'])
    if ACL_RESOURCE['model_desc'] is not None:
        acl.mdl.destroy_desc(ACL_RESOURCE['model_desc'])
    if ACL_RESOURCE['stream'] is not None:
        acl.rt.destroy_stream(ACL_RESOURCE['stream'])
    if ACL_RESOURCE['context'] is not None:
        acl.rt.destroy_context(ACL_RESOURCE['context'])
    acl.rt.reset_device(ACL_RESOURCE['device_id'])
    acl.finalize()
    print("ACL资源已释放")

if __name__ == "__main__":
    YOLO_CKPT = "/cv_space/NWPU-Crowd/runs/exp_yolo11x3/weights/best.pt"
    IMAGE_PATH = "0001.jpg"
    OUTPUT_IMAGE = "result_om1.jpg"
    DEVICE_ID = 0  # 昇腾设备ID
    om_path = "./best.om"
    
    try:
        print("========== 初始化ACL资源 ==========")
        init_acl_resource(device_id=DEVICE_ID)
        
        print("\n========== 加载OM模型 ==========")
        load_om_model(om_path)
        
        img_bgr = load_image(IMAGE_PATH)
        print(f"\n图片尺寸: {img_bgr.shape}")
        
        print("\n========== OM推理 (昇腾310) ==========")
        boxes = om_infer(
            om_path, 
            img_bgr, 
            conf_thres=0.08,
            iou_thres=0.35,
            debug=True
        )
        
        save_result(img_bgr, boxes, OUTPUT_IMAGE)
        print(f"\n结果已保存至: {OUTPUT_IMAGE}")
        print(f"检测框数量: OM={len(boxes)}")
        
    except Exception as e:
        print(f"\n错误: {e}")
        import traceback
        traceback.print_exc()
    finally:
        print("\n========== 释放资源 ==========")
        release_acl_resource()
