import os
import cv2
import numpy as np
import acl
import time

# 内存分配类型常量
try:
    MEM_MALLOC_NORMAL_ONLY = acl.rt.MEM_MALLOC_NORMAL_ONLY
except AttributeError:
    MEM_MALLOC_NORMAL_ONLY = 0

try:
    MEMCPY_HOST_TO_DEVICE = acl.rt.MEMCPY_HOST_TO_DEVICE
    MEMCPY_DEVICE_TO_HOST = acl.rt.MEMCPY_DEVICE_TO_HOST
except AttributeError:
    try:
        MEMCPY_HOST_TO_DEVICE = acl.memcpy_host_to_device
        MEMCPY_DEVICE_TO_HOST = acl.memcpy_device_to_host
    except AttributeError:
        MEMCPY_HOST_TO_DEVICE = 0
        MEMCPY_DEVICE_TO_HOST = 1

ACL_RESOURCE = {
    'context': None,
    'stream': None,
    'model_id': None,
    'model_desc': None,
    'device_id': 0,
    # ===== 优化1: 预分配缓冲区，避免每次推理重复申请/释放内存 =====
    'input_buffer': None,
    'output_buffer': None,
    'input_size': 0,
    'output_size': 0,
    'input_dataset': None,
    'output_dataset': None,
    'input_data': None,
    'output_data': None,
    'output_dims': None,
    # ===== 优化2: 预分配Host端输出缓冲区 =====
    'host_output_buffer': None,
}


def init_acl_resource(device_id=0):
    t0 = time.time()
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
    t1 = time.time()
    print(f"ACL资源初始化完成 (Device: {device_id})，耗时: {(t1 - t0)*1000:.2f} ms")


def load_om_model(om_path):
    t0 = time.time()
    model_id, ret = acl.mdl.load_from_file(om_path)
    if ret != 0:
        raise Exception(f"加载模型失败, 错误码: {ret}")
    model_desc = acl.mdl.create_desc()
    ret = acl.mdl.get_desc(model_desc, model_id)
    if ret != 0:
        raise Exception(f"获取模型描述失败, 错误码: {ret}")
    ACL_RESOURCE['model_id'] = model_id
    ACL_RESOURCE['model_desc'] = model_desc

    # ===== 优化1: 模型加载时预分配输入/输出缓冲区 =====
    input_size = acl.mdl.get_input_size_by_index(model_desc, 0)
    output_size = acl.mdl.get_output_size_by_index(model_desc, 0)

    input_buffer, ret = acl.rt.malloc(input_size, MEM_MALLOC_NORMAL_ONLY)
    if ret != 0:
        raise Exception(f"预分配输入内存失败, 错误码: {ret}")
    output_buffer, ret = acl.rt.malloc(output_size, MEM_MALLOC_NORMAL_ONLY)
    if ret != 0:
        acl.rt.free(input_buffer)
        raise Exception(f"预分配输出内存失败, 错误码: {ret}")

    # 预创建dataset和data_buffer
    input_dataset = acl.mdl.create_dataset()
    input_data = acl.create_data_buffer(input_buffer, input_size)
    acl.mdl.add_dataset_buffer(input_dataset, input_data)

    output_dataset = acl.mdl.create_dataset()
    output_data = acl.create_data_buffer(output_buffer, output_size)
    acl.mdl.add_dataset_buffer(output_dataset, output_data)

    # 预获取输出维度
    output_dims = acl.mdl.get_output_dims(model_desc, 0)
    if isinstance(output_dims, dict):
        dims = output_dims['dims']
    elif isinstance(output_dims, tuple):
        if len(output_dims) == 2:
            dims_info, ret = output_dims
            if ret != 0:
                raise Exception(f"获取输出维度失败, 错误码: {ret}")
            dims = dims_info['dims'] if isinstance(dims_info, dict) else dims_info
        else:
            dims = output_dims
    else:
        dims = output_dims
    if hasattr(dims, '__iter__'):
        dims = list(dims)

    ACL_RESOURCE['input_buffer'] = input_buffer
    ACL_RESOURCE['output_buffer'] = output_buffer
    ACL_RESOURCE['input_size'] = input_size
    ACL_RESOURCE['output_size'] = output_size
    ACL_RESOURCE['input_dataset'] = input_dataset
    ACL_RESOURCE['output_dataset'] = output_dataset
    ACL_RESOURCE['input_data'] = input_data
    ACL_RESOURCE['output_data'] = output_data
    ACL_RESOURCE['output_dims'] = dims

    # ===== 优化2: 预分配Host端输出缓冲区 =====
    import ctypes
    host_buffer = (ctypes.c_byte * output_size)()
    ACL_RESOURCE['host_output_buffer'] = host_buffer

    input_num = acl.mdl.get_num_inputs(model_desc)
    output_num = acl.mdl.get_num_outputs(model_desc)
    t1 = time.time()
    print(f"OM模型加载成功, Model ID: {model_id}")
    print(f"模型输入数量: {input_num}, 输出数量: {output_num}")
    print(f"预分配缓冲区: 输入={input_size}B, 输出={output_size}B")
    print(f"加载OM模型耗时: {(t1-t0)*1000:.2f} ms")
    return model_id, model_desc


def load_image(image_path):
    t0 = time.time()
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Image not found: {image_path}")
    t1 = time.time()
    print(f"图片读取耗时: {(t1-t0)*1000:.2f} ms")
    return img


def nms_fast(boxes, scores, iou_thres):
    """优化后的NMS，使用向量化操作"""
    if len(boxes) == 0:
        return []
    boxes = boxes.astype(np.float32)
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        if order.size == 1:
            break
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        w = np.maximum(0., xx2 - xx1)
        h = np.maximum(0., yy2 - yy1)
        inter = w * h
        iou = inter / (areas[i] + areas[order[1:]] - inter + 1e-6)
        inds = np.where(iou <= iou_thres)[0]
        order = order[inds + 1]
    return keep


# ===== 优化3: 预编译letterbox的resize参数 =====
_LETTERBOX_CACHE = {}

def letterbox(img, new_shape=(640, 640), color=(114, 114, 114), auto=False, scaleFill=False, scaleup=True, stride=32):
    shape = img.shape[:2]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)
    
    # 使用缓存避免重复计算（对于相同尺寸的图片）
    cache_key = (shape, new_shape, auto, scaleFill, scaleup, stride)
    if cache_key in _LETTERBOX_CACHE:
        cached = _LETTERBOX_CACHE[cache_key]
        ratio, new_unpad, dw, dh, top, bottom, left, right = cached
        if shape[::-1] != new_unpad:
            img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
        img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
        return img, ratio, (dw, dh)
    
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
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    
    # 缓存计算结果
    _LETTERBOX_CACHE[cache_key] = (ratio, new_unpad, dw, dh, top, bottom, left, right)
    
    if shape[::-1] != new_unpad:
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    return img, ratio, (dw, dh)


def om_infer(om_path, img_bgr, conf_thres=0.25, iou_thres=0.45, img_size=1024, debug=True, return_times=False):
    times = {}
    t_all0 = time.time()
    
    # -- 预处理 (优化4: 减少内存拷贝)
    t0 = time.time()
    orig_h, orig_w = img_bgr.shape[:2]
    img_in, ratio, (dw, dh) = letterbox(img_bgr, new_shape=(img_size, img_size))
    
    # 优化: 合并颜色转换和transpose，减少中间数组创建
    img_in = cv2.cvtColor(img_in, cv2.COLOR_BGR2RGB)
    img_in = np.ascontiguousarray(img_in.transpose(2, 0, 1), dtype=np.float32)
    img_in *= (1.0 / 255.0)  # 原地操作比 /= 255.0 更快
    img_in = img_in[np.newaxis, ...]  # 比 expand_dims 略快
    
    t1 = time.time()
    times['preprocess_ms'] = (t1 - t0) * 1000
    if debug:
        print(f"【阶段耗时】预处理: {times['preprocess_ms']:.2f} ms")

    # 使用预分配的缓冲区
    model_id = ACL_RESOURCE['model_id']
    input_buffer = ACL_RESOURCE['input_buffer']
    output_buffer = ACL_RESOURCE['output_buffer']
    input_size = ACL_RESOURCE['input_size']
    output_size = ACL_RESOURCE['output_size']
    input_dataset = ACL_RESOURCE['input_dataset']
    output_dataset = ACL_RESOURCE['output_dataset']
    dims = ACL_RESOURCE['output_dims']
    host_buffer = ACL_RESOURCE['host_output_buffer']

    # -- 内存拷贝 (优化5: 只做H2D拷贝，无需重新分配)
    t0 = time.time()
    input_bytes = img_in.tobytes()
    input_ptr = acl.util.bytes_to_ptr(input_bytes)
    ret = acl.rt.memcpy(input_buffer, input_size, input_ptr, len(input_bytes), MEMCPY_HOST_TO_DEVICE)
    if ret != 0:
        raise Exception(f"拷贝输入数据失败, 错误码: {ret}")
    t1 = time.time()
    times['memcopy_ms'] = (t1 - t0) * 1000
    if debug:
        print(f"【阶段耗时】输入内存复制: {times['memcopy_ms']:.2f} ms")

    # -- 推理
    t0 = time.time()
    ret = acl.mdl.execute(model_id, input_dataset, output_dataset)
    if ret != 0:
        raise Exception(f"模型推理失败, 错误码: {ret}")
    t1 = time.time()
    times['inference_ms'] = (t1 - t0) * 1000
    if debug:
        print(f"【阶段耗时】NPU推理: {times['inference_ms']:.2f} ms")

    # -- 输出拷贝 (优化6: 使用预分配的host buffer)
    t0 = time.time()
    import ctypes
    host_ptr = ctypes.cast(host_buffer, ctypes.c_void_p).value
    ret = acl.rt.memcpy(host_ptr, output_size, output_buffer, output_size, MEMCPY_DEVICE_TO_HOST)
    if ret != 0:
        raise Exception(f"拷贝输出数据到Host失败, 错误码: {ret}")
    output_np = np.frombuffer(host_buffer, dtype=np.float32).copy()
    preds = output_np.reshape(dims)
    t1 = time.time()
    times['output_ms'] = (t1 - t0) * 1000
    if debug:
        print(f"【阶段耗时】输出搬运和reshape: {times['output_ms']:.2f} ms")

    if debug:
        print(f"OM输出shape: {preds.shape}")
    if preds.shape[1] < preds.shape[2]:
        preds = np.transpose(preds, (0, 2, 1))
    preds = preds[0]
    num_outputs = preds.shape[1]
    is_single_class = (num_outputs == 5)
    num_classes = 1 if is_single_class else num_outputs - 4

    # -- 后处理 (优化7: 进一步向量化)
    t0 = time.time()
    cx, cy, w, h = preds[:, 0], preds[:, 1], preds[:, 2], preds[:, 3]
    
    if is_single_class:
        confs = preds[:, 4]
        cls_ids = np.zeros(len(confs), dtype=np.int32)
    else:
        class_scores = preds[:, 4:]
        cls_ids = np.argmax(class_scores, axis=1)
        confs = np.take_along_axis(class_scores, cls_ids[:, None], axis=1).ravel()

    # 置信度筛选
    mask = confs >= conf_thres
    
    if isinstance(ratio, tuple):
        ratio_w, ratio_h = ratio
    else:
        ratio_w = ratio_h = ratio

    boxes_out = []
    if np.any(mask):
        cx, cy, w, h = cx[mask], cy[mask], w[mask], h[mask]
        confs, cls_ids = confs[mask], cls_ids[mask]
        
        # 坐标解码
        half_w, half_h = w * 0.5, h * 0.5
        x1 = np.clip((cx - half_w - dw) / ratio_w, 0, orig_w)
        y1 = np.clip((cy - half_h - dh) / ratio_h, 0, orig_h)
        x2 = np.clip((cx + half_w - dw) / ratio_w, 0, orig_w)
        y2 = np.clip((cy + half_h - dh) / ratio_h, 0, orig_h)
        
        valid = (x2 > x1) & (y2 > y1)
        if np.any(valid):
            x1, y1, x2, y2 = x1[valid], y1[valid], x2[valid], y2[valid]
            confs, cls_ids = confs[valid], cls_ids[valid]
            
            boxes_for_nms = np.stack((x1, y1, x2, y2), axis=1)
            
            # 按类别NMS
            unique_classes = np.unique(cls_ids)
            for cls in unique_classes:
                cls_mask = (cls_ids == cls)
                this_boxes = boxes_for_nms[cls_mask]
                this_scores = confs[cls_mask]
                keep = nms_fast(this_boxes, this_scores, iou_thres)
                for i in keep:
                    boxes_out.append([
                        float(this_boxes[i, 0]), float(this_boxes[i, 1]),
                        float(this_boxes[i, 2]), float(this_boxes[i, 3]),
                        float(this_scores[i]), int(cls)
                    ])

    t1 = time.time()
    times['postprocess_ms'] = (t1 - t0) * 1000
    if debug:
        print(f"【阶段耗时】后处理(NMS等): {times['postprocess_ms']:.2f} ms")
        print(f"NMS后boxes数量: {len(boxes_out)}")

    t_all1 = time.time()
    times['total_ms'] = (t_all1 - t_all0) * 1000
    print(f"【推理流程总耗时】: {times['total_ms']:.2f} ms")
    for k, v in times.items():
        if k != 'total_ms':
            print(f"  > {k}: {v:.2f} ms")

    if return_times:
        return boxes_out, times
    return boxes_out


def save_result(img_bgr, boxes_out, out_img_path, class_names=None):
    t0 = time.time()
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
    t1 = time.time()
    print(f"检测到 {len(boxes_out)} 个目标，结果保存耗时: {(t1-t0)*1000:.2f} ms")


def release_acl_resource():
    t0 = time.time()
    # 释放预分配的缓冲区
    if ACL_RESOURCE['input_buffer'] is not None:
        acl.rt.free(ACL_RESOURCE['input_buffer'])
    if ACL_RESOURCE['output_buffer'] is not None:
        acl.rt.free(ACL_RESOURCE['output_buffer'])
    if ACL_RESOURCE['input_data'] is not None:
        acl.destroy_data_buffer(ACL_RESOURCE['input_data'])
    if ACL_RESOURCE['output_data'] is not None:
        acl.destroy_data_buffer(ACL_RESOURCE['output_data'])
    if ACL_RESOURCE['input_dataset'] is not None:
        acl.mdl.destroy_dataset(ACL_RESOURCE['input_dataset'])
    if ACL_RESOURCE['output_dataset'] is not None:
        acl.mdl.destroy_dataset(ACL_RESOURCE['output_dataset'])
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
    t1 = time.time()
    print(f"ACL资源已释放，资源释放耗时: {(t1-t0)*1000:.2f} ms")


if __name__ == "__main__":
    IMAGE_PATH = "7000.jpg"
    OUTPUT_IMAGE = "result_om1.jpg"
    DEVICE_ID = 0
    om_path = "weight/best.om"

    try:
        t0 = time.time()
        print("========== 初始化ACL资源 ==========")
        init_acl_resource(device_id=DEVICE_ID)
        t1 = time.time()
        print(f"总耗时 - 初始化ACL资源: {(t1-t0)*1000:.2f} ms\n")

        t0 = time.time()
        print("========== 加载OM模型 ==========")
        load_om_model(om_path)
        t1 = time.time()
        print(f"总耗时 - 加载OM模型: {(t1-t0)*1000:.2f} ms\n")

        t0 = time.time()
        img_bgr = load_image(IMAGE_PATH)
        t1 = time.time()
        print(f"\n图片尺寸: {img_bgr.shape}")
        print(f"总耗时 - 读取图片: {(t1-t0)*1000:.2f} ms\n")

        t0 = time.time()
        print("========== OM推理 (昇腾310) ==========")
        boxes = om_infer(
            om_path,
            img_bgr,
            conf_thres=0.08,
            iou_thres=0.35,
            debug=True
        )
        t1 = time.time()
        print(f"总耗时 - OM推理: {(t1-t0)*1000:.2f} ms\n")

        # ===== 多次推理测试（验证预分配缓冲区效果）=====
        print("\n========== 连续推理测试 ==========")
        for i in range(3):
            t0 = time.time()
            boxes = om_infer(om_path, img_bgr, conf_thres=0.08, iou_thres=0.35, debug=False)
            t1 = time.time()
            print(f"第{i+1}次推理耗时: {(t1-t0)*1000:.2f} ms, 检测框: {len(boxes)}")

        t0 = time.time()
        save_result(img_bgr, boxes, OUTPUT_IMAGE)
        t1 = time.time()
        print(f"总耗时 - 绘制保存: {(t1-t0)*1000:.2f} ms\n")

        print(f"\n结果已保存至: {OUTPUT_IMAGE}")
        print(f"检测框数量: OM={len(boxes)}")

    except Exception as e:
        print(f"\n错误: {e}")
        import traceback
        traceback.print_exc()
    finally:
        print("\n========== 释放资源 ==========")
        release_acl_resource()