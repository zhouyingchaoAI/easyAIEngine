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
import time   # 新增：计时

# 手动定义内存分配类型常量（兼容昇腾acl API变化）
try:
    MEM_MALLOC_NORMAL_ONLY = acl.rt.MEM_MALLOC_NORMAL_ONLY
except AttributeError:
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
        MEMCPY_HOST_TO_DEVICE = 0
        MEMCPY_DEVICE_TO_HOST = 1

ACL_RESOURCE = {
    'context': None,
    'stream': None,
    'model_id': None,
    'model_desc': None,
    'device_id': 0
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
    input_num = acl.mdl.get_num_inputs(model_desc)
    output_num = acl.mdl.get_num_outputs(model_desc)
    t1 = time.time()
    print(f"OM模型加载成功, Model ID: {model_id}")
    print(f"模型输入数量: {input_num}, 输出数量: {output_num}")
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
    """boxes: [N, 4], scores: [N], both ndarray. Returns keep indices in original order."""
    if len(boxes) == 0:
        return []
    # Convert to float if not already
    boxes = boxes.astype(np.float32)
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
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

def om_infer(om_path, img_bgr, conf_thres=0.25, iou_thres=0.45, img_size=1024, debug=True):
    times = {}
    t_all0 = time.time()
    # -- 预处理
    t0 = time.time()
    orig_h, orig_w = img_bgr.shape[:2]
    img_in, ratio, (dw, dh) = letterbox(img_bgr, new_shape=(img_size, img_size))
    img_in = cv2.cvtColor(img_in, cv2.COLOR_BGR2RGB)
    img_in = img_in.transpose(2, 0, 1)
    img_in = np.ascontiguousarray(img_in).astype(np.float32) / 255.0
    img_in = np.expand_dims(img_in, 0)
    t1 = time.time()
    times['preprocess_ms'] = (t1 - t0) * 1000
    if debug:
        print(f"【阶段耗时】预处理: {times['preprocess_ms']:.2f} ms")
        print(f"原始图片尺寸: {img_bgr.shape}")
        print(f"Letterbox后尺寸: {img_in.shape}")
        print(f"缩放比例: {ratio}")
        print(f"Padding: dw={dw}, dh={dh}")

    model_id = ACL_RESOURCE['model_id']
    model_desc = ACL_RESOURCE['model_desc']

    # -- 内存准备阶段
    t0 = time.time()
    input_size = acl.mdl.get_input_size_by_index(model_desc, 0)
    if debug:
        print(f"模型期望输入大小: {input_size} bytes")
        print(f"实际数据大小: {img_in.nbytes} bytes")
    if img_in.nbytes != input_size:
        print(f"警告: 数据大小不匹配! 期望{input_size}, 实际{img_in.nbytes}")
    input_buffer, ret = acl.rt.malloc(input_size, MEM_MALLOC_NORMAL_ONLY)
    if ret != 0:
        raise Exception(f"申请输入内存失败, 错误码: {ret}")
    input_bytes = img_in.tobytes()
    input_ptr = acl.util.bytes_to_ptr(input_bytes)
    ret = acl.rt.memcpy(input_buffer, input_size, input_ptr, len(input_bytes), MEMCPY_HOST_TO_DEVICE)
    if ret != 0:
        acl.rt.free(input_buffer)
        raise Exception(f"拷贝输入数据失败, 错误码: {ret}")
    input_dataset = acl.mdl.create_dataset()
    input_data = acl.create_data_buffer(input_buffer, input_size)
    ret = acl.mdl.add_dataset_buffer(input_dataset, input_data)
    if isinstance(ret, tuple):
        actual_ret = ret[1] if len(ret) > 1 else ret[0]
        if debug:
            print(f"add_dataset_buffer返回元组: {ret}, 使用ret_code: {actual_ret}")
    else:
        actual_ret = ret
    if actual_ret != 0 and debug:
        print(f"警告: 添加输入buffer返回非0值: {actual_ret}")
    output_size = acl.mdl.get_output_size_by_index(model_desc, 0)
    if debug:
        print(f"模型输出大小: {output_size} bytes")
    output_buffer, ret = acl.rt.malloc(output_size, MEM_MALLOC_NORMAL_ONLY)
    if ret != 0:
        acl.rt.free(input_buffer)
        acl.destroy_data_buffer(input_data)
        acl.mdl.destroy_dataset(input_dataset)
        raise Exception(f"申请输出内存失败, 错误码: {ret}")
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
    t1 = time.time()
    times['memcopy_ms'] = (t1 - t0) * 1000
    if debug:
        print(f"【阶段耗时】输入内存分配及复制: {times['memcopy_ms']:.2f} ms")

    # -- 推理阶段
    output_np = None
    t0 = time.time()
    if debug:
        print("开始执行模型推理...")
    try:
        ret = acl.mdl.execute(model_id, input_dataset, output_dataset)
        if ret != 0:
            raise Exception(f"模型推理失败, 错误码: {ret}")
        t1 = time.time()
        times['inference_ms'] = (t1 - t0) * 1000
        if debug:
            print("模型推理成功")
            print(f"【阶段耗时】NPU推理: {times['inference_ms']:.2f} ms")
        # -- 输出拷贝与解析阶段
        t0 = time.time()
        if debug:
            print("开始获取输出数据...")
        try:
            import ctypes
            host_buffer = (ctypes.c_byte * output_size)()
            host_ptr = ctypes.cast(host_buffer, ctypes.c_void_p).value
            ret = acl.rt.memcpy(host_ptr, output_size, output_buffer, output_size, MEMCPY_DEVICE_TO_HOST)
            if ret != 0:
                raise Exception(f"拷贝输出数据到Host失败, 错误码: {ret}")
            if debug:
                print("成功拷贝输出数据到Host")
            output_np = np.frombuffer(host_buffer, dtype=np.float32).copy()
        except Exception as e:
            if debug:
                print(f"方案1失败: {e}, 尝试方案2")
            try:
                host_ptr, ret = acl.rt.malloc_host(output_size)
                if ret != 0:
                    raise Exception(f"分配Host内存失败, 错误码: {ret}")
                ret = acl.rt.memcpy(host_ptr, output_size, output_buffer, output_size, MEMCPY_DEVICE_TO_HOST)
                if ret != 0:
                    acl.rt.free_host(host_ptr)
                    raise Exception(f"拷贝数据失败, 错误码: {ret}")
                if debug:
                    print("方案2: 使用malloc_host成功")
                output_np = acl.util.ptr_to_numpy(host_ptr, (output_size // 4,), np.float32).copy()
                acl.rt.free_host(host_ptr)
            except Exception as e2:
                if debug:
                    print(f"方案2失败: {e2}, 尝试方案3")
                output_ptr = acl.get_data_buffer_addr(output_data)
                output_np = acl.util.ptr_to_numpy(output_ptr, (output_size // 4,), np.float32).copy()
        output_dims = acl.mdl.get_output_dims(model_desc, 0)
        if isinstance(output_dims, dict):
            dims = output_dims['dims']
        elif isinstance(output_dims, tuple):
            if len(output_dims) == 2:
                dims_info, ret = output_dims
                if ret != 0:
                    raise Exception(f"获取输出维度失败, 错误码: {ret}")
                if isinstance(dims_info, dict):
                    dims = dims_info['dims']
                else:
                    dims = dims_info
            else:
                dims = output_dims
        else:
            dims = output_dims
        if debug:
            print(f"输出维度信息: {dims}")
        if hasattr(dims, '__iter__'):
            dims = list(dims)
        preds = output_np.reshape(dims)
        t1 = time.time()
        times['output_ms'] = (t1-t0)*1000
        if debug:
            print(f"【阶段耗时】模型输出搬运和reshape: {times['output_ms']:.2f} ms")
    finally:
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
    if output_np is None:
        raise Exception("未能成功获取输出数据")
    if debug:
        print(f"OM输出shape: {preds.shape}")
    if preds.shape[1] < preds.shape[2]:
        preds = np.transpose(preds, (0, 2, 1))
    preds = preds[0]
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
    # -- 后处理 (优化: 使用numpy矢量化和批量NMS)
    t0 = time.time()
    # 1. 批量处理所有框和conf
    if is_single_class:
        cx = preds[:, 0]
        cy = preds[:, 1]
        w = preds[:, 2]
        h = preds[:, 3]
        confs = preds[:, 4]
        cls_ids = np.zeros_like(confs, dtype=np.int32)
        class_scores = None
    else:
        cx = preds[:, 0]
        cy = preds[:, 1]
        w = preds[:, 2]
        h = preds[:, 3]
        class_scores = preds[:, 4:]
        cls_ids = np.argmax(class_scores, axis=1)
        confs = class_scores[np.arange(len(class_scores)), cls_ids]
    # 筛选置信度
    mask = confs >= conf_thres
    if isinstance(ratio, tuple):
        ratio_w, ratio_h = ratio
    else:
        ratio_w = ratio_h = ratio
    if np.any(mask):
        cx, cy, w, h, confs, cls_ids = cx[mask], cy[mask], w[mask], h[mask], confs[mask], cls_ids[mask]
        # x1y1x2y2解码
        x1 = (cx - w / 2 - dw) / ratio_w
        y1 = (cy - h / 2 - dh) / ratio_h
        x2 = (cx + w / 2 - dw) / ratio_w
        y2 = (cy + h / 2 - dh) / ratio_h
        x1 = np.clip(x1, 0, orig_w)
        y1 = np.clip(y1, 0, orig_h)
        x2 = np.clip(x2, 0, orig_w)
        y2 = np.clip(y2, 0, orig_h)
        valid = (x2 > x1) & (y2 > y1)
        x1, y1, x2, y2, confs, cls_ids = x1[valid], y1[valid], x2[valid], y2[valid], confs[valid], cls_ids[valid]
        # 构建boxes_for_nms: [N, 4], scores: [N], cls_ids: [N]
        boxes_for_nms = np.stack((x1, y1, x2, y2), axis=1)
        scores_for_nms = confs
        cls_ids_for_nms = cls_ids
    else:
        boxes_for_nms = np.zeros((0, 4), dtype=np.float32)
        scores_for_nms = np.array([], dtype=np.float32)
        cls_ids_for_nms = np.array([], dtype=np.int32)
    # 输出置信度统计
    if debug:
        all_max_confs = confs if np.any(mask) else np.array([])
        print(f"\n置信度统计:")
        if all_max_confs.size > 0:
            print(f"  - 最大值: {all_max_confs.max():.4f}")
            print(f"  - 均值: {all_max_confs.mean():.4f}")
            print(f"  - >0.25的数量: {(all_max_confs > 0.25).sum()}")
        else:
            print("  - 无有效置信目标")
        print(f"置信度过滤后boxes数量: {boxes_for_nms.shape[0]}")
    boxes_out = []
    if boxes_for_nms.shape[0] > 0:
        # 对每个类别做nms
        unique_classes = np.unique(cls_ids_for_nms)
        for cls in unique_classes:
            cls_mask = (cls_ids_for_nms == cls)
            this_boxes = boxes_for_nms[cls_mask]
            this_scores = scores_for_nms[cls_mask]
            this_cls_ids = np.full((this_boxes.shape[0],), cls)
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
        print(f"【推理流程总耗时】: {(t_all1-t_all0)*1000:.2f} ms")
        for k, v in times.items():
            print(f"    > {k}: {v:.2f} ms")
    return boxes_out

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
    YOLO_CKPT = "/cv_space/NWPU-Crowd/runs/exp_yolo11x3/weights/best.pt"
    IMAGE_PATH = "7000.jpg"
    OUTPUT_IMAGE = "result_om1.jpg"
    DEVICE_ID = 0  # 昇腾设备ID
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
