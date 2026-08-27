# ARM Docker 部署与启动（算法管理服务）

本文用于在 ARM + Ascend 环境启动 `easyAIEngine` 算法管理服务。容器名称、端口规划与当前项目的多实例调度逻辑保持一致。

## 1. 前置条件

- ARM 主机已安装 Docker 和 Ascend 驱动；驱动目录为 `/usr/local/Ascend/driver`。
- 已获取 Harbor 中包含 Python、CANN/ACL 运行时与项目依赖的 ARM 镜像：`harbor.chencytech.com/chency-ai/easyaiengine:v1-arm64`。
- 将本仓库克隆到主机目录，例如 `/opt/rail-passenger-flow/code`。

> 镜像与 Ascend 驱动版本必须兼容。部署前可执行 `docker run --rm --privileged -v /usr/local/Ascend/driver:/usr/local/Ascend/driver:ro harbor.chencytech.com/chency-ai/easyaiengine:v1-arm64 npu-smi info` 进行确认。

## 2. 启动算法容器

下面命令会创建算法运行容器，并映射管理端口与自动分配给算法实例的端口范围：

```bash
export PROJECT_ROOT=/opt/rail-passenger-flow/code
export ALGORITHM_IMAGE=harbor.chencytech.com/chency-ai/easyaiengine:v1-arm64

docker run -d \
  --name head_detect_new \
  --restart unless-stopped \
  --privileged \
  -p 9022:22 \
  -p 7900-7999:7900-7999 \
  -v "${PROJECT_ROOT}:/code" \
  -v /usr/local/Ascend/driver:/usr/local/Ascend/driver:ro \
  -v /usr/local/sbin:/usr/local/sbin:ro \
  "${ALGORITHM_IMAGE}" \
  bash
```

## 3. 启动算法管理平台

管理平台使用 `7900` 端口；算法实例由平台自动从 `7901-7999` 分配端口。请先启动平台，再从界面的“实例管理”创建实时人数统计或绊线统计实例。

```bash
docker exec -d head_detect_new sh -lc '
  cd /code/predict &&
  mkdir -p logs &&
  exec env ALGORITHM_MANAGER_PORT=7900 python3 -u algorithm_manager.py \
    >> logs/algorithm_manager.log 2>&1
'
```

访问地址：`http://<服务器IP>:7900`

## 4. 服务校验与运维命令

```bash
# 查看容器与管理平台日志
docker ps --filter name=head_detect_new
docker exec head_detect_new tail -n 100 /code/predict/logs/algorithm_manager.log

# 检查管理页面可访问性
curl -f http://127.0.0.1:7900/

# 查看 Ascend NPU 状态
docker exec head_detect_new npu-smi info

# 重启或停止容器
docker restart head_detect_new
docker stop head_detect_new
```

## 5. 端口说明

| 端口 | 用途 |
| --- | --- |
| 7900 | 算法管理平台 Web 页面 |
| 7901-7999 | 实时人数统计、绊线统计等算法实例端口（由平台自动分配） |
| 9022 | 容器 SSH（仅在确有运维需要时使用） |

## 注意事项

- 请不要将摄像机地址、账号密码、平台访问令牌或现场配置文件提交到仓库。
- 算法实例需填写智能客流识别平台的访问地址，格式为 `http://<平台服务器IP>:5066`，以便完成注册和调用。
- 首次部署前，请确认模型文件已位于 `predict/weight/`，并与 Ascend 环境匹配。
