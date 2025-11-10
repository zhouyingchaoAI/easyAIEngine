# easyAIEngine

easyAIEngine 是一套基于 YOLOv11 的人群分析解决方案，包含实时检测与绊线统计两类算法服务，并可选配 Web 管理界面。项目已对源码、脚本与文档进行重新整理，提供更清晰的目录结构与上手流程。

## 目录结构

```
.
├── algorithm_service.py              # 实时检测/人数统计算法服务
├── algorithm_service_line_crossing.py# 绊线统计算法服务
├── algorithm_manager.py              # 管理界面后端（如需）
├── configs/                          # 配置模板与示例
├── docs/                             # 使用指南与进阶文档
├── logs/                             # 运行时日志（保留目录，默认空）
├── scripts/                          # 启动、维护相关脚本
├── tests/                            # 自测脚本
├── weight/                           # 模型权重（`.om`/`.pt`）
└── requirements.txt                  # Python 依赖
```

> 所有历史的 Markdown 文档现已集中在 `docs/` 目录中，日志与临时文件已经清理，仅保留必要的示例配置与模型权重。

## 快速开始

1. **准备环境**
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   ```

2. **启动服务**
   - Web 管理界面（可选）：`./scripts/start_manager.sh`
   - 实时检测服务：`./scripts/start_algorithm_service.sh`
   - 绊线统计服务：`./scripts/start_line_crossing_service.sh`

   > 所有脚本均支持 `--help` 查看自定义参数（端口、GPU、注册选项等）。

3. **访问接口**
   - 实时检测：`POST http://<host>:7902/infer`
   - 绊线统计：`POST http://<host>:7903/infer`
   - 健康检查：`GET http://<host>:<port>/health`

更多 API 调用说明参考 `docs/服务说明.md` 与 `docs/告警机制说明.md`。

## 日志与配置

- 默认日志目录为 `logs/`，已通过 `.gitkeep` 保留结构但不再追踪历史日志。
- 若需自定义算法区域、置信度阈值等，可在请求体中附带 `algo_config`，或参考 `configs/` 下的示例。
- 运行过程中产生的模型 checkpoint、输出图片建议存放到单独目录，避免再次进入版本控制。

## 测试

项目提供基础的自测脚本，确保服务部署后功能正常：

```bash
python tests/test_algorithm_service.py --base-url http://localhost:7902
python tests/test_batch_inference.py --base-url http://localhost:7902 --image-url <图片地址>
python tests/test_region_filter.py
```

## 文档索引

常用文档已归档至 `docs/`：

- `快速开始.md`：部署与运行指引
- `端口速查.md`：核心服务端口与常用命令
- `GPU配置说明.md`：多 GPU/Ascend 环境配置
- `绊线告警机制详解.md`：绊线统计内部流程
- `日志管理说明.md`：日志巡检、清理与诊断

更多内容请查阅 `docs/README` 或根据文件名快速定位。

---

如需为项目贡献新特性或文档，建议遵循当前目录结构并在提交前运行测试脚本，保持仓库整洁。*** End Patch
