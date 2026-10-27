# README

## 启动速度修复

`cu130-megapak-pt211` 在构建阶段安装 TIPO 与 Discord voice 依赖，修正 LLM_party 的 extras 包检测及 Manager 的系统包查询。已有节点卷会在启动时应用同样的兼容补丁，保留用户配置。TIPO 默认禁止自动安装依赖，可显式设置 `TIPO_NO_AUTO_INSTALL=0` 恢复上游行为。

修改需要重新构建镜像并重建容器后生效：

```bash
make build-cu130-megapak-pt211
```

Makefile 测试默认复用 `~/.local/share/comfyui-custom/tests/<镜像标签>`，可用 `TEST_STORAGE_ROOT` 指定其他目录；只有设置 `COLD_START=true` 才新建独立存储。启动日志新增 `[STARTUP]` 分阶段计时，镜像新增 HTTP 健康检查。

详细诊断、迁移和验证记录见 [启动分析](cu130-megapak-pt211/STARTUP_ANALYSIS.md)。

## 项目介绍
    本项目是一个基于 [yanwk/ComfyUI](https://github.com/yanwk/ComfyUI)的自定义节点集合，用于扩展 ComfyUI 功能。

## 可选3DGS运行依赖

ToolBox始终提供3DGS节点、工作流和文档。`make build-cu130-megapak-pt211 INSTALL_3DGS=true` 在构建时从Wheels-Containerfiles的固定Release安装PyCOLMAP、Brush和Vulkan Loader，使用独立`-3dgs`标签；默认false不安装这些运行依赖。模型和设备驱动由用户环境提供。[构建与离线材料说明](cu130-megapak-pt211/3DGS.md)。

## FlashVSR 视频超分依赖

cu130-megapak-pt211 集成 FlashVSR+ Turbo / Triton 和本地 2K、4K、分块 8K 工作流所需依赖，容器启用 NVIDIA `compute,utility,video` 能力。[依赖分工、模型挂载与容器重建说明](cu130-megapak-pt211/FLASHVSR.md)。
