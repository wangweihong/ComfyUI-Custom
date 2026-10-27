# README

## 启动速度修复

`cu130-megapak-pt211` 在构建阶段安装 TIPO 与 Discord voice 依赖，修正 LLM_party 的 extras 包检测及 Manager 的系统包查询。已有节点卷会在启动时应用同样的兼容补丁，保留用户配置。TIPO 默认禁止自动安装依赖，可显式设置 `TIPO_NO_AUTO_INSTALL=0` 恢复上游行为。

修改需要重新构建镜像并重建容器后生效：

```bash
make build-cu130-megapak-pt211
```

Makefile 测试默认复用 `~/.local/share/comfyui-custom/tests/<镜像标签>`，可用 `TEST_STORAGE_ROOT` 指定其他目录；只有设置 `COLD_START=true` 才新建独立存储。启动日志新增 `[STARTUP]` 分阶段计时，镜像新增 HTTP 健康检查。

## 项目介绍
    本项目是一个基于 [yanwk/ComfyUI](https://github.com/yanwk/ComfyUI)的自定义节点集合，用于扩展 ComfyUI 功能。

## 可选3DGS运行依赖

ToolBox 提供 3DGS 节点、工作流和使用文档。构建时设置 `INSTALL_3DGS=true` 安装 PyCOLMAP、Brush 和 Vulkan Loader，镜像标签增加 `-3dgs`；默认 `false`。模型和设备驱动由用户提供，启动时不安装依赖。

```bash
make build-cu130-megapak-pt211 INSTALL_3DGS=true

# 或使用 Compose：
cd cu130-megapak-pt211
docker compose -f docker-compose.yml -f docker-compose.3dgs.yml build
```

离线材料从 [Wheels-Containerfiles 的 3dgs-v1 Release](https://github.com/wangweihong/Wheels-Containerfiles/releases/tag/3dgs-v1) 获取，放入 `cu130-megapak-pt211/offline-cache/3dgs/`：

- `pycolmap-4.2.1-cp313-cp313-manylinux_2_28_x86_64.whl`
- `brush-app-x86_64-unknown-linux-gnu.tar.xz`
- `vulkan-loader-1.4.328.1-1.el10.x86_64.rpm`

在仓库根目录执行 `make build-cu130-megapak-pt211 INSTALL_3DGS=true OFFLINE_MODE=true`。构建会按 [3dgs.sha256](cu130-megapak-pt211/3dgs.sha256) 校验材料，缺失或校验失败时终止；其他节点和依赖仍需按项目原有离线流程准备。

## FlashVSR 视频超分依赖

cu130-megapak-pt211 集成 FlashVSR+ Turbo / Triton 和本地 2K、4K、分块 8K 工作流所需依赖，容器启用 NVIDIA `compute,utility,video` 能力。[依赖分工、模型挂载与容器重建说明](cu130-megapak-pt211/FLASHVSR.md)。
