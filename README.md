# README

## 启动速度修复

`cu130-megapak-pt211` 在构建阶段安装 TIPO 与 Discord voice 依赖，修正 LLM_party 的 extras 包检测及 Manager 的系统包查询。已有节点卷会在启动时应用同样的兼容补丁，保留用户配置。TIPO 默认禁止自动安装依赖，可显式设置 `TIPO_NO_AUTO_INSTALL=0` 恢复上游行为。

修改需要重新构建镜像并重建容器后生效：

```bash
make build-cu130-megapak-pt211
```

Makefile 测试默认复用 `~/.local/share/comfyui-custom/tests/<镜像标签>`，可用 `TEST_STORAGE_ROOT` 指定其他目录；只有设置 `COLD_START=true` 才新建独立存储。启动日志新增 `[STARTUP]` 分阶段计时，镜像新增 HTTP 健康检查。

### Pixaroma 图片扫描与 ToolBox 基础节点

`cu130-megapak-pt211` 默认停用 `PixaromaLoadImage`、`PixaromaLoadImageMini`、`PixaromaPromptReader`，避免这些节点在首页查询节点信息时递归扫描素材目录。其他 Pixaroma 功能及 `PixaromaLoadImagesFolder` 批量加载节点保留，Crystools 也保留。

首次启动将镜像中的禁用清单复制到 `/root/user-scripts/disabled-nodes.yaml`（默认挂载为 `storage-user/user-scripts/disabled-nodes.yaml`），并自动追加 `--disabled-nodes-config`。已有清单、`pre-start.sh` 和 `comfy.settings.json` 不会被这一步覆盖；用户脚本或 `CLI_ARGS` 已指定禁用清单时优先使用指定路径，不重复追加参数。清单内容就是生效配置，需要恢复某节点时从清单移除对应名称并重启；`disabled_nodes: []` 可恢复全部三个节点。

使用这些 Pixaroma 节点的已有工作流需先替换节点。停用只阻止节点注册，不跳过整个插件包的导入，也不能消除其他插件的素材扫描。

在线构建将 ToolBox 固定到 `f2ff93b75b52bfc258463a37f046f1aff6006b93`，包含 `tb-switch-any`、`tb-switch-image`、`tb-primitive-integer`、`tb-primitive-string-multiline`。离线构建需自行将 `cu130-megapak-pt211/offline-cache/repos/ComfyUI-ToolBox` 更新到该提交（或包含它的后续版本）；已有持久化节点卷仍按原有规则保留，不会自动替换其源码。上述节点迁移不自动改写工作流。

此项目变更需重新构建镜像并重建容器后生效；不要求改写已有用户脚本。

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

## Spark H3 后端

核心 `comfy-kitchen` 沿用 ComfyUI 要求的版本。Spark 的 SM120 / CUDA 13.0 定制算子以 `comfyui_spark_cuda` 私有包提供，不替换核心依赖或注册为全局 CUDA 后端；已有 Spark 节点目录在启动时应用相同适配。

离线构建需将 `comfy_kitchen-0.2.37+spark.h3.sm120.cu130.1-cp312-abi3-linux_x86_64.whl` 放入 `cu130-megapak-pt211/offline-cache/spark/`。在线和离线构建均校验固定 SHA256；不要将该旧 wheel 全局 pip 安装到依赖 0.2.39 的环境。
