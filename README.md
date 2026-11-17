# README

## 启动速度修复

`cu130-megapak-pt211` 在构建阶段安装 TIPO 与 Discord voice 依赖，修正 LLM_party 的 extras 包检测及 Manager 的系统包查询。已有节点卷会在启动时应用同样的兼容补丁，保留用户配置。TIPO 默认禁止自动安装依赖，可显式设置 `TIPO_NO_AUTO_INSTALL=0` 恢复上游行为。

启动检查所需包直接声明于 `cu130-megapak-pt211/builder-scripts/pak1.txt`：`tipo-kgen` 提供 `kgen`，`py-cord[voice]` 提供 `discord`，并固定 `PyNaCl`（`nacl`）与 `davey`。构建使用运行 ComfyUI 的同一 Python 3.13 安装到系统环境，不依赖基础镜像是否预装这些包。

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

核心 `comfy-kitchen` 沿用 ComfyUI 要求的版本。Spark 采用官方独立发行包 `spark-h3-kitchen`，使用 `spark_h3_kitchen` 命名空间，完整提供 Spark、INT8 attention 和 CUDA helper API，不替换核心依赖。在线源码固定为 `8edfc07033d97dce29e83f51749d2c435556b91b`，不再跟随主分支漂移。

构建使用 [Spark-H3 v0.2.0 官方发布包](https://github.com/zechengtang/Spark-H3/releases/tag/comfyui-v0.2.0) 中的 SM120 / CUDA 13.0 / `0.2.37` 隔离后端。在线和离线构建均校验 ZIP 的固定 SHA256，并使用 `--no-index --no-deps` 安装完整 wheel。构建时检查核心导入、Spark 原生入口以及 INT8 接口；无需 GPU 即可执行该导入检查。

离线构建需将 `ComfyUI-Spark-H3-0.2.0-linux-x86_64-sm120-cu130.zip` 放入 `cu130-megapak-pt211/offline-cache/spark/`，并将离线 `Spark-H3` 源码快照更新到上述固定提交。旧 `comfy_kitchen-0.2.37+spark.h3...whl` 不再作为构建材料。

启动时兼容持久化卷中的旧 Spark：把旧核心导入或旧 `comfyui_spark_cuda` 导入改接到官方隔离后端；新版隔离导入保持原样。未知或重复的导入结构仍明确报错，防止构建成功却接错后端。已有节点卷不会自动升级其他 Spark 源码或改写工作流；需要新版功能时需单独升级该卷中的插件。
