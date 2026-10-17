# 可选3DGS运行依赖

本项目只负责组装镜像，不维护3DGS节点源码或另设Python安装器。

| 项目 | 职责 |
| --- | --- |
| ComfyUI-ToolBox | 四个节点、四套工作流、两张示例图、原理与操作文档 |
| Wheels-Containerfiles | 3dgs-v1 Release：官方PyCOLMAP wheel、Brush归档和Vulkan Loader RPM |
| ComfyUI-Custom | Dockerfile直接wget、SHA256校验和安装；预置ToolBox示例资源 |

```bash
make build-cu130-megapak-pt211 INSTALL_3DGS=false
make build-cu130-megapak-pt211 INSTALL_3DGS=true
```

INSTALL_3DGS只接受true或false，默认false。两种变体都有ToolBox节点、工作流和说明；false不安装本功能的三个原生运行依赖。节点在执行时提示缺少PyCOLMAP或Brush，不在导入或启动时安装依赖。抽帧、PLY加载仍可使用。

启用版使用cu130-megapak-pt211-3dgs和相应日期标签，关闭版保留原标签。Make构建、测试、清理和状态入口使用同一标签规则；Actions手动输入只控制依赖。Compose使用覆盖文件同时选择正确的参数和标签：

```bash
cd cu130-megapak-pt211
docker compose -f docker-compose.yml build
docker compose -f docker-compose.yml -f docker-compose.3dgs.yml build
```

## 固定发行材料

从[Wheels-Containerfiles的3dgs-v1](https://github.com/wangweihong/Wheels-Containerfiles/releases/tag/3dgs-v1)下载PyCOLMAP4.2.1 CPU / cp313 / Linux amd64 wheel、Brush0.3.0官方Linux程序，以及Rocky Linux10的通用Vulkan Loader RPM。本仓库3dgs.sha256固定预期摘要；不只信任与下载物一起获取的校验文件。更新发行材料必须同步版本和摘要。

PyCOLMAP使用已有NumPy，pip安装带--no-deps，不改变PyTorch或CUDA。Brush安装在/opt/3dgs/bin/brush_app，保留原归档许可证；依赖版本摘要见/opt/3dgs/manifest.json和SHA256SUMS。构建时执行CPU导入、重建接口、Brush版本/帮助及动态链接检查，不需要GPU。

## 离线材料

将同一Release的三个原始文件准备到offline-cache/3dgs，然后运行：

```bash
make build-cu130-megapak-pt211 INSTALL_3DGS=true OFFLINE_MODE=true
```

缺件或SHA256错误立即失败，不回退联网。RPM从已校验文件安装，禁用远程仓库。此承诺仅适用于本功能，项目原有其他构建步骤的离线行为不改造。原有节点离线缓存中的ToolBox也须更新到上述固定提交；只准备三个运行依赖不代表整个项目已具备离线材料。准备命令见[离线材料说明](offline-cache/3dgs/README.md)。

## 节点与资源

构建固定ToolBox提交为`1a131682cfe467f2211626007c63c7678f6c3db9`，更新该提交需重新检查接口和资源。节点只从预装ComfyUI-ToolBox加载，不再复制独立ComfyUI-Local-3DGS插件。类型名和接口保留，旧工作流可继续使用。升级已有环境时应由用户停用旧独立插件，避免重复注册；镜像不接管已有节点或持久化卷。

构建时从ToolBox复制工作流、docs和示例图片到预装ComfyUI的user/default/workflows及input，沿用项目首次初始化机制。工作流目录为006视频生成/minimaxh3/单图转3DGS_COLMAP_Brush，完整说明在其docs/README.md中。03/04没有历史路径，须选择当前output内已有数据集或PLY。

H3权重继续由用户挂载。通用Vulkan Loader不是设备驱动；Brush训练仍需实际环境提供可用的Vulkan设备与驱动。节点不锁定厂商、驱动版本、ICD或本机库目录，继承用户提供的环境。首次只支持当前组件的Linux amd64、Python3.13组合。

## 检查

```bash
python3 -m unittest discover -s cu130-megapak-pt211/tests -v
```

节点及工作流测试在ToolBox，发行材料测试在Wheels-Containerfiles。完整测试使用独立镜像与端口，不替换当前8188服务，不推送镜像。

实际构建、Release和功能测试结果见[迁移验证记录](3DGS_VERIFICATION.md)。
