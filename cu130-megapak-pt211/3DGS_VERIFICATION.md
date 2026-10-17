# 三仓库3DGS迁移验证

验证日期：2026-10-17。当前组件只安装运行依赖；节点、工作流和文档均由ToolBox提供。没有推送镜像，也没有修改、启动或替换原8188容器；本轮开始时该容器已经停止。

## 已发布材料

- ToolBox节点与资源提交：`1a131682cfe467f2211626007c63c7678f6c3db9`，位于[master分支](https://github.com/wangweihong/ComfyUI-ToolBox/tree/master)。Custom预载脚本固定此提交。
- Wheels发行配方提交：`8d38da08186f9ce8514e436ff2695a1fba17328d`，位于[master分支](https://github.com/wangweihong/Wheels-Containerfiles/tree/master)。
- [3dgs-v1 Release](https://github.com/wangweihong/Wheels-Containerfiles/releases/tag/3dgs-v1)已公开，包含PyCOLMAP wheel、Brush tar.xz、Vulkan Loader RPM、SHA256SUMS和SOURCE_MANIFEST.json。五个文件发布前后均回读验证。

ToolBox/Wheels源码已按用户要求合并并推送至master；初次发布的独立功能分支保留。ToolBox原有README、节点映射、locale及其他未提交功能未混入发布提交；本地原有修改保留。Custom中原有grok.py及入口脚本未改动。

## 自动与构建检查

| 检查 | 结果 |
| --- | --- |
| ToolBox节点、缺依赖、环境继承、工作流和文档 | 6项测试通过 |
| Wheels下载校验、损坏拒绝、混合资产及禁止覆盖 | 6项测试通过 |
| Custom开关、默认值、标签、清理与状态 | 5项测试通过 |
| 在线安装 | 使用真实3dgs-v1 URL通过wget下载并校验三份发行文件，成功 |
| 离线安装 | Docker构建网络为none，从本地材料安装成功 |
| 缺失、损坏材料 | 实际Docker构建按预期失败，不回退联网 |
| 非法INSTALL_3DGS | Make拒绝；即使同名外部镜像存在，Docker最终阶段仍拒绝 |
| Compose | 默认false和覆盖文件true的参数、标签匹配 |
| 无GPU依赖检查 | PyCOLMAP4.2.1 CPU导入与global/incremental接口、Brush0.3.0版本/帮助、动态链接及Vulkan Loader加载均通过 |

基础NumPy2.3.2、Pillow12.3.0、OpenCV4.13.0、PyTorch2.11.0+cu130没有被3DGS安装替换。Vulkan RPM保留其LICENSE.txt，Brush保留原LICENSE，PyCOLMAP wheel保留COPYING.txt。基础镜像没有xz命令，Dockerfile使用Python标准库直接解压归档，没有增加额外软件包。

## 完整生产构建

两个Make构建均完整结束、退出0：

```bash
make build-cu130-megapak-pt211 INSTALL_3DGS=false IMAGE_NAME=codex/comfyui-3dgs-split-verify
make build-cu130-megapak-pt211 INSTALL_3DGS=true IMAGE_NAME=codex/comfyui-3dgs-split-verify
```

本地镜像前缀为`registry.cn-hangzhou.aliyuncs.com/codex/comfyui-3dgs-split-verify`：

| 标签（另有同名-20261017日期标签） | 本地manifest list摘要 |
| --- | --- |
| cu130-megapak-pt211 | 7b96b5c5ce03a1b4060711d0caa16c5a263f38438ec9c9f1515d0ce69ebf6046 |
| cu130-megapak-pt211-3dgs | d222ebf7172c1380c2bce99a7a0e7983098c01de24fdd0c797cd42eea9ecebe2 |

两份镜像的17个节点源码、工作流、文档和图片摘要均与ToolBox源文件匹配，ToolBox Git提交也匹配。关闭镜像无PyCOLMAP、/opt/3dgs及旧独立插件；节点和资源存在。启用镜像从真实Release下载运行依赖。原features/3dgs代码、资源与Python安装器已移除。

## 独立功能实测

先复用既有公共镜像、拉取已发布ToolBox提交，并逐字复用当前Dockerfile资源与可选阶段构建独立测试镜像；从真实Release在线安装后运行完整01工作流。随后使用上述完整生产启用镜像验证注册、03/04复用和抽帧。

| 项目 | 结果 |
| --- | --- |
| 关闭依赖、无GPU启动 | 四节点及原有ToolBoxLatentSize成功注册，python_module均来自ComfyUI-ToolBox |
| 缺PyCOLMAP/Brush执行 | 分别返回明确的安装提示，其他节点仍可用 |
| 完整H3 → COLMAP → Brush → 原生预览 | 成功，277.179秒 |
| COLMAP | CPU/global/exhaustive，共享SIMPLE_PINHOLE，相机70/70，9456个点 |
| 重投影误差与耗时 | 0.91292像素，29.312秒 |
| Brush | 5000步，31.007秒，导出32187个高斯 |
| PLY | binary_little_endian，7597682字节；原生预览通过/view实际取回文件 |
| 最终生产镜像03/04 | 复用已有数据集训练和已有PLY预览均成功 |
| 最终生产镜像抽帧 | 返回(4,768,768,3)的float32张量 |
| 视频ffprobe | 768×768，141帧，24fps |

测试使用独立8189端口、独立输出、只读模型挂载，仅加载ToolBox和必需的KJNodes；不将其他原有第三方节点的默认启动列为已验收项目。测试机的无窗口Vulkan ICD配置仅在测试容器外提供，没有加入仓库、镜像或节点代码；没有打包任何设备驱动。实际硬件可训练条件仍需用户环境满足。

测试容器在验证完成后停止、移除。测试输出和日志保留在本机独立临时目录，未打包进镜像。几何质量仍受生成视角一致性、截断和漂浮高斯限制，运行成功不代表成品质量合格。
