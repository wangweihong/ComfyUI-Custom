# README
## 项目介绍
    本项目是一个基于 [yanwk/ComfyUI](https://github.com/yanwk/ComfyUI)的自定义节点集合，用于扩展 ComfyUI 功能。

## 可选3DGS运行依赖

ToolBox始终提供3DGS节点、工作流和文档。`make build-cu130-megapak-pt211 INSTALL_3DGS=true` 在构建时从Wheels-Containerfiles的固定Release安装PyCOLMAP、Brush和Vulkan Loader，使用独立`-3dgs`标签；默认false不安装这些运行依赖。模型和设备驱动由用户环境提供。[构建与离线材料说明](cu130-megapak-pt211/3DGS.md)。
