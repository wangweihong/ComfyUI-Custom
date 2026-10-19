# FlashVSR 镜像依赖

此组件补齐本地 2K / 4K / 分块 8K 工作流使用的 FlashVSR+ Turbo / Triton 依赖。

## 文件分工

- `builder-scripts/preload-nodes.sh`：下载 FlashVSR 并固定到已验证提交 `153cf03343a116c85aee40d291125921cc33935a`；基础镜像缺少 VideoHelperSuite 时补装已验证版本。
- `builder-scripts/pak1.txt`：安装 FlashVSR 推理、视频预览和本地编排/资源采样所需的 Python 包。已有的 OpenCV、ModelScope、Transformers 复用现有声明。
- [Wheels-Containerfiles/accelerator/flashvsr-attn](https://github.com/wangweihong/Wheels-Containerfiles/tree/master/accelerator/flashvsr-attn)：保存固定版本的 Triton 算子源码并构建 `flashvsr-attn 0.1.0` 通用 wheel，发布到该仓库的 [flashvsr-attn Release](https://github.com/wangweihong/Wheels-Containerfiles/releases/tag/flashvsr-attn)。
- `Dockerfile`：从上述 Release 安装 wheel（URL 固定 SHA256），安装系统 FFmpeg（含 ffprobe），设置 `NVIDIA_DRIVER_CAPABILITIES=compute,utility,video`。PyTorch / CUDA / Triton / flash-attn 沿用匹配的基础镜像版本。
- Compose 与根目录 Makefile 的两种测试容器启动命令均传入相同的 NVIDIA 能力设置；3DGS 变体也会继承。

这条 Plus / Triton 推理链路无需额外安装 `block_sparse_attn` 或 `pytorch-lightning`。
这里没有安装上游完整训练环境，也不替换基础镜像中的 CUDA 专用 wheel。

## 构建和运行

在项目根目录构建：

```bash
make build-cu130-megapak-pt211
```

在组件目录使用本地构建的镜像重建容器：

```bash
docker compose up -d --build --pull never --force-recreate
```

`video` 由 NVIDIA Container Toolkit 在创建容器时用于挂载 NVENC/NVDEC 驱动库，
因此已有容器需要重新创建，普通 restart 不会更新其能力配置。
宿主机仍需提供 NVIDIA 驱动及 Container Toolkit，不把驱动库打进镜像。
该设置使硬件编解码可用；原工作流的 libx264 编码方式不会因此自动切换为 NVENC。

## 外部文件

模型仍通过 `ComfyUI/models/FlashVSR/` 挂载，Plus Turbo 所用文件为：

- `turbo_dit_plus.safetensors`
- `lq_proj_in_plus_turbo.safetensors`
- `TCDecoder_plus.ckpt`
- `posi_prompt_plus.pth`

现有的一键工作流还需要本地定制节点 `ComfyUI-LocalFlashVSR`，它不是上游
FlashVSR 仓库的一部分。请保留此前部署的该节点、启动恢复脚本和工作流挂载；
此次仅将运行依赖和上游节点纳入镜像构建，不复制模型、测试视频或本地编排源码。

离线构建时，先按项目的离线流程准备节点，并手动确认
`offline-cache/repos/ComfyUI_FlashVSR` 检出上述提交；离线模式会跳过
`preload-nodes.sh`，因此其中的下载和固定提交命令不会执行。

配置检查无需运行超分：

```bash
bash -n builder-scripts/preload-nodes.sh
docker compose config
docker compose -f docker-compose.yml -f docker-compose.3dgs.yml config
```
