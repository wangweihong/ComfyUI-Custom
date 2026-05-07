# nfnode2

节点名：**南风LTX视频转Latent**

这个自定义节点现在只做第一步：

- 从 `VIDEO` 提取帧和 FPS
- 按需要执行 `frame_skip`
- 把帧数处理成 LTX 更稳的 `8n+1`
- 把宽高补到 `32` 的倍数
- 用 `vae.encode()` 输出 `video_latent`

## 输入

- `video: VIDEO`
- `vae: VAE`
- `snap_to_8n_plus_1: BOOLEAN`
- `trim_mode: auto / tail / head / pad_tail / pad_head / strict_error`
- `frame_skip: INT`
- 可选：`frames_in: IMAGE`
- 可选：`fps_in: FLOAT`

## 输出

- `video_latent: LATENT`
- `frames: IMAGE`
- `fps: FLOAT`
- `debug: STRING`

## 附带工作流

仓库里提供了一个最简工作流文件：

- `E:\ComfyUIV8\ComfyUI\custom_nodes\nfnode2\f_ltx_video_to_latent.json`

工作流只包含 4 个节点：

- `LoadVideo`
- `CheckpointLoaderSimple`
- `南风LTX视频转Latent`
- `PreviewImage`

作用是先把视频读进来，生成 `video_latent`，同时把帧预览出来，方便你后面继续往下接。

## 安装

把 `nfnode2` 放到：

```text
ComfyUI/custom_nodes/
```

然后重启 ComfyUI。
