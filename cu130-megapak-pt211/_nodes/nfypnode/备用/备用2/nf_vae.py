import math
import time

import torch


class NanFengVAE:
    CATEGORY = "南风阳平/VAE"
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("图像",)
    FUNCTION = "decode"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "vae": ("VAE",),
                "latent": ("LATENT",),
                "横向分块数": ("INT", {"default": 1, "min": 1, "max": 6, "step": 1}),
                "纵向分块数": ("INT", {"default": 1, "min": 1, "max": 6, "step": 1}),
                "重叠宽度": ("INT", {"default": 1, "min": 1, "max": 8, "step": 1}),
                "最后一帧修复": ("BOOLEAN", {"default": False}),
            },
            "optional": {
                "工作设备": (["cpu", "auto"], {"default": "auto"}),
                "工作精度": (["float16", "float32", "auto"], {"default": "auto"}),
            },
        }

    def _print(self, msg: str):
        print(f"[南风VAE] {msg}")

    def _progress_bar(self, current: int, total: int, prefix: str = "解码进度"):
        total = max(1, int(total))
        current = max(0, min(int(current), total))
        width = 28
        ratio = current / total
        filled = int(round(width * ratio))
        bar = "█" * filled + "-" * (width - filled)
        self._print(f"{prefix} [{bar}] {current}/{total} ({ratio * 100:.1f}%)")

    def _resolve_dtype(self, samples, working_dtype: str):
        if working_dtype == "auto":
            return samples.dtype
        if working_dtype == "float16":
            return torch.float16
        if working_dtype == "float32":
            return torch.float32
        raise ValueError(f"不支持的工作精度: {working_dtype}")

    def decode(
        self,
        vae,
        latent,
        横向分块数,
        纵向分块数,
        重叠宽度,
        最后一帧修复,
        工作设备="auto",
        工作精度="auto",
    ):
        samples = latent["samples"]
        if not torch.is_tensor(samples):
            raise RuntimeError("输入的 latent['samples'] 不是 tensor，无法解码。")
        if samples.ndim != 5:
            raise RuntimeError(f"南风VAE 需要视频 latent（5维），当前 shape={tuple(samples.shape)}")

        if 最后一帧修复:
            last_frame = samples[:, :, -1:, :, :]
            samples = torch.cat([samples, last_frame], dim=2)
            self._print("已启用最后一帧修复：先复制 1 帧到 latent 尾部，解码完成后再裁掉。")

        batch, channels, frames, height, width = samples.shape
        time_scale_factor, width_scale_factor, height_scale_factor = vae.downscale_index_formula
        image_frames = 1 + (frames - 1) * time_scale_factor
        output_height = height * height_scale_factor
        output_width = width * width_scale_factor

        target_device = samples.device if 工作设备 == "auto" else 工作设备
        target_dtype = self._resolve_dtype(samples, 工作精度)

        base_tile_height = (height + (纵向分块数 - 1) * 重叠宽度) // 纵向分块数
        base_tile_width = (width + (横向分块数 - 1) * 重叠宽度) // 横向分块数

        output = torch.zeros(
            (batch, image_frames, output_height, output_width, 3),
            device=target_device,
            dtype=target_dtype,
        )
        weights = torch.zeros(
            (batch, image_frames, output_height, output_width, 1),
            device=target_device,
            dtype=target_dtype,
        )

        total_tiles = int(横向分块数) * int(纵向分块数)
        done_tiles = 0
        start_time = time.time()

        self._print(
            f"开始分块 VAE 解码 | latent_shape={tuple(samples.shape)} | 输出={image_frames}帧 {output_width}x{output_height} | "
            f"横向分块数={横向分块数} | 纵向分块数={纵向分块数} | 重叠宽度={重叠宽度} | 工作设备={target_device} | 工作精度={target_dtype}"
        )
        self._progress_bar(0, total_tiles)

        for v in range(纵向分块数):
            for h in range(横向分块数):
                h_start = h * (base_tile_width - 重叠宽度)
                v_start = v * (base_tile_height - 重叠宽度)

                h_end = min(h_start + base_tile_width, width) if h < 横向分块数 - 1 else width
                v_end = min(v_start + base_tile_height, height) if v < 纵向分块数 - 1 else height

                tile_height = v_end - v_start
                tile_width = h_end - h_start

                self._print(
                    f"开始处理分块 row={v + 1}/{纵向分块数}, col={h + 1}/{横向分块数} | "
                    f"latent区域=({v_start}:{v_end}, {h_start}:{h_end}) | 大小={tile_height}x{tile_width}"
                )

                tile = samples[:, :, :, v_start:v_end, h_start:h_end]
                decoded_tile = vae.decode(tile)

                out_h_start = v_start * height_scale_factor
                out_h_end = v_end * height_scale_factor
                out_w_start = h_start * width_scale_factor
                out_w_end = h_end * width_scale_factor

                tile_out_height = out_h_end - out_h_start
                tile_out_width = out_w_end - out_w_start
                tile_weights = torch.ones(
                    (batch, image_frames, tile_out_height, tile_out_width, 1),
                    device=decoded_tile.device,
                    dtype=decoded_tile.dtype,
                )

                overlap_out_h = 重叠宽度 * height_scale_factor
                overlap_out_w = 重叠宽度 * width_scale_factor

                if h > 0 and overlap_out_w > 0:
                    h_blend = torch.linspace(0, 1, overlap_out_w, device=decoded_tile.device, dtype=decoded_tile.dtype)
                    tile_weights[:, :, :, :overlap_out_w, :] *= h_blend.view(1, 1, 1, -1, 1)
                if h < 横向分块数 - 1 and overlap_out_w > 0:
                    h_blend = torch.linspace(1, 0, overlap_out_w, device=decoded_tile.device, dtype=decoded_tile.dtype)
                    tile_weights[:, :, :, -overlap_out_w:, :] *= h_blend.view(1, 1, 1, -1, 1)

                if v > 0 and overlap_out_h > 0:
                    v_blend = torch.linspace(0, 1, overlap_out_h, device=decoded_tile.device, dtype=decoded_tile.dtype)
                    tile_weights[:, :, :overlap_out_h, :, :] *= v_blend.view(1, 1, -1, 1, 1)
                if v < 纵向分块数 - 1 and overlap_out_h > 0:
                    v_blend = torch.linspace(1, 0, overlap_out_h, device=decoded_tile.device, dtype=decoded_tile.dtype)
                    tile_weights[:, :, -overlap_out_h:, :, :] *= v_blend.view(1, 1, -1, 1, 1)

                output[:, :, out_h_start:out_h_end, out_w_start:out_w_end, :] += (
                    decoded_tile * tile_weights
                ).to(target_device, target_dtype)
                weights[:, :, out_h_start:out_h_end, out_w_start:out_w_end, :] += tile_weights.to(target_device, target_dtype)

                done_tiles += 1
                elapsed = max(1e-6, time.time() - start_time)
                speed = done_tiles / elapsed
                eta = (total_tiles - done_tiles) / max(speed, 1e-6)
                self._progress_bar(done_tiles, total_tiles)
                self._print(
                    f"已完成分块 {done_tiles}/{total_tiles} | 当前块输出区域=({out_h_start}:{out_h_end}, {out_w_start}:{out_w_end}) | "
                    f"累计耗时={elapsed:.1f}s | 预计剩余={eta:.1f}s"
                )

        output /= weights + 1e-8
        output = output.view(batch * image_frames, output_height, output_width, output.shape[-1])

        if 最后一帧修复:
            output = output[:-time_scale_factor, :, :]
            self._print(f"最后一帧修复已回裁输出尾帧，共裁掉 {time_scale_factor} 帧。")

        total_elapsed = time.time() - start_time
        self._print(f"VAE 解码完成 | 最终输出 shape={tuple(output.shape)} | 总耗时={total_elapsed:.2f}s")
        return (output,)


NODE_CLASS_MAPPINGS = {
    "NanFengVAE": NanFengVAE,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "NanFengVAE": "南风VAE",
}
