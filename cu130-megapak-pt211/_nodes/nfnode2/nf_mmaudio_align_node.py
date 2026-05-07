import json
import math
from typing import Dict, Tuple

import torch


CATEGORY_NAME = "南风节点/音频"


def _infer_duration(frame_count: int, source_fps: float, fallback_fps: float) -> float:
    source_fps = float(source_fps)
    fallback_fps = float(fallback_fps)
    if source_fps > 0:
        return float(frame_count) / source_fps
    if fallback_fps > 0:
        return float(frame_count) / fallback_fps
    return 0.0


def _temporal_resample_nearest(images: torch.Tensor, target_frames: int) -> torch.Tensor:
    source_frames = int(images.shape[0])
    target_frames = max(1, int(target_frames))

    if source_frames <= 0:
        raise RuntimeError("输入图像序列为空，无法做 MMAudio 对齐。")
    if source_frames == target_frames:
        return images.contiguous()
    if source_frames == 1:
        return images.repeat(target_frames, 1, 1, 1).contiguous()

    positions = torch.linspace(
        0,
        source_frames - 1,
        target_frames,
        device=images.device,
        dtype=torch.float32,
    )
    indices = positions.round().clamp(0, source_frames - 1).long()
    return images.index_select(0, indices).contiguous()


def _temporal_resample_blend(images: torch.Tensor, target_frames: int) -> torch.Tensor:
    source_frames = int(images.shape[0])
    target_frames = max(1, int(target_frames))

    if source_frames <= 0:
        raise RuntimeError("输入图像序列为空，无法做 MMAudio 对齐。")
    if source_frames == target_frames:
        return images.contiguous()
    if source_frames == 1:
        return images.repeat(target_frames, 1, 1, 1).contiguous()

    positions = torch.linspace(
        0,
        source_frames - 1,
        target_frames,
        device=images.device,
        dtype=torch.float32,
    )
    left = positions.floor().clamp(0, source_frames - 1).long()
    right = positions.ceil().clamp(0, source_frames - 1).long()
    weight = (positions - left.to(dtype=positions.dtype)).view(-1, 1, 1, 1)

    left_frames = images.index_select(0, left)
    right_frames = images.index_select(0, right)
    blended = left_frames * (1.0 - weight) + right_frames * weight
    return blended.to(dtype=images.dtype).contiguous()


def _temporal_resample(images: torch.Tensor, target_frames: int, mode: str) -> torch.Tensor:
    mode = str(mode or "nearest").lower()
    if mode == "blend":
        return _temporal_resample_blend(images, target_frames)
    return _temporal_resample_nearest(images, target_frames)


class NanFengMMAudioAlign:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "source_fps": ("FLOAT", {"default": 24.0, "min": 0.001, "max": 240.0, "step": 0.001}),
                "target_duration": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 36000.0, "step": 0.001}),
                "target_fps": ("FLOAT", {"default": 25.0, "min": 0.001, "max": 240.0, "step": 0.001}),
                "resample_mode": (["nearest", "blend"], {"default": "blend"}),
            }
        }

    RETURN_TYPES = ("IMAGE", "FLOAT", "FLOAT", "INT", "STRING")
    RETURN_NAMES = ("image", "duration", "frame_rate", "frame_count", "debug")
    FUNCTION = "align"
    CATEGORY = CATEGORY_NAME

    def align(
        self,
        image,
        source_fps,
        target_duration,
        target_fps,
        resample_mode,
    ):
        debug: Dict[str, object] = {
            "node": "南风MMAudio对齐",
            "ok": False,
        }

        try:
            if not isinstance(image, torch.Tensor):
                raise RuntimeError("image 输入不是有效的 IMAGE Tensor")
            if image.ndim != 4 or image.shape[-1] != 3:
                raise RuntimeError(f"image 输入形状不对，期望 [T,H,W,3]，实际为 {list(image.shape)}")

            source_frames = int(image.shape[0])
            source_fps = float(source_fps)
            target_fps = max(0.001, float(target_fps))

            inferred_duration = _infer_duration(source_frames, source_fps, target_fps)
            resolved_duration = float(target_duration) if float(target_duration) > 0 else inferred_duration
            if resolved_duration <= 0:
                raise RuntimeError("无法推断目标时长，请检查 source_fps 或 target_duration。")

            mmaudio_required_frames = max(1, int(resolved_duration * target_fps))
            output_frames = max(1, int(math.ceil(resolved_duration * target_fps)))
            aligned = _temporal_resample(image, output_frames, resample_mode)

            debug.update(
                {
                    "ok": True,
                    "source_frame_count": int(source_frames),
                    "source_fps": float(source_fps),
                    "source_duration": float(inferred_duration),
                    "resolved_duration": float(resolved_duration),
                    "target_fps": float(target_fps),
                    "mmaudio_required_frames": int(mmaudio_required_frames),
                    "output_frame_count": int(output_frames),
                    "resample_mode": str(resample_mode),
                    "input_shape": [int(v) for v in image.shape],
                    "output_shape": [int(v) for v in aligned.shape],
                }
            )

            return (
                aligned,
                float(resolved_duration),
                float(target_fps),
                int(output_frames),
                json.dumps(debug, ensure_ascii=False, indent=2),
            )
        except Exception as exc:
            debug.update({"error": str(exc)})
            raise RuntimeError(json.dumps(debug, ensure_ascii=False, indent=2))


NODE_CLASS_MAPPINGS = {
    "NanFengMMAudioAlign": NanFengMMAudioAlign,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "NanFengMMAudioAlign": "南风MMAudio对齐",
}
