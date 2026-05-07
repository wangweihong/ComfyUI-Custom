import json
import traceback
from typing import Any, Dict, Iterable, Tuple

import comfy.samplers
import comfy.utils
import folder_paths
import nodes
import torch
from nodes import CLIPTextEncode

from comfy_extras.nodes_custom_sampler import (
    CFGGuider,
    KSamplerSelect,
    ManualSigmas,
    RandomNoise,
    SamplerCustomAdvanced,
)
from comfy_extras.nodes_lt import LTXVConditioning, LTXVConcatAVLatent, LTXVSeparateAVLatent
from comfy_extras.nodes_lt_audio import (
    LTXAVTextEncoderLoader,
    LTXVAudioVAEDecode,
    LTXVAudioVAELoader,
    LTXVEmptyLatentAudio,
)

try:
    from nodes import NODE_CLASS_MAPPINGS as CORE_NODE_CLASS_MAPPINGS
except Exception:
    CORE_NODE_CLASS_MAPPINGS = {}

try:
    import nodes_video
except Exception:
    nodes_video = None

DEFAULT_CHECKPOINT = "ltx-2.3-22b-dev-fp8.safetensors"
DEFAULT_TEXT_ENCODER = "gemma312BAbliterated_v10aExperimental.safetensors"
DEFAULT_POSITIVE_PROMPT = (
    "Realistic environmental ambience and physical action sounds only. "
    "Natural impact sounds, body movement, cloth rustling, object interaction, "
    "collision sounds, footsteps, and room ambience. Non-vocal, non-musical scene audio only."
)
DEFAULT_NEGATIVE_PROMPT = (
    "speech, talking, dialogue, human voice, vocalization, singing, whispering, murmuring, "
    "mumbling, narration, voiceover, humming, chanting, laughing, giggling, crying, shouting, "
    "screaming, coughing, sneezing, moaning, mouth sounds, lip smacks, tongue clicks, saliva sounds, "
    "chewing sounds, swallowing sounds, breathy vocal sounds, music, background music, soundtrack, "
    "score, melody, instrumental music, ambient music, cinematic music, dramatic music, emotional music, "
    "piano, guitar, violin, orchestra, synth, bass, drum beat, rhythm, song, musical layer, background track, "
    "underscoring, audio bed, jingle"
)
DEFAULT_SIGMAS = "1.0, 0.985, 0.96, 0.92, 0.84, 0.72, 0.58, 0.42, 0.28, 0.16, 0.08, 0.0"


CATEGORY_NAME = "南风节点/配音"


def _find_core_node(*candidates: str):
    for candidate in candidates:
        if candidate in CORE_NODE_CLASS_MAPPINGS:
            return CORE_NODE_CLASS_MAPPINGS[candidate]

    for cls in CORE_NODE_CLASS_MAPPINGS.values():
        cls_name = getattr(cls, "__name__", "")
        display_name = getattr(cls, "NODE_NAME", "")
        if cls_name in candidates or display_name in candidates:
            return cls
    return None


def _instantiate_get_video_components():
    cls = _find_core_node("GetVideoComponents", "Get Video Components")
    if cls is not None:
        return cls()

    if nodes_video is not None:
        for attr in ("GetVideoComponents", "GetVideoComponent", "Get_Video_Components"):
            if hasattr(nodes_video, attr):
                return getattr(nodes_video, attr)()
    return None


def _call_node_instance(node: Any, **kwargs):
    fn_name = getattr(node, "FUNCTION", None)
    if fn_name and hasattr(node, fn_name):
        return getattr(node, fn_name)(**kwargs)

    for name in ("execute", "run", "generate", "load", "get_components", "extract"):
        if hasattr(node, name) and callable(getattr(node, name)):
            return getattr(node, name)(**kwargs)

    raise RuntimeError(f"Node {type(node).__name__} does not expose a callable entrypoint.")


def _normalize_result(result: Any) -> Tuple[Any, ...]:
    if hasattr(result, "result"):
        result = result.result
    if result is None:
        return tuple()
    if isinstance(result, tuple):
        return result
    if isinstance(result, list):
        return tuple(result)
    return (result,)


def _to_float(value: Any, default: float = 0.0) -> float:
    try:
        if isinstance(value, torch.Tensor):
            return float(value.detach().cpu().item())
        return float(value)
    except Exception:
        return float(default)


def _coerce_frame_fit_mode(value: Any) -> str:
    allowed = {"auto", "tail", "head", "pad_tail", "pad_head", "strict_error"}

    if isinstance(value, bool):
        return "tail" if value else "head"

    text = str(value).strip().lower()
    alias = {
        "trim_tail": "tail",
        "trim_head": "head",
        "true": "tail",
        "false": "head",
        "1": "tail",
        "0": "head",
        "yes": "tail",
        "no": "head",
        "y": "tail",
        "n": "head",
    }
    text = alias.get(text, text)
    if text in allowed:
        return text
    return "auto"


def _shape_of(value: Any):
    if hasattr(value, "shape"):
        try:
            return list(value.shape)
        except Exception:
            return str(value.shape)
    return None


def _ensure_image_tensor(images: Any) -> torch.Tensor:
    if isinstance(images, torch.Tensor):
        out = images
    else:
        out = torch.as_tensor(images)

    if out.ndim == 3:
        out = out.unsqueeze(0)

    if out.ndim != 4:
        raise ValueError(
            f"Image/video frames must be [T, H, W, C], got {tuple(out.shape)} instead."
        )

    if out.shape[-1] < 3:
        raise ValueError(
            f"Image/video frames need at least 3 channels, got shape {tuple(out.shape)}."
        )

    out = out[..., :3].contiguous()
    if out.dtype != torch.float32:
        out = out.float()

    max_value = float(out.max().item()) if out.numel() > 0 else 1.0
    if max_value > 1.0:
        out = out / 255.0

    return out.clamp(0.0, 1.0)


def _extract_video_components(
    video: Any,
    frames_in: Any = None,
    fps_in: float = 0.0,
) -> Tuple[torch.Tensor, float, Dict[str, Any]]:
    meta: Dict[str, Any] = {
        "video_python_type": str(type(video)),
        "used_explicit_frames_inputs": False,
        "extract_path": None,
    }

    if frames_in is not None:
        fps = _to_float(fps_in, 25.0)
        if fps <= 0:
            fps = 25.0
        meta["used_explicit_frames_inputs"] = True
        meta["extract_path"] = "explicit_frames_fps"
        meta["frames_in_shape"] = _shape_of(frames_in)
        meta["fps_in"] = float(fps)
        return _ensure_image_tensor(frames_in), fps, meta

    node = _instantiate_get_video_components()
    if node is not None:
        try:
            result = _call_node_instance(node, video=video)
            result = _normalize_result(result)
            if len(result) >= 3:
                images, _, fps = result[:3]
            elif len(result) >= 1:
                images = result[0]
                fps = 25.0
            else:
                images = None
                fps = 25.0

            if images is not None:
                meta["extract_path"] = f"node:{type(node).__name__}"
                meta["frames_in_shape"] = _shape_of(images)
                meta["fps_in"] = float(_to_float(fps, 25.0))
                return _ensure_image_tensor(images), _to_float(fps, 25.0), meta
        except Exception as exc:
            meta["get_video_components_error"] = str(exc)

    if isinstance(video, (tuple, list)) and len(video) >= 1:
        try:
            images = video[0]
            fps = video[2] if len(video) >= 3 else 25.0
            meta["extract_path"] = "tuple_or_list"
            meta["frames_in_shape"] = _shape_of(images)
            meta["fps_in"] = float(_to_float(fps, 25.0))
            return _ensure_image_tensor(images), _to_float(fps, 25.0), meta
        except Exception as exc:
            meta["tuple_extract_error"] = str(exc)

    if isinstance(video, dict):
        image_keys = ("images", "frames", "image", "imgs")
        fps_keys = ("fps", "frame_rate", "framerate")

        images = None
        fps = 25.0

        for key in image_keys:
            if key in video and video[key] is not None:
                images = video[key]
                break
        for key in fps_keys:
            if key in video and video[key] is not None:
                fps = video[key]
                break

        if images is not None:
            meta["extract_path"] = "dict_fields"
            meta["video_dict_keys"] = list(video.keys())[:50]
            meta["frames_in_shape"] = _shape_of(images)
            meta["fps_in"] = float(_to_float(fps, 25.0))
            return _ensure_image_tensor(images), _to_float(fps, 25.0), meta

    for image_attr in ("images", "frames", "image", "imgs"):
        if hasattr(video, image_attr):
            images = getattr(video, image_attr)
            fps = 25.0
            for fps_attr in ("fps", "frame_rate", "framerate"):
                if hasattr(video, fps_attr):
                    fps = getattr(video, fps_attr)
                    break
            meta["extract_path"] = f"attr:{image_attr}"
            meta["frames_in_shape"] = _shape_of(images)
            meta["fps_in"] = float(_to_float(fps, 25.0))
            return _ensure_image_tensor(images), _to_float(fps, 25.0), meta

    raise RuntimeError(
        "无法从 VIDEO 输入中提取 frames/fps。请额外连接 GetVideoComponents 的 images/fps 到 frames_in/fps_in。"
    )


def _extract_fps_from_video_info(video_info: Any, default: float = 0.0) -> float:
    if not isinstance(video_info, dict):
        return float(default)

    for key in ("loaded_fps", "source_fps", "fps", "frame_rate", "framerate"):
        if key in video_info and video_info[key] is not None:
            fps = _to_float(video_info[key], default)
            if fps > 0:
                return float(fps)

    return float(default)


def _nearest_8n1_lengths(src_len: int) -> Tuple[int, int]:
    if src_len <= 1:
        return 1, 1
    prev_len = ((src_len - 1) // 8) * 8 + 1
    next_len = prev_len if prev_len == src_len else prev_len + 8
    return max(1, prev_len), max(1, next_len)


def _trim_frames(images: torch.Tensor, dst_len: int, from_head: bool) -> torch.Tensor:
    src_len = int(images.shape[0])
    if dst_len >= src_len:
        return images
    if from_head:
        return images[src_len - dst_len :].contiguous()
    return images[:dst_len].contiguous()


def _pad_frames(images: torch.Tensor, dst_len: int, at_head: bool) -> torch.Tensor:
    src_len = int(images.shape[0])
    if dst_len <= src_len:
        return images
    pad_count = dst_len - src_len
    if src_len <= 0:
        raise RuntimeError("Cannot pad an empty frame sequence.")

    ref = images[:1] if at_head else images[-1:]
    pad = ref.repeat(pad_count, 1, 1, 1)
    if at_head:
        return torch.cat([pad, images], dim=0).contiguous()
    return torch.cat([images, pad], dim=0).contiguous()


def _fit_to_8n_plus_1(
    images: torch.Tensor,
    mode: str = "auto",
) -> Tuple[torch.Tensor, Dict[str, Any]]:
    src_len = int(images.shape[0])
    mode = _coerce_frame_fit_mode(mode)

    info: Dict[str, Any] = {
        "frame_fit_mode": mode,
        "frame_fit_raw_count": int(src_len),
        "frame_fit_is_already_valid": False,
        "frame_fit_action": "none",
        "frame_fit_from": int(src_len),
        "frame_fit_to": int(src_len),
        "frame_fit_prev_valid": None,
        "frame_fit_next_valid": None,
        "frame_fit_trim_count": 0,
        "frame_fit_pad_count": 0,
    }

    if src_len <= 0:
        raise RuntimeError("输入视频没有帧，无法执行 8n+1 处理。")

    prev_len, next_len = _nearest_8n1_lengths(src_len)
    info["frame_fit_prev_valid"] = int(prev_len)
    info["frame_fit_next_valid"] = int(next_len)

    if src_len == prev_len:
        info["frame_fit_is_already_valid"] = True
        return images, info

    if mode == "strict_error":
        raise RuntimeError(
            f"当前帧数 {src_len} 不是合法的 8n+1 长度，最近的合法长度是 {prev_len} 或 {next_len}。"
        )

    trim_cost = src_len - prev_len
    pad_cost = next_len - src_len

    if mode == "auto":
        mode = "pad_tail" if pad_cost <= trim_cost else "tail"

    info["frame_fit_resolved_mode"] = mode

    if mode == "tail":
        out = _trim_frames(images, prev_len, from_head=False)
        info["frame_fit_action"] = "trim_tail"
        info["frame_fit_to"] = int(prev_len)
        info["frame_fit_trim_count"] = int(trim_cost)
        return out, info

    if mode == "head":
        out = _trim_frames(images, prev_len, from_head=True)
        info["frame_fit_action"] = "trim_head"
        info["frame_fit_to"] = int(prev_len)
        info["frame_fit_trim_count"] = int(trim_cost)
        return out, info

    if mode == "pad_tail":
        out = _pad_frames(images, next_len, at_head=False)
        info["frame_fit_action"] = "pad_tail_last_frame"
        info["frame_fit_to"] = int(next_len)
        info["frame_fit_pad_count"] = int(pad_cost)
        return out, info

    if mode == "pad_head":
        out = _pad_frames(images, next_len, at_head=True)
        info["frame_fit_action"] = "pad_head_first_frame"
        info["frame_fit_to"] = int(next_len)
        info["frame_fit_pad_count"] = int(pad_cost)
        return out, info

    raise RuntimeError(f"Unsupported frame_fit_mode: {mode}")


def _apply_frame_skip(images: torch.Tensor, fps: float, frame_skip: int) -> Tuple[torch.Tensor, float]:
    frame_skip = max(1, int(frame_skip))
    if frame_skip == 1:
        return images, fps
    new_images = images[::frame_skip].contiguous()
    new_fps = fps / frame_skip if fps > 0 else fps
    return new_images, new_fps


def _next_multiple(value: int, multiple: int) -> int:
    if multiple <= 1:
        return int(value)
    return ((int(value) + multiple - 1) // multiple) * multiple


def _pad_spatial_to_multiple(
    images: torch.Tensor,
    multiple: int = 32,
) -> Tuple[torch.Tensor, Dict[str, Any]]:
    height = int(images.shape[1])
    width = int(images.shape[2])
    dst_height = _next_multiple(height, multiple)
    dst_width = _next_multiple(width, multiple)
    pad_height = max(0, dst_height - height)
    pad_width = max(0, dst_width - width)

    info: Dict[str, Any] = {
        "spatial_fit_multiple": int(multiple),
        "spatial_fit_input_hw": [int(height), int(width)],
        "spatial_fit_output_hw": [int(dst_height), int(dst_width)],
        "spatial_fit_pad_bottom": int(pad_height),
        "spatial_fit_pad_right": int(pad_width),
        "spatial_fit_action": "none",
    }

    if pad_height == 0 and pad_width == 0:
        info["spatial_fit_is_already_valid"] = True
        return images, info

    out = images
    if pad_width > 0:
        right = out[:, :, -1:, :].repeat(1, 1, pad_width, 1)
        out = torch.cat([out, right], dim=2)
    if pad_height > 0:
        bottom = out[:, -1:, :, :].repeat(1, pad_height, 1, 1)
        out = torch.cat([out, bottom], dim=1)

    info["spatial_fit_is_already_valid"] = False
    info["spatial_fit_action"] = "pad_right_bottom_edge_replicate"
    return out.contiguous(), info


def _extract_samples(encoded: Any) -> torch.Tensor:
    if isinstance(encoded, dict):
        encoded = encoded.get("samples", encoded)
    elif isinstance(encoded, (tuple, list)):
        encoded = encoded[0]

    if not isinstance(encoded, torch.Tensor):
        encoded = torch.as_tensor(encoded)

    if encoded.ndim == 4:
        if encoded.shape[0] <= 32 and encoded.shape[1] >= 64:
            encoded = encoded.permute(1, 0, 2, 3).unsqueeze(0)
        else:
            encoded = encoded.unsqueeze(0)
    elif encoded.ndim != 5:
        raise RuntimeError(
            f"VAE encode returned an unsupported latent rank: {tuple(encoded.shape)}"
        )

    return encoded.contiguous()


def _encode_video_to_latent(vae: Any, images: torch.Tensor) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    pixels = images[..., :3].contiguous()
    meta = {
        "video_encode_layout": "THWC",
        "video_encode_pixels_shape": list(pixels.shape),
        "video_encode_pixels_dtype": str(pixels.dtype),
        "video_encode_pixels_min": float(pixels.min().item()) if pixels.numel() > 0 else 0.0,
        "video_encode_pixels_max": float(pixels.max().item()) if pixels.numel() > 0 else 0.0,
    }

    try:
        encoded = vae.encode(pixels)
        samples = _extract_samples(encoded)
        meta["video_encode_ok"] = True
        meta["video_latent_shape"] = list(samples.shape)
        return {"samples": samples}, meta
    except Exception as exc:
        meta["video_encode_ok"] = False
        meta["video_encode_error"] = str(exc)
        raise RuntimeError(
            json.dumps(
                {
                    "error": f"视频 VAE 编码失败: {exc}",
                    **meta,
                },
                ensure_ascii=False,
                indent=2,
            )
        )


def _to_int(value: Any, default: int = 0) -> int:
    try:
        if isinstance(value, torch.Tensor):
            return int(value.detach().cpu().item())
        return int(round(float(value)))
    except Exception:
        return int(default)


def _execute_node(node_or_cls: Any, **kwargs) -> Tuple[Any, ...]:
    if isinstance(node_or_cls, type) and hasattr(node_or_cls, "execute"):
        return _normalize_result(node_or_cls.execute(**kwargs))
    return _normalize_result(_call_node_instance(node_or_cls, **kwargs))


def _pick_default_option(options: Iterable[str], preferred: str = "") -> str:
    options = list(options or [])
    if preferred and preferred in options:
        return preferred
    return options[0] if options else preferred


def _combo_options(folder_name: str, preferred: str) -> Tuple[list[str], str]:
    options = list(folder_paths.get_filename_list(folder_name))
    default_value = _pick_default_option(options, preferred)
    return options, default_value


def _resize_frames(
    images: torch.Tensor,
    scale: float,
) -> Tuple[torch.Tensor, Dict[str, Any]]:
    src_height = int(images.shape[1])
    src_width = int(images.shape[2])
    scale = max(0.05, float(scale))
    dst_width = max(1, int(round(src_width * scale)))
    dst_height = max(1, int(round(src_height * scale)))

    info = {
        "resize_input_hw": [src_height, src_width],
        "resize_output_hw": [dst_height, dst_width],
        "resize_applied": False,
        "input_scale": float(scale),
    }

    if dst_width == src_width and dst_height == src_height:
        return images, info

    resized = comfy.utils.common_upscale(
        images.movedim(-1, 1),
        dst_width,
        dst_height,
        "bilinear",
        "center",
    ).movedim(1, -1)
    info["resize_applied"] = True
    return resized.contiguous(), info


def _repeat_single_frame(images: torch.Tensor, target_frames: int) -> torch.Tensor:
    target_frames = max(1, int(target_frames))
    if int(images.shape[0]) == target_frames:
        return images
    if int(images.shape[0]) != 1:
        return images
    return images.repeat(target_frames, 1, 1, 1).contiguous()


def _derive_video_meta_from_latent(video_latent: Dict[str, Any], fallback_fps: float) -> Tuple[int, float]:
    samples = video_latent.get("samples")
    if samples is None:
        raise RuntimeError("video_latent is missing samples.")
    if getattr(samples, "is_nested", False):
        raise RuntimeError("video_latent must be a pure video latent, not a combined AV latent.")

    frame_count = _to_int(video_latent.get("frame_count"), 0)
    if frame_count <= 0:
        latent_frames = int(samples.shape[2])
        frame_count = max(1, (latent_frames - 1) * 8 + 1)

    fps = _to_float(video_latent.get("fps"), 0.0)
    if fps <= 0:
        fps = _to_float(video_latent.get("frame_rate"), 0.0)
    if fps <= 0:
        fps = float(fallback_fps)
    if fps <= 0:
        fps = 25.0

    return int(frame_count), float(fps)


def _resize_video_latent(video_latent: Dict[str, Any], scale: float) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    samples = video_latent.get("samples")
    if samples is None:
        raise RuntimeError("video_latent is missing samples.")
    if getattr(samples, "is_nested", False):
        raise RuntimeError("video_latent must be a pure video latent, not a combined AV latent.")

    scale = max(0.05, float(scale))
    src_h = int(samples.shape[3])
    src_w = int(samples.shape[4])
    dst_h = max(1, int(round(src_h * scale)))
    dst_w = max(1, int(round(src_w * scale)))

    info = {
        "latent_resize_input_hw": [src_h, src_w],
        "latent_resize_output_hw": [dst_h, dst_w],
        "latent_scale": float(scale),
        "latent_resize_applied": False,
    }

    if dst_h == src_h and dst_w == src_w:
        return dict(video_latent), info

    batch, channels, frames, _, _ = samples.shape
    flat = samples.permute(0, 2, 1, 3, 4).reshape(batch * frames, channels, src_h, src_w)
    resized = torch.nn.functional.interpolate(
        flat,
        size=(dst_h, dst_w),
        mode="bilinear",
        align_corners=False,
    )
    resized = resized.reshape(batch, frames, channels, dst_h, dst_w).permute(0, 2, 1, 3, 4).contiguous()

    out = dict(video_latent)
    out["samples"] = resized

    noise_mask = out.get("noise_mask")
    if isinstance(noise_mask, torch.Tensor) and noise_mask.ndim == 5 and (
        int(noise_mask.shape[3]) > 1 or int(noise_mask.shape[4]) > 1
    ):
        mask_batch, mask_channels, mask_frames, mask_h, mask_w = noise_mask.shape
        mask_flat = noise_mask.permute(0, 2, 1, 3, 4).reshape(mask_batch * mask_frames, mask_channels, mask_h, mask_w)
        resized_mask = torch.nn.functional.interpolate(mask_flat, size=(dst_h, dst_w), mode="nearest")
        out["noise_mask"] = (
            resized_mask.reshape(mask_batch, mask_frames, mask_channels, dst_h, dst_w)
            .permute(0, 2, 1, 3, 4)
            .contiguous()
        )

    out["encoded_width"] = int(dst_w * 32)
    out["encoded_height"] = int(dst_h * 32)
    info["latent_resize_applied"] = True
    return out, info


def _prepare_audio_only_masks(
    video_latent: Dict[str, Any],
    audio_latent: Dict[str, Any],
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    out_video = dict(video_latent)
    out_audio = dict(audio_latent)

    video_samples = out_video["samples"]
    audio_samples = out_audio["samples"]

    out_video["noise_mask"] = torch.zeros_like(video_samples)[:, :1]
    out_audio["noise_mask"] = torch.ones_like(audio_samples)
    return out_video, out_audio


def _prepare_source_video_latent(
    vae,
    video=None,
    image=None,
    video_latent=None,
    frame_rate: float = 25.0,
    frames_number: int = 97,
    input_scale: float = 1.0,
    latent_scale: float = 1.0,
    snap_to_8n_plus_1: bool = True,
    trim_mode: str = "auto",
    frame_skip: int = 1,
) -> Tuple[Dict[str, Any], int, float, Dict[str, Any]]:
    debug: Dict[str, Any] = {
        "source_video_connected": video is not None,
        "source_image_connected": image is not None,
        "source_video_latent_connected": video_latent is not None,
    }

    connected_sources = sum(1 for item in (video, image, video_latent) if item is not None)
    if connected_sources == 0:
        raise RuntimeError("Please connect one of: video, image, or video_latent.")
    if connected_sources > 1:
        raise RuntimeError("Only one source input can be connected at a time: video, image, or video_latent.")

    if video_latent is not None:
        resized_video_latent, latent_resize_info = _resize_video_latent(video_latent, latent_scale)
        frame_count, resolved_fps = _derive_video_meta_from_latent(resized_video_latent, frame_rate)
        debug.update(
            {
                "source_mode": "video_latent",
                "source_frame_count": int(frame_count),
                "source_fps": float(resolved_fps),
                "latent_shape": _shape_of(resized_video_latent.get("samples")),
            }
        )
        debug.update(latent_resize_info)
        return dict(resized_video_latent), int(frame_count), float(resolved_fps), debug

    if video is not None:
        frames, resolved_fps, extract_meta = _extract_video_components(video=video, fps_in=frame_rate)
        debug.update(extract_meta)
        debug["source_mode"] = "video"
    else:
        frames = _ensure_image_tensor(image)
        resolved_fps = float(frame_rate if frame_rate > 0 else 25.0)
        if int(frames.shape[0]) == 1:
            frames = _repeat_single_frame(frames, frames_number)
            debug["image_repeated_to_frames"] = int(frames.shape[0])
        debug["source_mode"] = "image"

    original_count = int(frames.shape[0])
    original_height = int(frames.shape[1])
    original_width = int(frames.shape[2])

    frames, resize_info = _resize_frames(frames, input_scale)
    debug.update(resize_info)

    frames, resolved_fps = _apply_frame_skip(frames, resolved_fps, frame_skip)
    debug["frame_skip"] = int(frame_skip)
    debug["after_skip_frame_count"] = int(frames.shape[0])
    debug["after_skip_fps"] = float(resolved_fps)

    if snap_to_8n_plus_1:
        frames, fit_info = _fit_to_8n_plus_1(frames, trim_mode)
        debug.update(fit_info)
    else:
        debug["frame_fit_action"] = "disabled"

    frames, spatial_info = _pad_spatial_to_multiple(frames, multiple=32)
    debug.update(spatial_info)

    prepared_count = int(frames.shape[0])
    encoded_latent, encode_meta = _encode_video_to_latent(vae, frames)
    debug.update(encode_meta)

    encoded_latent = dict(encoded_latent)
    encoded_latent["type"] = "nf_video_latent"
    encoded_latent["fps"] = float(resolved_fps)
    encoded_latent["frame_count"] = int(prepared_count)
    encoded_latent["source_width"] = int(original_width)
    encoded_latent["source_height"] = int(original_height)
    encoded_latent["encoded_width"] = int(frames.shape[2])
    encoded_latent["encoded_height"] = int(frames.shape[1])

    debug.update(
        {
            "original_frame_count": int(original_count),
            "prepared_frame_count": int(prepared_count),
            "source_size": [int(original_width), int(original_height)],
            "prepared_size": [int(frames.shape[2]), int(frames.shape[1])],
        }
    )
    return encoded_latent, int(prepared_count), float(resolved_fps), debug


class NanFengLTXVideoLatent:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "vae": ("VAE",),
                "snap_to_8n_plus_1": ("BOOLEAN", {"default": True}),
                "trim_mode": (
                    ["auto", "tail", "head", "pad_tail", "pad_head", "strict_error"],
                    {"default": "auto"},
                ),
                "frame_skip": ("INT", {"default": 1, "min": 1, "max": 16, "step": 1}),
            },
            "optional": {
                "video": ("VIDEO",),
                "frames_in": ("IMAGE",),
                "fps_in": ("FLOAT", {"default": 0.0, "min": 0.0, "step": 0.001}),
                "video_info_in": ("VHS_VIDEOINFO",),
            },
        }

    RETURN_TYPES = ("LATENT", "IMAGE", "FLOAT", "STRING")
    RETURN_NAMES = ("video_latent", "frames", "fps", "debug")
    FUNCTION = "build"
    CATEGORY = CATEGORY_NAME

    def build(
        self,
        vae,
        snap_to_8n_plus_1,
        trim_mode,
        frame_skip,
        video=None,
        frames_in=None,
        fps_in=0.0,
        video_info_in=None,
    ):
        debug: Dict[str, Any] = {
            "node": "南风LTX视频转Latent",
            "ok": False,
        }

        try:
            trim_mode = _coerce_frame_fit_mode(trim_mode)
            if frames_in is None and video is None:
                raise RuntimeError("请至少提供 video 或 frames_in。")

            resolved_fps_in = float(fps_in)
            if resolved_fps_in <= 0:
                resolved_fps_in = _extract_fps_from_video_info(video_info_in, 0.0)

            debug["video_input_connected"] = video is not None
            debug["frames_in_connected"] = frames_in is not None
            debug["video_info_in_connected"] = video_info_in is not None
            debug["fps_in_resolved"] = float(resolved_fps_in)

            frames, fps, extract_meta = _extract_video_components(
                video=video,
                frames_in=frames_in,
                fps_in=resolved_fps_in,
            )
            debug.update(extract_meta)

            original_frame_count = int(frames.shape[0])
            original_height = int(frames.shape[1])
            original_width = int(frames.shape[2])
            original_fps = float(fps if fps > 0 else 25.0)

            frames, fps = _apply_frame_skip(frames, original_fps, frame_skip)
            after_skip_count = int(frames.shape[0])

            fit_info: Dict[str, Any] = {
                "frame_fit_mode": trim_mode,
                "frame_fit_action": "disabled",
                "frame_fit_from": after_skip_count,
                "frame_fit_to": after_skip_count,
            }
            if snap_to_8n_plus_1:
                frames, fit_info = _fit_to_8n_plus_1(frames, trim_mode)

            frames, spatial_info = _pad_spatial_to_multiple(frames, multiple=32)

            frame_count = int(frames.shape[0])
            video_latent, video_encode_meta = _encode_video_to_latent(vae, frames)
            debug.update(video_encode_meta)

            video_latent = dict(video_latent)
            video_latent["type"] = "nf_video_latent"
            video_latent["fps"] = float(fps)
            video_latent["frame_count"] = int(frame_count)
            video_latent["source_width"] = int(original_width)
            video_latent["source_height"] = int(original_height)
            video_latent["encoded_width"] = int(frames.shape[2])
            video_latent["encoded_height"] = int(frames.shape[1])

            debug.update(
                {
                    "ok": True,
                    "input_video_frame_count": original_frame_count,
                    "input_video_size": [int(original_width), int(original_height)],
                    "source_fps": float(original_fps),
                    "frame_skip": int(frame_skip),
                    "effective_fps": float(fps),
                    "after_skip_frame_count": after_skip_count,
                    "snap_to_8n_plus_1": bool(snap_to_8n_plus_1),
                    "trim_mode": trim_mode,
                    **fit_info,
                    **spatial_info,
                    "final_frame_count": int(frame_count),
                    "final_size": [int(frames.shape[2]), int(frames.shape[1])],
                    "frames_output_shape": list(frames.shape),
                    "video_latent_shape": (
                        list(video_latent["samples"].shape)
                        if "samples" in video_latent and hasattr(video_latent["samples"], "shape")
                        else None
                    ),
                }
            )

            return (
                video_latent,
                frames,
                float(fps),
                json.dumps(debug, ensure_ascii=False, indent=2),
            )
        except Exception as exc:
            debug.update(
                {
                    "error": str(exc),
                    "traceback": traceback.format_exc(),
                }
            )
            raise RuntimeError(json.dumps(debug, ensure_ascii=False, indent=2))


class NanFengLTXAudioDub:
    @classmethod
    def INPUT_TYPES(cls):
        checkpoint_options, checkpoint_default = _combo_options("checkpoints", DEFAULT_CHECKPOINT)
        text_encoder_options, text_encoder_default = _combo_options("text_encoders", DEFAULT_TEXT_ENCODER)
        return {
            "required": {
                "model": ("MODEL",),
                "vae": ("VAE",),
                "positive_prompt": (
                    "STRING",
                    {
                        "multiline": True,
                        "dynamicPrompts": True,
                        "default": DEFAULT_POSITIVE_PROMPT,
                    },
                ),
                "negative_prompt": (
                    "STRING",
                    {
                        "multiline": True,
                        "dynamicPrompts": True,
                        "default": DEFAULT_NEGATIVE_PROMPT,
                    },
                ),
                "frame_rate": ("FLOAT", {"default": 25.0, "min": 0.01, "max": 120.0, "step": 0.01}),
                "frames_number": ("INT", {"default": 97, "min": 1, "max": 4096, "step": 1}),
                "input_scale": ("FLOAT", {"default": 1.0, "min": 0.05, "max": 4.0, "step": 0.05}),
                "latent_scale": ("FLOAT", {"default": 1.0, "min": 0.05, "max": 4.0, "step": 0.05}),
                "checkpoint_name": (checkpoint_options, {"default": checkpoint_default}),
                "text_encoder": (text_encoder_options, {"default": text_encoder_default}),
                "text_encoder_device": (["default", "cpu"], {"default": "default"}),
                "cfg": ("FLOAT", {"default": 1.1, "min": 0.0, "max": 20.0, "step": 0.01}),
                "sampler_name": (list(comfy.samplers.SAMPLER_NAMES), {"default": "euler"}),
                "sigmas": ("STRING", {"default": DEFAULT_SIGMAS, "multiline": False}),
                "noise_seed": (
                    "INT",
                    {
                        "default": 0,
                        "min": 0,
                        "max": 0xFFFFFFFFFFFFFFFF,
                        "control_after_generate": True,
                    },
                ),
                "snap_to_8n_plus_1": ("BOOLEAN", {"default": True}),
                "trim_mode": (
                    ["auto", "tail", "head", "pad_tail", "pad_head", "strict_error"],
                    {"default": "auto"},
                ),
                "frame_skip": ("INT", {"default": 1, "min": 1, "max": 16, "step": 1}),
            },
            "optional": {
                "video": ("VIDEO",),
                "image": ("IMAGE",),
                "video_latent": ("LATENT",),
            },
        }

    RETURN_TYPES = ("AUDIO",)
    RETURN_NAMES = ("audio",)
    FUNCTION = "dub_audio"
    CATEGORY = CATEGORY_NAME

    def dub_audio(
        self,
        model,
        vae,
        positive_prompt,
        negative_prompt,
        frame_rate,
        frames_number,
        input_scale,
        latent_scale,
        checkpoint_name,
        text_encoder,
        text_encoder_device,
        cfg,
        sampler_name,
        sigmas,
        noise_seed,
        snap_to_8n_plus_1,
        trim_mode,
        frame_skip,
        video=None,
        image=None,
        video_latent=None,
    ):
        debug: Dict[str, Any] = {
            "node": "南风配音节点",
            "ok": False,
        }

        try:
            source_video_latent, source_frame_count, resolved_fps, source_meta = _prepare_source_video_latent(
                vae=vae,
                video=video,
                image=image,
                video_latent=video_latent,
                frame_rate=float(frame_rate),
                frames_number=int(frames_number),
                input_scale=float(input_scale),
                latent_scale=float(latent_scale),
                snap_to_8n_plus_1=bool(snap_to_8n_plus_1),
                trim_mode=trim_mode,
                frame_skip=int(frame_skip),
            )
            debug.update(source_meta)

            clip = _execute_node(
                LTXAVTextEncoderLoader,
                text_encoder=text_encoder,
                ckpt_name=checkpoint_name,
                device=text_encoder_device,
            )[0]

            clip_text_encode = CLIPTextEncode()
            positive = _execute_node(clip_text_encode, clip=clip, text=positive_prompt)[0]
            negative = _execute_node(clip_text_encode, clip=clip, text=negative_prompt)[0]

            positive, negative = _execute_node(
                LTXVConditioning,
                positive=positive,
                negative=negative,
                frame_rate=float(resolved_fps),
            )

            audio_vae = _execute_node(LTXVAudioVAELoader, ckpt_name=checkpoint_name)[0]
            audio_latent = _execute_node(
                LTXVEmptyLatentAudio,
                frames_number=int(source_frame_count),
                frame_rate=max(1, int(round(float(resolved_fps)))),
                batch_size=1,
                audio_vae=audio_vae,
            )[0]

            masked_video_latent, masked_audio_latent = _prepare_audio_only_masks(
                source_video_latent,
                audio_latent,
            )

            av_latent = _execute_node(
                LTXVConcatAVLatent,
                video_latent=masked_video_latent,
                audio_latent=masked_audio_latent,
            )[0]

            noise = _execute_node(RandomNoise, noise_seed=int(noise_seed))[0]
            guider = _execute_node(
                CFGGuider,
                model=model,
                positive=positive,
                negative=negative,
                cfg=float(cfg),
            )[0]
            sampler = _execute_node(KSamplerSelect, sampler_name=sampler_name)[0]
            sigma_values = _execute_node(ManualSigmas, sigmas=sigmas)[0]

            sampled_av_latent, _ = _execute_node(
                SamplerCustomAdvanced,
                noise=noise,
                guider=guider,
                sampler=sampler,
                sigmas=sigma_values,
                latent_image=av_latent,
            )

            _, sampled_audio_latent = _execute_node(
                LTXVSeparateAVLatent,
                av_latent=sampled_av_latent,
            )

            audio = _execute_node(
                LTXVAudioVAEDecode,
                samples=sampled_audio_latent,
                audio_vae=audio_vae,
            )[0]

            debug.update(
                {
                    "ok": True,
                    "resolved_fps": float(resolved_fps),
                    "resolved_frame_count": int(source_frame_count),
                    "checkpoint_name": checkpoint_name,
                    "text_encoder": text_encoder,
                    "sampler_name": sampler_name,
                    "cfg": float(cfg),
                    "noise_seed": int(noise_seed),
                    "audio_latent_shape": _shape_of(audio_latent.get("samples")),
                    "video_latent_shape": _shape_of(source_video_latent.get("samples")),
                    "sampled_audio_latent_shape": _shape_of(sampled_audio_latent.get("samples")),
                    "audio_waveform_shape": _shape_of(audio.get("waveform")),
                    "audio_sample_rate": _to_int(audio.get("sample_rate"), 0),
                }
            )

            return (audio,)
        except Exception as exc:
            debug.update(
                {
                    "error": str(exc),
                    "traceback": traceback.format_exc(),
                }
            )
            raise RuntimeError(json.dumps(debug, ensure_ascii=False, indent=2))


NODE_CLASS_MAPPINGS = {
    "NanFengLTXVideoLatent": NanFengLTXAudioDub,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "NanFengLTXVideoLatent": "南风配音节点",
}
