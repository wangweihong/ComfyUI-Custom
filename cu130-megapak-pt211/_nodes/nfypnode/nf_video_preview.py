import datetime
import hashlib
import itertools
import json
import os
import re
import shutil
import subprocess
import tempfile
import wave
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from comfy.utils import ProgressBar

import folder_paths

try:
    from aiohttp import web
except Exception:
    web = None

try:
    from server import PromptServer
except Exception:
    PromptServer = None


ENCODE_ARGS = ("utf-8", "backslashreplace")
SOURCE_EXTENSION = "mp4"
SOURCE_VIDEO_CODEC = "libx264"
SOURCE_PIX_FMT = "yuv420p"
SOURCE_AUDIO_CODEC = "aac"
SOURCE_AUDIO_BITRATE = "192k"
SOURCE_CRF = 14
SOURCE_MOVFLAGS = "+faststart"
SOURCE_OUTPUT_ARGS = [
    "-preset", "medium",
    "-color_range", "tv",
    "-colorspace", "bt709",
    "-color_primaries", "bt709",
    "-color_trc", "bt709",
]
SOURCE_VIDEO_FILTERS = [
    "scale=out_color_matrix=bt709",
]


class MultiInput(str):
    def __new__(cls, string, allowed_types="*"):
        res = super().__new__(cls, string)
        res.allowed_types = allowed_types
        return res

    def __ne__(self, other):
        if self.allowed_types == "*" or other == "*":
            return False
        return other not in self.allowed_types


class ContainsAll(dict):
    def __contains__(self, other):
        return True

    def __getitem__(self, key):
        return super().get(key, (None, {}))


imageOrLatent = MultiInput("IMAGE", ["IMAGE", "LATENT"])
floatOrInt = MultiInput("FLOAT", ["FLOAT", "INT"])


MODULE_DIR = Path(__file__).resolve().parent
CONFIG_DIR = MODULE_DIR / "nf_video_formats"
CACHE_DIR = MODULE_DIR / "_nf_video_preview_cache"


DEFAULT_CONFIGS: Dict[str, Dict] = {
    "h264-mp4.json": {
        "display_name": "h264-mp4",
        "extension": "mp4",
        "video_codec": "libx264",
        "pix_fmt": "yuv420p",
        "audio_codec": "aac",
        "audio_bitrate": "192k",
        "movflags": "+faststart",
        "dim_alignment": 2,
        "input_color_depth": "8bit",
        "video_filters": [
            "scale=out_color_matrix=bt709"
        ],
        "output_args": [
            "-preset", "medium",
            "-color_range", "tv",
            "-colorspace", "bt709",
            "-color_primaries", "bt709",
            "-color_trc", "bt709"
        ],
        "preview_style": {
            "filter": "none",
            "background": "rgba(12,16,24,0.65)",
            "border": "1px solid rgba(110,150,255,0.28)",
            "box_shadow": "0 0 0 1px rgba(110,150,255,0.05) inset"
        }
    },
    "h264-mp4去AI加噪.json": {
        "display_name": "h264-mp4去AI加噪",
        "extension": "mp4",
        "video_codec": "libx264",
        "pix_fmt": "yuv420p",
        "audio_codec": "aac",
        "audio_bitrate": "192k",
        "movflags": "+faststart",
        "dim_alignment": 2,
        "input_color_depth": "8bit",
        "video_filters": [
            "hqdn3d=1.3:1.2:6.0:6.0",
            "unsharp=5:5:0.35:5:5:0.00"
        ],
        "output_args": [
            "-preset", "medium",
            "-color_range", "tv",
            "-colorspace", "bt709",
            "-color_primaries", "bt709",
            "-color_trc", "bt709"
        ],
        "preview_style": {
            "filter": "contrast(1.06) saturate(1.02) brightness(1.01)",
            "background": "rgba(18,22,28,0.68)",
            "border": "1px solid rgba(112,212,255,0.33)",
            "box_shadow": "0 0 0 1px rgba(112,212,255,0.08) inset"
        }
    },
    "h264-mp4调色.json": {
        "display_name": "h264-mp4调色",
        "extension": "mp4",
        "video_codec": "libx264",
        "pix_fmt": "yuv420p",
        "audio_codec": "aac",
        "audio_bitrate": "192k",
        "movflags": "+faststart",
        "dim_alignment": 2,
        "input_color_depth": "8bit",
        "video_filters": [
            "eq=contrast=1.06:brightness=0.015:saturation=1.12:gamma=1.00"
        ],
        "output_args": [
            "-preset", "medium",
            "-color_range", "tv",
            "-colorspace", "bt709",
            "-color_primaries", "bt709",
            "-color_trc", "bt709"
        ],
        "preview_style": {
            "filter": "contrast(1.08) saturate(1.16) brightness(1.02)",
            "background": "rgba(18,18,30,0.70)",
            "border": "1px solid rgba(255,168,110,0.35)",
            "box_shadow": "0 0 0 1px rgba(255,168,110,0.08) inset"
        }
    }
}


def _ensure_default_configs() -> None:
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    for name, content in DEFAULT_CONFIGS.items():
        path = CONFIG_DIR / name
        if not path.exists():
            path.write_text(json.dumps(content, ensure_ascii=False, indent=2), encoding="utf-8")


_ensure_default_configs()


def _get_ffmpeg_path() -> Optional[str]:
    forced = os.environ.get("VHS_FORCE_FFMPEG_PATH")
    if forced:
        return forced
    try:
        from imageio_ffmpeg import get_ffmpeg_exe
        return get_ffmpeg_exe()
    except Exception:
        pass
    return shutil.which("ffmpeg") or shutil.which("ffmpeg.exe")


FFMPEG_PATH = _get_ffmpeg_path()


def _tensor_to_uint(tensor: torch.Tensor, bits: int) -> np.ndarray:
    tensor = tensor.detach().cpu().numpy() * ((2 ** bits) - 1) + 0.5
    tensor = np.clip(tensor, 0, (2 ** bits) - 1)
    return tensor.astype(np.uint16 if bits == 16 else np.uint8)


def tensor_to_bytes(tensor: torch.Tensor) -> np.ndarray:
    return _tensor_to_uint(tensor, 8)


def tensor_to_shorts(tensor: torch.Tensor) -> np.ndarray:
    return _tensor_to_uint(tensor, 16)


def _list_config_names() -> List[str]:
    _ensure_default_configs()
    names = [p.stem for p in CONFIG_DIR.glob("*.json") if p.is_file()]
    names.sort(key=lambda x: x.lower())
    if not names:
        names = ["h264-mp4"]
    return names


def _load_config(config_name: str) -> Dict:
    _ensure_default_configs()
    path = CONFIG_DIR / f"{config_name}.json"
    if not path.exists():
        raise FileNotFoundError(f"找不到配置文件: {path}")
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    data.setdefault("extension", "mp4")
    data.setdefault("video_codec", "libx264")
    data.setdefault("pix_fmt", "yuv420p")
    data.setdefault("audio_codec", "aac")
    data.setdefault("audio_bitrate", "192k")
    data.setdefault("movflags", "+faststart")
    data.setdefault("dim_alignment", 2)
    data.setdefault("input_color_depth", "8bit")
    data.setdefault("video_filters", [])
    data.setdefault("output_args", [])
    data.setdefault("preview_style", {})
    return data


def _get_preview_styles() -> Dict[str, Dict]:
    styles: Dict[str, Dict] = {}
    for name in _list_config_names():
        try:
            styles[name] = _load_config(name).get("preview_style", {})
        except Exception:
            styles[name] = {}
    return styles


def _normalize_prefix_to_relative_parts(filename_prefix: str) -> List[str]:
    prefix = str(filename_prefix or "NF_video").replace("\\", "/").strip()
    if not prefix:
        prefix = "NF_video"
    parts = [p for p in prefix.split("/") if p not in ("", ".")]
    cleaned: List[str] = []
    for part in parts:
        if part == "..":
            continue
        cleaned.append(part)
    if not cleaned:
        cleaned = ["NF_video"]
    return cleaned


def _split_prefix_parts(filename_prefix: str) -> Tuple[List[str], str]:
    parts = _normalize_prefix_to_relative_parts(filename_prefix)
    subfolder_parts = parts[:-1]
    stem = Path(parts[-1]).stem or "NF_video"
    return subfolder_parts, stem


def _sanitize_rel_subfolder(parts: List[str]) -> str:
    return "/".join(parts)


def _safe_token(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", text).strip("_") or "nf_preview"


def _make_session_token(unique_id: Optional[str], filename_prefix: str) -> str:
    raw = f"{unique_id or 'no_uid'}|{'/'.join(_normalize_prefix_to_relative_parts(filename_prefix))}"
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:24]


def _manifest_path(token: str) -> Path:
    return CACHE_DIR / f"{_safe_token(token)}_manifest.json"


def _read_manifest(token: str) -> Dict:
    path = _manifest_path(token)
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _write_manifest(token: str, data: Dict) -> None:
    path = _manifest_path(token)
    payload = dict(data)
    payload["updated_at"] = datetime.datetime.now().isoformat(timespec="seconds")
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _hash_tensor_brief(tensor: torch.Tensor, label: str) -> str:
    if not isinstance(tensor, torch.Tensor):
        tensor = torch.as_tensor(tensor)
    tensor = tensor.detach().to(device="cpu")
    h = hashlib.sha1()
    h.update(label.encode("utf-8"))
    h.update(str(tuple(tensor.shape)).encode("utf-8"))
    h.update(str(tensor.dtype).encode("utf-8"))
    if tensor.numel() <= 0:
        return h.hexdigest()

    if tensor.ndim == 0:
        flat = tensor.reshape(-1).to(dtype=torch.float32)
        h.update(flat.numpy().tobytes())
        return h.hexdigest()

    candidate_indices = [0, max(tensor.shape[0] // 2, 0), max(tensor.shape[0] - 1, 0)]
    seen = set()
    for idx in candidate_indices:
        idx = max(0, min(idx, tensor.shape[0] - 1))
        if idx in seen:
            continue
        seen.add(idx)
        piece = tensor[idx] if tensor.ndim >= 1 else tensor
        flat = piece.reshape(-1)
        if flat.numel() <= 0:
            continue
        sample_count = min(int(flat.numel()), 8192)
        step = max(int(flat.numel() // sample_count), 1)
        sampled = flat[::step][:sample_count].to(dtype=torch.float32).contiguous()
        h.update(sampled.numpy().tobytes())
    return h.hexdigest()


def _compute_source_signature(images, audio, frame_rate) -> str:
    h = hashlib.sha1()
    h.update(f"fps={frame_rate}".encode("utf-8"))

    if isinstance(images, dict) and isinstance(images.get("samples"), torch.Tensor):
        latent_samples = images["samples"]
        h.update(b"images:latent")
        h.update(_hash_tensor_brief(latent_samples, "latent_samples").encode("utf-8"))
    elif isinstance(images, torch.Tensor):
        h.update(b"images:tensor")
        h.update(_hash_tensor_brief(images, "image_tensor").encode("utf-8"))
    else:
        h.update(f"images_type={type(images).__name__}".encode("utf-8"))
        h.update(repr(images).encode("utf-8", errors="ignore")[:2048])

    waveform, sample_rate = _get_audio_waveform_and_rate(audio)
    if waveform is not None and sample_rate is not None:
        h.update(f"audio_sr={sample_rate}".encode("utf-8"))
        h.update(_hash_tensor_brief(waveform, "audio_waveform").encode("utf-8"))
    else:
        h.update(b"audio:none")

    return h.hexdigest()


def _pad_image_tensor(image: torch.Tensor, target_w: int, target_h: int) -> torch.Tensor:
    if image.ndim != 3:
        return image
    h, w = image.shape[0], image.shape[1]
    pad_l = max((target_w - w) // 2, 0)
    pad_r = max(target_w - w - pad_l, 0)
    pad_t = max((target_h - h) // 2, 0)
    pad_b = max(target_h - h - pad_t, 0)
    if pad_l == pad_r == pad_t == pad_b == 0:
        return image
    chw = image.permute(2, 0, 1).unsqueeze(0).to(dtype=torch.float32)
    padded = F.pad(chw, (pad_l, pad_r, pad_t, pad_b), mode="replicate")
    return padded.squeeze(0).permute(1, 2, 0)


def _prepare_images_iter(images, vae):
    if isinstance(images, dict):
        if vae is None:
            raise ValueError("传入的是 LATENT，但没有连接 VAE，无法输出视频。")
        latent_samples = images["samples"]
        downscale_ratio = int(getattr(vae, "downscale_ratio", 8))
        width = int(latent_samples.shape[-1]) * downscale_ratio
        height = int(latent_samples.shape[-2]) * downscale_ratio
        images_iter: Iterable[torch.Tensor] = _batched_decode(latent_samples, vae, width, height)
        num_frames = int(latent_samples.shape[0])
    elif isinstance(images, torch.Tensor):
        images_iter = iter(images)
        num_frames = int(images.shape[0]) if images.ndim >= 1 else 1
    else:
        images_iter = iter(images)
        num_frames = None

    try:
        first_image = next(iter(images_iter))
    except StopIteration:
        return None, None, 0

    images_iter = itertools.chain([first_image], images_iter)
    while isinstance(first_image, torch.Tensor) and len(first_image.shape) > 3:
        first_image = first_image[0]

    if not isinstance(first_image, torch.Tensor):
        first_image = torch.as_tensor(first_image)

    return images_iter, first_image, num_frames


def _get_audio_waveform_and_rate(audio):
    if audio is None:
        return None, None

    waveform = None
    sample_rate = None

    if isinstance(audio, dict):
        waveform = audio.get("waveform")
        sample_rate = audio.get("sample_rate")
    else:
        waveform = getattr(audio, "waveform", None)
        sample_rate = getattr(audio, "sample_rate", None)

    if waveform is None or sample_rate is None:
        return None, None

    if not isinstance(waveform, torch.Tensor):
        waveform = torch.as_tensor(waveform)

    if waveform.ndim == 3:
        if waveform.shape[0] == 1:
            waveform = waveform[0]
        else:
            waveform = waveform.reshape(-1, waveform.shape[-1])
    elif waveform.ndim == 1:
        waveform = waveform.unsqueeze(0)
    elif waveform.ndim != 2:
        raise ValueError(f"不支持的音频 waveform 维度: {tuple(waveform.shape)}")

    if waveform.shape[0] > waveform.shape[1] and waveform.shape[1] <= 8:
        waveform = waveform.transpose(0, 1)

    waveform = waveform.detach().to(dtype=torch.float32, device="cpu").contiguous()
    waveform = torch.nan_to_num(waveform, nan=0.0, posinf=0.0, neginf=0.0).clamp(-1.0, 1.0)
    return waveform, int(sample_rate)


def _write_temp_wav_from_audio(audio, cache_dir: Path) -> Tuple[Optional[str], Optional[int]]:
    waveform, sample_rate = _get_audio_waveform_and_rate(audio)
    if waveform is None or sample_rate is None:
        return None, None

    channels = int(waveform.shape[0])
    if channels <= 0:
        return None, None

    pcm = (waveform.transpose(0, 1).numpy() * 32767.0).round().astype(np.int16)

    tmp = tempfile.NamedTemporaryFile(prefix="nf_audio_", suffix=".wav", dir=str(cache_dir), delete=False)
    tmp_path = tmp.name
    tmp.close()

    with wave.open(tmp_path, "wb") as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm.tobytes())

    return tmp_path, channels


def _guess_video_mime(extension: str) -> str:
    ext = extension.lower().lstrip(".")
    if ext == "mp4":
        return "video/mp4"
    if ext == "webm":
        return "video/webm"
    if ext == "mov":
        return "video/quicktime"
    if ext == "mkv":
        return "video/x-matroska"
    if ext == "avi":
        return "video/x-msvideo"
    return f"video/{ext}"


def _batched_decode(latents: torch.Tensor, vae, width: int, height: int) -> Iterator[torch.Tensor]:
    frames_per_batch = max((1920 * 1080 * 16) // max(width * height, 1), 1)
    total = int(latents.shape[0])
    pbar = ProgressBar(total)
    for start in range(0, total, frames_per_batch):
        batch = latents[start:start + frames_per_batch]
        decoded = vae.decode(batch)
        for frame in decoded:
            yield frame
        pbar.update(batch.shape[0])


def _encode_raw_frames_to_video(
    *,
    images_iter,
    first_image: torch.Tensor,
    num_frames: Optional[int],
    frame_rate,
    output_path: str,
    crf: int,
    video_codec: str,
    pix_fmt: str,
    input_color_depth: str,
    dim_alignment: int,
    movflags: Optional[str],
    output_args: List[str],
    ffmpeg_filters: List[str],
) -> int:
    src_h = int(first_image.shape[0])
    src_w = int(first_image.shape[1])
    target_w = src_w + (-src_w % max(int(dim_alignment), 1))
    target_h = src_h + (-src_h % max(int(dim_alignment), 1))

    if input_color_depth == "16bit":
        to_frame = tensor_to_shorts
        input_pix_fmt = "rgba64" if first_image.shape[-1] == 4 else "rgb48"
    else:
        to_frame = tensor_to_bytes
        input_pix_fmt = "rgba" if first_image.shape[-1] == 4 else "rgb24"

    final_filters = []
    if target_w != src_w or target_h != src_h:
        final_filters.append(f"pad={target_w}:{target_h}:(ow-iw)/2:(oh-ih)/2")
    final_filters.extend([f for f in ffmpeg_filters if str(f).strip()])

    cmd = [
        FFMPEG_PATH,
        "-y",
        "-v", "error",
        "-f", "rawvideo",
        "-pix_fmt", input_pix_fmt,
        "-s", f"{src_w}x{src_h}",
        "-r", str(frame_rate),
        "-i", "-",
        "-an",
        "-c:v", str(video_codec),
        "-pix_fmt", str(pix_fmt),
        "-crf", str(crf),
    ]

    if movflags:
        cmd += ["-movflags", str(movflags)]

    cmd += list(output_args or [])

    if final_filters:
        cmd += ["-vf", ",".join(final_filters)]

    cmd += [output_path]

    process = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
    written_frames = 0
    pbar = ProgressBar(int(num_frames) if num_frames is not None else 0)
    try:
        for frame in images_iter:
            if not isinstance(frame, torch.Tensor):
                frame = torch.as_tensor(frame)
            while frame.ndim > 3:
                frame = frame[0]
            if frame.shape[0] != src_h or frame.shape[1] != src_w:
                frame = _pad_image_tensor(frame, src_w, src_h)
                frame = frame[:src_h, :src_w, :]
            frame_bytes = to_frame(frame).tobytes()
            assert process.stdin is not None
            process.stdin.write(frame_bytes)
            written_frames += 1
            if num_frames is not None:
                pbar.update(1)
        assert process.stdin is not None
        process.stdin.close()
        stderr = process.stderr.read() if process.stderr is not None else b""
        return_code = process.wait()
        if return_code != 0:
            raise RuntimeError(stderr.decode(*ENCODE_ARGS))
    finally:
        try:
            if process.stdin and not process.stdin.closed:
                process.stdin.close()
        except Exception:
            pass
    return written_frames


def _mux_video_with_audio(
    *,
    video_path: str,
    audio,
    final_path: str,
    audio_codec: str,
    audio_bitrate: Optional[str],
) -> bool:
    temp_audio_path = None
    try:
        temp_audio_path, _channels = _write_temp_wav_from_audio(audio, CACHE_DIR)
        if temp_audio_path is None:
            os.replace(video_path, final_path)
            return False

        mux_tmp = str(CACHE_DIR / f"{Path(final_path).stem}_mux_tmp{Path(final_path).suffix}")
        mux_cmd = [
            FFMPEG_PATH,
            "-y",
            "-v", "error",
            "-i", video_path,
            "-i", temp_audio_path,
            "-map", "0:v:0",
            "-map", "1:a:0",
            "-c:v", "copy",
            "-c:a", str(audio_codec),
        ]
        if audio_bitrate:
            mux_cmd += ["-b:a", str(audio_bitrate)]
        mux_cmd += ["-shortest", mux_tmp]
        mux_res = subprocess.run(mux_cmd, capture_output=True)
        if mux_res.returncode != 0:
            raise RuntimeError("音频合并失败:\n" + mux_res.stderr.decode(*ENCODE_ARGS))
        os.replace(mux_tmp, final_path)
        if os.path.exists(video_path):
            os.remove(video_path)
        return True
    finally:
        if temp_audio_path and os.path.exists(temp_audio_path):
            try:
                os.remove(temp_audio_path)
            except Exception:
                pass


def _build_source_master(images, vae, audio, frame_rate, source_master_path: str) -> Dict:
    images_iter, first_image, num_frames = _prepare_images_iter(images, vae)
    if images_iter is None or first_image is None:
        raise ValueError("没有可写入的视频帧。")

    tmp_video = str(CACHE_DIR / f"{Path(source_master_path).stem}_video_tmp.{SOURCE_EXTENSION}")
    written_frames = _encode_raw_frames_to_video(
        images_iter=images_iter,
        first_image=first_image,
        num_frames=num_frames,
        frame_rate=frame_rate,
        output_path=tmp_video,
        crf=SOURCE_CRF,
        video_codec=SOURCE_VIDEO_CODEC,
        pix_fmt=SOURCE_PIX_FMT,
        input_color_depth="8bit",
        dim_alignment=2,
        movflags=SOURCE_MOVFLAGS,
        output_args=SOURCE_OUTPUT_ARGS,
        ffmpeg_filters=SOURCE_VIDEO_FILTERS,
    )
    has_audio = _mux_video_with_audio(
        video_path=tmp_video,
        audio=audio,
        final_path=source_master_path,
        audio_codec=SOURCE_AUDIO_CODEC,
        audio_bitrate=SOURCE_AUDIO_BITRATE,
    )
    if not has_audio and os.path.exists(tmp_video):
        os.replace(tmp_video, source_master_path)

    return {
        "written_frames": int(written_frames),
        "has_audio": bool(has_audio),
    }


def _allocate_numbered_preview_path(output_root: str, filename_prefix: str, extension: str) -> Tuple[str, str, str]:
    subfolder_parts, stem = _split_prefix_parts(filename_prefix)
    full_output_folder = os.path.join(output_root, *subfolder_parts) if subfolder_parts else output_root
    os.makedirs(full_output_folder, exist_ok=True)
    rel_subfolder = _sanitize_rel_subfolder(subfolder_parts)

    ext = extension.lstrip(".")
    pattern = re.compile(rf"^{re.escape(stem)}_(\d+)\.{re.escape(ext)}$", re.IGNORECASE)
    max_index = 0
    for entry in os.listdir(full_output_folder):
        match = pattern.match(entry)
        if match:
            max_index = max(max_index, int(match.group(1)))

    next_index = max_index + 1
    filename = f"{stem}_{next_index:05d}.{ext}"
    return os.path.join(full_output_folder, filename), rel_subfolder, filename


def _ensure_preview_path_extension(preview_fullpath: str, extension: str) -> str:
    extension = extension.lstrip(".")
    p = Path(preview_fullpath)
    if p.suffix.lower() == f".{extension.lower()}":
        return str(p)
    return str(p.with_suffix(f".{extension}"))


def _render_preview_from_source(source_master_path: str, preview_fullpath: str, config: Dict, crf: int) -> None:
    extension = str(config.get("extension", "mp4"))
    preview_fullpath = _ensure_preview_path_extension(preview_fullpath, extension)
    preview_tmp = str(CACHE_DIR / f"{Path(preview_fullpath).stem}_preview_build_tmp.{extension.lstrip('.')}" )
    preview_dir = str(Path(preview_fullpath).parent)
    os.makedirs(preview_dir, exist_ok=True)

    cmd = [
        FFMPEG_PATH,
        "-y",
        "-v", "error",
        "-i", source_master_path,
        "-map", "0:v:0",
        "-map", "0:a?",
        "-c:v", str(config.get("video_codec", "libx264")),
        "-pix_fmt", str(config.get("pix_fmt", "yuv420p")),
        "-crf", str(int(crf)),
    ]

    movflags = config.get("movflags")
    if movflags:
        cmd += ["-movflags", str(movflags)]

    cmd += list(config.get("output_args", []))

    ffmpeg_filters = [f for f in config.get("video_filters", []) if str(f).strip()]
    if ffmpeg_filters:
        cmd += ["-vf", ",".join(ffmpeg_filters)]

    audio_codec = config.get("audio_codec")
    audio_bitrate = config.get("audio_bitrate")
    if audio_codec:
        cmd += ["-c:a", str(audio_codec)]
        if audio_bitrate:
            cmd += ["-b:a", str(audio_bitrate)]
    else:
        cmd += ["-c:a", "copy"]

    cmd += ["-shortest", preview_tmp]

    result = subprocess.run(cmd, capture_output=True)
    if result.returncode != 0:
        raise RuntimeError(result.stderr.decode(*ENCODE_ARGS))

    os.replace(preview_tmp, preview_fullpath)


def _build_preview_payload(*, token: str, preview_fullpath: str, subfolder: str, frame_rate, has_audio: bool, format_name: str, crf: int, preview_style: Dict) -> Dict:
    extension = Path(preview_fullpath).suffix.lstrip(".")
    cache_bust = datetime.datetime.now().strftime("%Y%m%d%H%M%S%f")
    return {
        "filename": os.path.basename(preview_fullpath),
        "subfolder": subfolder,
        "type": "output",
        "format": _guess_video_mime(extension),
        "frame_rate": frame_rate,
        "fullpath": preview_fullpath,
        "created_at": datetime.datetime.now().isoformat(timespec="seconds"),
        "autoplay": True,
        "loop": True,
        "muted": False,
        "controls": True,
        "nf_preview_format": format_name,
        "nf_preview_crf": int(crf),
        "nf_preview_style": preview_style or {},
        "nf_has_audio": bool(has_audio),
        "nf_cache_bust": cache_bust,
        "nf_force_overwrite": True,
        "nf_preview_token": token,
    }


class NanFengVideoPreview:
    @classmethod
    def INPUT_TYPES(cls):
        config_names = _list_config_names()
        return {
            "required": {
                "images": (imageOrLatent,),
                "frame_rate": (floatOrInt, {"default": 24, "min": 1, "step": 1}),
                "filename_prefix": ("STRING", {"default": "NF_video"}),
                "format": (
                    config_names,
                    {
                        "default": config_names[0] if config_names else "h264-mp4",
                        "nf_preview_styles": _get_preview_styles(),
                    },
                ),
                "crf": ("INT", {"default": 19, "min": 0, "max": 100, "step": 1}),
                "save_output": ("BOOLEAN", {"default": True}),
            },
            "optional": {
                "audio": ("AUDIO",),
                "meta_batch": ("VHS_BatchManager",),
                "vae": ("VAE",),
            },
            "hidden": ContainsAll({
                "prompt": "PROMPT",
                "extra_pnginfo": "EXTRA_PNGINFO",
                "unique_id": "UNIQUE_ID",
            }),
        }

    RETURN_TYPES = ("VHS_FILENAMES",)
    RETURN_NAMES = ("文件名",)
    OUTPUT_NODE = True
    FUNCTION = "combine_video"
    CATEGORY = "南风阳平/视频"

    def combine_video(
        self,
        images=None,
        frame_rate=24,
        filename_prefix="NF_video",
        format="h264-mp4",
        crf=19,
        save_output=True,
        audio=None,
        meta_batch=None,
        vae=None,
        prompt=None,
        extra_pnginfo=None,
        unique_id=None,
        latents=None,
        **kwargs,
    ):
        del meta_batch, prompt, extra_pnginfo, kwargs, save_output

        if latents is not None:
            images = latents
        if images is None:
            return ((True, []),)

        if FFMPEG_PATH is None:
            raise ProcessLookupError("找不到 ffmpeg。请安装 ffmpeg 或 imageio-ffmpeg 后再使用南风视频预览。")

        config = _load_config(format)
        extension = str(config.get("extension", "mp4")).lstrip(".")
        output_dir = folder_paths.get_output_directory()
        os.makedirs(output_dir, exist_ok=True)

        session_token = _make_session_token(str(unique_id or ""), filename_prefix)
        manifest = _read_manifest(session_token)
        source_signature = _compute_source_signature(images, audio, frame_rate)

        source_master_path = str(CACHE_DIR / f"{session_token}_source.{SOURCE_EXTENSION}")
        source_changed = manifest.get("source_signature") != source_signature or not os.path.exists(source_master_path)

        preview_fullpath = manifest.get("preview_fullpath")
        preview_subfolder = manifest.get("preview_subfolder", "")
        preview_filename = manifest.get("preview_filename", "")

        if source_changed or not preview_fullpath or not os.path.exists(preview_fullpath):
            preview_fullpath, preview_subfolder, preview_filename = _allocate_numbered_preview_path(
                output_root=output_dir,
                filename_prefix=filename_prefix,
                extension=extension,
            )
        else:
            preview_fullpath = _ensure_preview_path_extension(str(preview_fullpath), extension)
            preview_filename = os.path.basename(preview_fullpath)

        if source_changed:
            source_info = _build_source_master(images, vae, audio, frame_rate, source_master_path)
            has_audio = bool(source_info.get("has_audio", False))
            written_frames = int(source_info.get("written_frames", 0))
        else:
            has_audio = bool(manifest.get("has_audio", False))
            written_frames = int(manifest.get("written_frames", 0))

        _render_preview_from_source(source_master_path, preview_fullpath, config, crf)

        preview_payload = _build_preview_payload(
            token=session_token,
            preview_fullpath=preview_fullpath,
            subfolder=preview_subfolder,
            frame_rate=frame_rate,
            has_audio=has_audio,
            format_name=format,
            crf=int(crf),
            preview_style=config.get("preview_style", {}),
        )
        preview_payload["written_frames"] = written_frames

        manifest.update({
            "token": session_token,
            "source_signature": source_signature,
            "source_master_path": source_master_path,
            "preview_fullpath": preview_fullpath,
            "preview_subfolder": preview_subfolder,
            "preview_filename": preview_filename,
            "frame_rate": frame_rate,
            "has_audio": has_audio,
            "written_frames": written_frames,
            "last_format": format,
            "last_crf": int(crf),
            "last_extension": extension,
        })
        _write_manifest(session_token, manifest)

        return {"ui": {"gifs": [preview_payload]}, "result": ((True, [preview_fullpath]),)}


NODE_CLASS_MAPPINGS = {
    "NF_VideoPreview": NanFengVideoPreview,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "NF_VideoPreview": "南风视频预览",
}


async def _handle_preview_rebuild(request):
    if web is None:
        return None

    try:
        payload = await request.json()
    except Exception:
        return web.json_response({"ok": False, "error": "请求 JSON 无效。"}, status=400)

    token = str(payload.get("token") or "").strip()
    format_name = str(payload.get("format") or "").strip()
    crf = int(payload.get("crf", 19))

    if not token:
        return web.json_response({"ok": False, "error": "缺少 token。"}, status=400)
    if not format_name:
        return web.json_response({"ok": False, "error": "缺少 format。"}, status=400)

    manifest = _read_manifest(token)
    if not manifest:
        return web.json_response({"ok": False, "error": "没有找到该预览会话，请先执行一次节点。"}, status=404)

    source_master_path = str(manifest.get("source_master_path") or "")
    if not source_master_path or not os.path.exists(source_master_path):
        return web.json_response({"ok": False, "error": "原始母版视频不存在，请重新执行一次节点。"}, status=404)

    try:
        config = _load_config(format_name)
        preview_fullpath = _ensure_preview_path_extension(str(manifest.get("preview_fullpath") or ""), str(config.get("extension", "mp4")))
        if not preview_fullpath:
            return web.json_response({"ok": False, "error": "预览输出路径无效。"}, status=500)

        _render_preview_from_source(source_master_path, preview_fullpath, config, crf)

        manifest["preview_fullpath"] = preview_fullpath
        manifest["preview_filename"] = os.path.basename(preview_fullpath)
        manifest["last_format"] = format_name
        manifest["last_crf"] = int(crf)
        manifest["last_extension"] = str(config.get("extension", "mp4")).lstrip(".")
        _write_manifest(token, manifest)

        preview_payload = _build_preview_payload(
            token=token,
            preview_fullpath=preview_fullpath,
            subfolder=str(manifest.get("preview_subfolder") or ""),
            frame_rate=manifest.get("frame_rate", 24),
            has_audio=bool(manifest.get("has_audio", False)),
            format_name=format_name,
            crf=int(crf),
            preview_style=config.get("preview_style", {}),
        )
        preview_payload["written_frames"] = int(manifest.get("written_frames", 0))
        return web.json_response({"ok": True, "preview": preview_payload})
    except Exception as e:
        return web.json_response({"ok": False, "error": str(e)}, status=500)


if PromptServer is not None and web is not None:
    try:
        PromptServer.instance.routes.post("/nf_video_preview/rebuild")(_handle_preview_rebuild)
    except Exception:
        pass
